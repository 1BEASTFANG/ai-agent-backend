import os
import sys
# 🚀 RENDER IMPORT FIX
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import re
import uuid
import time
import traceback
import logging
import asyncio 
import httpx   
import certifi 
from pinecone import Pinecone # 🚀 Cloud Memory
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient 

from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

# ==========================================
# 🚀 1. ENTERPRISE SETTINGS & LOGGING
# ==========================================
os.environ["CREWAI_TRACING_ENABLED"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- MONGODB DATABASE SETUP ---
MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    logger.error("🚨 CRITICAL: MONGO_URL environment variable is missing!")

try:
    if MONGO_URL:
        mongo_client = MongoClient(MONGO_URL, tls=True, tlsCAFile=certifi.where(), tlsAllowInvalidCertificates=True)
        mongo_client.admin.command('ping')
        logger.info("Successfully connected to MongoDB Atlas! 🎉")
        
        db_mongo = mongo_client["ai_assistant_db"]
        messages_col = db_mongo["messages"]
        token_stats_col = db_mongo["token_stats_v2"]
except Exception as e:
    logger.error(f"MongoDB Connection Failed: {e}")

# --- PINECONE DATABASE SETUP ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")

index = None
try:
    if PINECONE_API_KEY and PINECONE_HOST:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(host=PINECONE_HOST)
        logger.info("Successfully connected to Pinecone Cloud Vector DB! 🌲")
except Exception as e:
    logger.error(f"Pinecone Setup Error: {e}")

app = FastAPI()

# ==========================================
# ⚡ 2. ENGINES, TOOLS & EMBEDDINGS
# ==========================================
HF_API_KEY = os.getenv("HF_API_KEY", "").strip()

# 🚀 Hugging Face Embeddings (100% Free, Perfect 384 Dimensions)
def get_embedding(text):
    api_url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        response = httpx.post(api_url, headers=headers, json={"inputs": text}, timeout=30.0)
        if response.status_code == 200:
            data = response.json()
            # Handle both flat lists and nested lists returned by HF
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list): return data[0]
                return data
        return [0.0] * 384 
    except Exception as e:
        logger.error(f"HF Embedding Error: {e}")
        return [0.0] * 384

def get_groq_keys(role):
    if role == "librarian": start, end = 1, 6
    elif role in ["manager", "critic"]: start, end = 6, 11
    else: start, end = 11, 51
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(start, end)]
    return [k for k in keys if k]

def create_llm(model_name, api_key):
    return LLM(model=model_name, api_key=api_key, temperature=0.1)

# 🚀 CLOUD ROUTER (Pinecone se 1 relevant example layega)
def get_dynamic_example(question: str, user_name: str) -> str:
    try:
        if index:
            query_vec = get_embedding(question)
            res = index.query(vector=query_vec, top_k=1, include_metadata=True, namespace="few-shot-examples")
            if res and res.get('matches'):
                best_match = res['matches'][0]['metadata']['template']
                return best_match.format(q=question, user_name=user_name)
    except Exception as e:
        logger.error(f"Dynamic Example Fetch Error: {e}")
        
    return f"Output: 'Ji {user_name} bhai! Iska jawab yeh raha...'"

class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str
    engine_choice: str = "groq_4_tier" 
    is_point_wise: bool = False 

# ==========================================
# 🧠 3. MAIN API ENDPOINT
# ==========================================
@app.post("/ask")
def ask_ai(request: UserRequest):
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    today_date = datetime.now().strftime("%Y-%m-%d")
    user_cmd = request.question.strip().lower()
    
    # --- ADMIN COMMANDS ---
    if user_cmd == "#total_tokens":
        if MONGO_URL: stat = token_stats_col.find_one({"date_str": today_date})
        else: stat = None
            
        if stat:
            msg = (f"📊 **SYSTEM ADMIN REPORT** 📊\n\n📅 **Date:** {today_date}\n"
                   f"🔄 **Total Tokens Today:** {stat.get('total_tokens', 0)}\n"
                   f"📞 **Total API Calls:** {stat.get('api_calls', 0)}\n"
                   f"🧠 Worker: {stat.get('worker_tokens', 0)} | 🕵️‍♂️ Critic: {stat.get('critic_tokens', 0)}")
        else: msg = "Aaj abhi tak koi token use nahi hua hai."
        return {"answer": f"{msg}\n\n[Engine: Admin Interceptor 🛡️]"}
        
    elif user_cmd == "#system_status":
        return {"answer": f"🟢 **SYSTEM STATUS: ONLINE (V16.1 Cloud Brain)** 🟢\n\n🚀 Server: Active\n🧠 Memory: Pinecone (HF Embeddings)\n💾 DB: MongoDB\n[Engine: Admin Interceptor 🛡️]"}
        
    elif user_cmd == "#flush_memory":
        if MONGO_URL: messages_col.delete_many({"session_id": request.session_id})
        try:
            if index: index.delete(delete_all=True, namespace=request.session_id)
        except Exception: pass
        return {"answer": f"🧹 **MEMORY FLUSHED** 🧹\n\n{request.user_name} bhai, saari yaadein delete ho gayi hain!"}

    # ------------------------------------------
    # 🔍 RAG: Deep Context Retrieval
    # ------------------------------------------
    vector_context = "No relevant past facts found."
    try:
        if index:
            query_vector = get_embedding(request.question)
            results = index.query(vector=query_vector, top_k=2, include_metadata=True, namespace=request.session_id)
            if results and results.get('matches'):
                vector_context = "\n---\n".join([match['metadata']['text'] for match in results['matches']])
    except Exception as e: pass

    # 🚀 Fetch history from MongoDB
    history = ""
    if MONGO_URL:
        past_messages = list(messages_col.find({"session_id": request.session_id}).sort("_id", -1).limit(3))
        history = "\n".join([f"U: {m['user_query']}\nA: {re.sub(r'\[Engine:.*?\]', '', m['ai_response']).strip()}" for m in reversed(past_messages)])

    point_rule = "Format response STRICTLY in clean bullet points." if request.is_point_wise else "Use well-structured concise paragraphs."

    # 🚀 THE MAGIC: Fetch 1 Relevant Example from Cloud
    single_relevant_example = get_dynamic_example(request.question, request.user_name)

    # ------------------------------------------
    # ⚡ FAST PATH (Groq 8B Fast Model)
    # ------------------------------------------
    if request.engine_choice == "gemini_native":
        try:
            fast_key = get_groq_keys("critic")[0] 
            fast_llm = create_llm("groq/llama-3.1-8b-instant", fast_key)
            prompt = (f"Facts: {vector_context}\nHistory: {history}\nQ: {request.question}\n"
                      f"Answer briefly mimicking this style:\n{single_relevant_example}")
            response = fast_llm.call(prompt)
            final_db_answer = f"{response}\n\n[Engine: Fast Groq ⚡ | Pinecone DB 🌲]"
        except Exception as e:
            final_db_answer = f"Fast Engine Error: {str(e)}"

    # ------------------------------------------
    # 🤖 DEEP RESEARCH PATH: ENTERPRISE GROQ (4-TIER)
    # ------------------------------------------
    else:
        lib_keys, mgr_keys, wrk_keys = get_groq_keys("librarian"), get_groq_keys("manager"), get_groq_keys("worker")
        clean_answer = ""
        success = False
        
        for i in range(len(wrk_keys)):
            try:
                l_key, m_key = lib_keys[i % len(lib_keys)], mgr_keys[i % len(mgr_keys)]
                w_key, c_key = wrk_keys[i], mgr_keys[(i + 1) % len(mgr_keys)]
                key_tracker = f"L:{i%len(lib_keys)+1} | M:{i%len(mgr_keys)+1} | W:{i+1} | C:{(i+1)%len(mgr_keys)+1}"

                lib_agent = Agent(role='Librarian', goal='Classify QUESTION.', backstory='DB Specialist.', llm=create_llm("groq/llama-3.1-8b-instant", l_key), allow_delegation=False)
                mgr_agent = Agent(role='Manager', goal='Provide command.', backstory='Lead.', llm=create_llm("groq/llama-3.1-8b-instant", m_key), allow_delegation=False)
                wrk_agent = Agent(role='Worker', goal='Answer ONLY the NEW QUESTION factually.', backstory='Senior AI Researcher. You ONLY use markdown blocks for code.', llm=create_llm("groq/llama-3.3-70b-versatile", w_key), tools=[SerperDevTool()], allow_delegation=False, max_iter=3)
                crt_agent = Agent(role='Critic', goal='Verify format matches example exactly.', backstory='Editor. You NEVER print internal logs.', llm=create_llm("groq/llama-3.1-8b-instant", c_key), allow_delegation=False)

                t1 = Task(description=f"Facts: {vector_context}\nQ: {request.question}\nAnalyze ONLY Q. Output 1 word: GREETING, FACT_STORE, MEMORY_RECALL, CONTINUATION, or NEW_TOPIC.", agent=lib_agent, expected_output="A single word summary.")
                t2 = Task(description=f"Based on classification, command Worker how to answer: '{request.question}'", agent=mgr_agent, context=[t1], expected_output="1-line command.")
                
                # 🚀 WORKER GETS THE EXAMPLE!
                t3 = Task(
                    description=(f"Execute Manager command to answer: '{request.question}'.\nFacts: {vector_context}\n"
                                 f"CRITICAL: Write raw answer EXACTLY mimicking this style format:\n{single_relevant_example}\n"), 
                    agent=wrk_agent, context=[t2], expected_output="Drafted text in correct style."
                )
                
                t4 = Task(
                    description=(f"Verify Worker's draft matches this style:\n{single_relevant_example}\n"
                                 f"2. NEVER output words like 'Word Count' or 'Note:'.\n3. {point_rule}\nOUTPUT ONLY FINAL MESSAGE."), 
                    agent=crt_agent, context=[t3], expected_output="Polished message."
                )

                crew = Crew(agents=[lib_agent, mgr_agent, wrk_agent, crt_agent], tasks=[t1, t2, t3, t4], verbose=False)
                result = crew.kickoff()
                
                clean_answer = re.sub(r'(?i)(Word Count|Manager\'s Rules Check|Revised Response|Note:|Validation|Code Quality|Empathy|Fact Store|Database).*', '', str(result), flags=re.DOTALL).strip()
                
                # 🚀 TOKEN TRACKING
                token_usage = 0
                try:
                    if hasattr(crew, 'usage_metrics') and crew.usage_metrics:
                        token_usage = crew.usage_metrics.total_tokens
                except Exception: pass

                if token_usage > 0 and MONGO_URL:
                    w_tok, c_tok = int(token_usage * 0.70), int(token_usage * 0.20)
                    token_stats_col.update_one({"date_str": today_date}, {"$inc": {"total_tokens": token_usage, "api_calls": 1, "worker_tokens": w_tok, "critic_tokens": c_tok, "lib_mgr_tokens": token_usage - w_tok - c_tok}}, upsert=True)

                token_display = token_usage if token_usage > 0 else "N/A"
                final_db_answer = f"{clean_answer}\n\n[Engine: Groq 🤖 | Tokens: {token_display} | V16.1 Cloud Brain 🌲]"
                success = True
                break 
                
            except Exception as e:
                logger.error(f"Groq Loop Failed: {str(e)}")
                if i == len(wrk_keys) - 1: final_db_answer = f"{request.user_name} bhai, Groq keys exhaust ho gayi hain."

    # ------------------------------------------
    # 💾 SAVE TO MONGODB & PINECONE
    # ------------------------------------------
    if MONGO_URL and clean_answer and "Error" not in clean_answer:
        try: messages_col.insert_one({"session_id": request.session_id, "user_query": request.question, "ai_response": final_db_answer, "timestamp": current_time})
        except: pass

    if clean_answer and "Error" not in clean_answer and index:
        try:
            vec = get_embedding(f"Q: {request.question} A: {clean_answer}")
            index.upsert(vectors=[{"id": str(uuid.uuid4()), "values": vec, "metadata": {"text": f"User: {request.question}\nAI: {clean_answer}", "timestamp": current_time}}], namespace=request.session_id)
        except Exception as e: logger.error(f"Pinecone Save Error: {str(e)}")

    return {"answer": final_db_answer}

# ==========================================
# 🚀 5. KEEP-ALIVE SYSTEM (Anti-Sleep)
# ==========================================
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "V16 Server Active"}

@app.get("/ping")
def ping():
    return {"status": "Main jag raha hoon bhai!"}

async def keep_alive_loop():
    while True:
        await asyncio.sleep(14 * 60) 
        try:
            # 🚀 Cleaned URL, no markdown brackets
            async with httpx.AsyncClient() as client:
                await client.get("https://ai-agent-backend-bek6.onrender.com/ping")
        except: pass

@app.on_event("startup")
async def startup_event(): asyncio.create_task(keep_alive_loop())
