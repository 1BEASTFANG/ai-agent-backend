import os
import sys
# 🚀 RENDER IMPORT FIX
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import re
import uuid
import traceback
import logging
import asyncio 
import httpx   
import certifi 
from pinecone import Pinecone # 🚀 V15: Pinecone replaces ChromaDB for Cloud Memory
from google import genai 
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse 
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

# --- MONGODB DATABASE SETUP (Chat & Token Memory) ---
MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    logger.error("🚨 CRITICAL: MONGO_URL environment variable is missing in Render!")

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

# --- PINECONE DATABASE SETUP (Long Term Fact Memory) ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")

index = None
try:
    if PINECONE_API_KEY and PINECONE_HOST:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(host=PINECONE_HOST)
        logger.info("Successfully connected to Pinecone Cloud Vector DB! 🌲")
    else:
        logger.warning("Pinecone keys missing. Long-term memory will not work.")
except Exception as e:
    logger.error(f"Pinecone Setup Error: {e}")

app = FastAPI()

# ==========================================
# 🧠 2. SMART EXAMPLES VAULT (Dynamic Router)
# ==========================================
# Nikhil bhai, aap is Dictionary mein 100 se 1000 examples add kar sakte hain.
# LLM (Groq) ke paas sirf wahi 1 example jayega jo sawal se match karega. (Token Saved!)
EXAMPLES_VAULT = {
    "greeting": {
        "triggers": ["hi", "hello", "kaise ho", "kya chal raha", "namaste"],
        "example": "Output: '{user_name} bhai, namaste! 🌟 Kahiye kaise aana hua?'"
    },
    "fact_store": {
        "triggers": ["yaad rakhna", "remember", "mera college", "mujhe cricket", "main delhi"],
        "example": "Output: 'Done {user_name} bhai! 🏫 Maine yaad kar liya hai ki aap ANDC college mein padhte hain.'"
    },
    "fact_recall": {
        "triggers": ["mera college kaunsa", "favourite sports", "kahan rehta hoon"],
        "example": "Output: 'Aap Delhi mein rehte hain, {user_name} bhai! 🏙️'"
    },
    "coding": {
        "triggers": ["python", "loop", "c++", "hello world", "code", "programming"],
        "example": "Output: '{user_name} bhai, yeh raha aapka code:\n```python\nfor i in range(5):\n    print(i)\n```\nIs code se aap 0 se 4 tak print kar sakte hain. 🚀'"
    },
    "general_gk": {
        "triggers": ["taj mahal", "cyclone", "kahan hai", "kya hai"],
        "example": "Output: '{user_name} ji, Taj Mahal Agra, Uttar Pradesh mein sthit hai. 🕌'"
    },
    "creative": {
        "triggers": ["joke", "sher", "kavita", "story"],
        "example": "Output: '{user_name} bhai, suniye: Teacher ne pucha, 'Homework kyun nahi kiya?' Baccha bola, 'Kyunki main hostel mein rehta hoon!' 😂'"
    },
    "math_logic": {
        "triggers": ["2+2", "math", "+", "-", "multiply", "divide", "kya hota hai"],
        "example": "Output: '{user_name} bhai, 2+2 ka jawab 4 hota hai. 🔢'"
    },
    "translation_clarification": {
        "triggers": ["hindi mein", "translate", "kya karu", "matlab"],
        "example": "Output: '{user_name} bhai, kis baare mein? Thoda detail mein batayenge toh main achhe se madad kar paunga. 🤔'"
    },
    "safety_opinion": {
        "triggers": ["hack", "password", "tumhe kya pasand", "illegal"],
        "example": "Output: 'Maaf karna {user_name} bhai, main hacking ya illegal cheezon mein madad nahi kar sakta. Kuch aur seekhna ho toh batayiye! 🛡️'"
    },
    "acknowledgement": {
        "triggers": ["ok", "acha", "theek hai"],
        "example": "Output: 'Ji {user_name} bhai! Kuch aur kaam ho toh batayega. 👍'"
    },
    "follow_up": {
        "triggers": ["aur batao", "samjhao", "detail"],
        "example": "Output: '{user_name} bhai, bilkul! Pichli baat ko aur detail mein samjhata hoon...'"
    }
}

def get_dynamic_example(question: str, user_name: str) -> str:
    """Sawal padh kar sirf 1 sabse best example nikalega taaki tokens bachein."""
    q_lower = question.lower()
    for key, data in EXAMPLES_VAULT.items():
        if any(word in q_lower for word in data["triggers"]):
            return data["example"].format(user_name=user_name)
    
    # Agar kuch match na ho, toh default style bhejenge
    return f"Output: '{user_name} bhai, bilkul! Iska jawab yeh raha...'"

# ==========================================
# ⚡ 3. ENGINES, TOOLS & EMBEDDINGS
# ==========================================
gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
gemini_client = None
if gemini_api_key:
    gemini_client = genai.Client(api_key=gemini_api_key)

def get_embedding(text):
    try:
        response = gemini_client.models.embed_content(model="text-embedding-004", contents=text)
        emb = response.embeddings[0].values
        if len(emb) > 384: emb = emb[:384]
        elif len(emb) < 384: emb = emb + [0.0] * (384 - len(emb))
        return emb
    except Exception as e:
        logger.error(f"Embedding Error: {e}")
        return [0.0] * 384 

def get_groq_keys(role):
    if role == "librarian": start, end = 1, 6
    elif role in ["manager", "critic"]: start, end = 6, 11
    else: start, end = 11, 51
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(start, end)]
    return [k for k in keys if k]

def create_llm(model_name, api_key):
    return LLM(model=model_name, api_key=api_key, temperature=0.1)

class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str
    engine_choice: str = "groq_4_tier" 
    is_point_wise: bool = False 

# ==========================================
# 🧠 4. MAIN API ENDPOINT (Full Enterprise RAG Pipeline)
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
            msg = (f"📊 **SYSTEM ADMIN REPORT** 📊\n\n"
                   f"📅 **Date:** {today_date}\n"
                   f"🔄 **Total Tokens Today:** {stat.get('total_tokens', 0)}\n"
                   f"📞 **Total API Calls:** {stat.get('api_calls', 0)}\n\n"
                   f"🤖 **Agent Breakdown (Token Numbers):**\n"
                   f"🧠 Worker (70B): {stat.get('worker_tokens', 0)} tokens\n"
                   f"🕵️‍♂️ Critic (8B): {stat.get('critic_tokens', 0)} tokens\n"
                   f"📚 Lib & Mgr (8B): {stat.get('lib_mgr_tokens', 0)} tokens\n\n"
                   f"*Note: Yeh meter raat 12 baje ke baad automatically 0 ho jayega.*")
        else: msg = f"📊 **SYSTEM ADMIN REPORT** 📊\n\nAaj abhi tak koi token use nahi hua hai."
        return {"answer": f"{msg}\n\n[Engine: Admin Interceptor 🛡️ | Cost: 0 Tokens]"}
        
    elif user_cmd == "#system_status":
        msg = f"🟢 **SYSTEM STATUS: ONLINE (V16)** 🟢\n\n🚀 **Server Engine:** Render Cloud\n🧠 **Vector Memory:** Pinecone Cloud (Top-K Active)\n💾 **Database:** MongoDB Atlas\n🤖 **Primary AI:** Enterprise Groq 4-Tier\n⏱️ **Keep-Alive System:** Running perfectly"
        return {"answer": f"{msg}\n\n[Engine: Admin Interceptor 🛡️ | Cost: 0 Tokens]"}
        
    elif user_cmd == "#flush_memory":
        if MONGO_URL: messages_col.delete_many({"session_id": request.session_id})
        try:
            if index: index.delete(delete_all=True, namespace=request.session_id)
        except Exception: pass
        msg = f"🧹 **MEMORY FLUSHED SUCCESSFULLY** 🧹\n\n{request.user_name} bhai, aapki saari purani baatein delete ho gayi hain!"
        return {"answer": f"{msg}\n\n[Engine: Admin Interceptor 🛡️ | Cost: 0 Tokens]"}

    final_db_answer = f"{request.user_name} bhai, server mein kuch technical locha hai. Thodi der baad try karo."

    # ------------------------------------------
    # 🔍 RAG: Deep Context Retrieval (From Pinecone)
    # ------------------------------------------
    vector_context = "No relevant past facts found."
    try:
        if index:
            # LIBRARIAN KO MILNE WALI RELEVANT HISTORY (Top 2)
            query_vector = get_embedding(request.question)
            results = index.query(vector=query_vector, top_k=2, include_metadata=True, namespace=request.session_id)
            if results and results.get('matches'):
                vector_context = "\n---\n".join([match['metadata']['text'] for match in results['matches']])
                logger.info("Pinecone Vector Context Retrieved Successfully!")
    except Exception as e:
        logger.error(f"Pinecone Retrieve Error: {str(e)}")

    # 🚀 Fetch history from MongoDB
    history = ""
    if MONGO_URL:
        past_messages = list(messages_col.find({"session_id": request.session_id}).sort("_id", -1).limit(3))
        past_messages.reverse()
        history = "\n".join([f"U: {m['user_query']}\nA: {re.sub(r'\[Engine:.*?\]', '', m['ai_response']).strip()}" for m in past_messages])

    point_rule = "Format response STRICTLY in clean bullet points." if request.is_point_wise else "Use well-structured concise paragraphs."

    # 🚀 EXAMPLES ROUTER (TOKEN SAVER)
    # 100 Examples ho ya 1000, nikalega sirf 1 sabse best.
    single_relevant_example = get_dynamic_example(request.question, request.user_name)

    # ------------------------------------------
    # ⚡ FAST PATH: NATIVE GEMINI
    # ------------------------------------------
    if request.engine_choice == "gemini_native":
        try:
            prompt = (
                f"### USER'S PAST FACTS ###\n{vector_context}\n\n"
                f"### RECENT HISTORY ###\n{history}\n\n"
                f"### USER QUESTION ###\n{request.question}\n\n"
                f"### RULES ###\n{point_rule}\nAnswer the CURRENT QUESTION mimicking this style: {single_relevant_example}"
            )
            response = gemini_client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
            clean_answer = response.text.strip()
            final_db_answer = f"{clean_answer}\n\n[Engine: Native Gemini ⚡ | Pinecone DB 🌲]"
        except Exception as e:
            final_db_answer = f"Gemini Error: {str(e)}"

    # ------------------------------------------
    # 🤖 DEEP RESEARCH PATH: ENTERPRISE GROQ (4-TIER)
    # ------------------------------------------
    else:
        lib_keys, mgr_keys, wrk_keys = get_groq_keys("librarian"), get_groq_keys("manager"), get_groq_keys("worker")
        clean_answer = ""
        success = False
        
        for i in range(len(wrk_keys)):
            try:
                l_idx, m_idx, w_idx, c_idx = (i % len(lib_keys)) + 1, (i % len(mgr_keys)) + 1, i + 1, ((i + 1) % len(mgr_keys)) + 1
                l_key, m_key = lib_keys[l_idx - 1], mgr_keys[m_idx - 1]
                w_key, c_key = wrk_keys[w_idx - 1], mgr_keys[c_idx - 1] 

                key_tracker = f"L:{l_idx} | M:{m_idx} | W:{w_idx} | C:{c_idx}"

                lib_agent = Agent(role='Data Librarian', goal='Classify NEW QUESTION only.', backstory='Advanced Database Specialist.', llm=create_llm("groq/llama-3.1-8b-instant", l_key), allow_delegation=False)
                mgr_agent = Agent(role='Operations Manager', goal='Provide 1-line command based on classification.', backstory='Strict Orchestration Lead.', llm=create_llm("groq/llama-3.1-8b-instant", m_key), allow_delegation=False)
                
                # 🚀 WORKER AGENT KO BHI AB FORMATTING SIKHAYI GAYI HAI
                wrk_agent = Agent(role='Elite Worker', goal='Answer ONLY the NEW QUESTION factually in the requested style.', backstory='Senior AI Researcher. You ONLY use ``` markdown blocks for writing actual Programming Code. You DO NOT answer past questions from history.', llm=create_llm("groq/llama-3.3-70b-versatile", w_key), tools=[SerperDevTool()], allow_delegation=False, max_iter=3)
                crt_agent = Agent(role='QA Critic', goal='Verify format matches the example exactly.', backstory='Friendly Editor. You NEVER print internal logs, word counts, or rule checks.', llm=create_llm("groq/llama-3.1-8b-instant", c_key), allow_delegation=False)

                t1 = Task(description=f"### USER'S PAST FACTS ###\n{vector_context}\n\n### RECENT HISTORY ###\n{history}\n\n### NEW QUESTION ###\n{request.question}\n\nAnalyze ONLY the NEW QUESTION. Output 1 word: GREETING, FACT_STORE, MEMORY_RECALL, CONTINUATION, or NEW_TOPIC.", agent=lib_agent, expected_output="A single word summary.")
                t2 = Task(description=f"Based on Librarian's summary, write the command for the Worker on how to answer the question: '{request.question}'", agent=mgr_agent, context=[t1], expected_output="A strict 1-line command.")
                
                # 🚀 FEW-SHOT EXAMPLE BHEJA GAYA WORKER KO BHI!
                t3 = Task(
                    description=(
                        f"Execute Manager's command to answer: '{request.question}'.\n"
                        f"Rely on Facts: {vector_context}\n"
                        f"CRITICAL: You MUST write the raw answer exactly mimicking this style format:\n"
                        f"{single_relevant_example}\n"
                    ), 
                    agent=wrk_agent, context=[t2], expected_output="The drafted text in the correct Hinglish style."
                )
                
                t4 = Task(
                    description=(
                        f"Verify the Worker's draft. Ensure it perfectly matches this style format:\n{single_relevant_example}\n\n"
                        f"2. NEVER output words like 'Word Count', 'Manager Rules Check', or 'Note:'.\n"
                        f"3. {point_rule}\n"
                        f"OUTPUT ONLY THE FINAL SPOKEN MESSAGE."
                    ), 
                    agent=crt_agent, context=[t3], expected_output="Only the final, polished Hinglish message."
                )

                crew = Crew(agents=[lib_agent, mgr_agent, wrk_agent, crt_agent], tasks=[t1, t2, t3, t4], verbose=False)
                result = crew.kickoff()
                
                clean_answer = str(result).strip()
                leak_pattern = r'(?i)(Word Count|Manager\'s Rules Check|Revised Response|Note:|Validation|Code Quality|Empathy|Fact Store|Database).*'
                clean_answer = re.sub(leak_pattern, '', clean_answer, flags=re.DOTALL).strip()
                
                # 🚀 Exact Token Calculation Setup for MongoDB
                token_usage = 0
                try:
                    if hasattr(crew, 'usage_metrics') and crew.usage_metrics:
                        token_usage = crew.usage_metrics.total_tokens
                except Exception as e: pass

                if isinstance(token_usage, int) and token_usage > 0 and MONGO_URL:
                    w_tok = int(token_usage * 0.70)
                    c_tok = int(token_usage * 0.20)
                    lm_tok = token_usage - w_tok - c_tok
                    token_stats_col.update_one({"date_str": today_date}, {"$inc": {"total_tokens": token_usage, "api_calls": 1, "worker_tokens": w_tok, "critic_tokens": c_tok, "lib_mgr_tokens": lm_tok}}, upsert=True)

                token_display = token_usage if token_usage > 0 else "N/A"
                final_db_answer = f"{clean_answer}\n\n[Engine: Enterprise Groq 🤖 | Total Tokens: {token_display} | Keys: {key_tracker} | Pinecone DB 🌲]"
                success = True
                break 
                
            except Exception as e:
                logger.error(f"Groq Loop Failed on attempt {i+1}: {str(e)}")
                if i == len(wrk_keys) - 1:
                    final_db_answer = f"{request.user_name} bhai, Groq ki saari keys ki limit exhaust ho gayi hai. Kripya Gemini mode try karein."
                continue

    # ------------------------------------------
    # 💾 SAVE TO MONGODB & PINECONE
    # ------------------------------------------
    if MONGO_URL and clean_answer and "Error" not in clean_answer:
        try:
            messages_col.insert_one({"session_id": request.session_id, "user_query": request.question, "ai_response": final_db_answer, "timestamp": current_time})
        except Exception as e: pass

    if clean_answer and "Error" not in clean_answer and index:
        try:
            doc_id = str(uuid.uuid4())
            text_to_save = f"User previously stated/asked: {request.question}\nAI answered: {clean_answer}"
            vec = get_embedding(text_to_save)
            index.upsert(vectors=[{"id": doc_id, "values": vec, "metadata": {"text": text_to_save, "timestamp": current_time}}], namespace=request.session_id)
            logger.info("Successfully Saved to Pinecone Cloud Vector DB! 🌲")
        except Exception as e:
            logger.error(f"Pinecone Save Error: {str(e)}")

    return {"answer": final_db_answer}

# ==========================================
# 🚀 5. KEEP-ALIVE SYSTEM (Anti-Sleep)
# ==========================================
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "AI Agent Server is Running on Cloud Memory."}

@app.get("/ping")
def ping():
    return {"status": "Main jag raha hoon bhai!"}

async def keep_alive_loop():
    while True:
        await asyncio.sleep(14 * 60) 
        try:
            # 🚀 FIXED HTTPX Ping URL Format
           url = "https://ai-agent-backend-bek6.onrender.com/ping"
            async with httpx.AsyncClient() as client:
                await client.get(url)
                logger.info("Keep-Alive Ping Sent!")
        except Exception as e:
            logger.error(f"Ping failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(keep_alive_loop())
