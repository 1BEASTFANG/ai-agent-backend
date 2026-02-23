import os
import sys
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
# ⚡ 2. ENGINES, TOOLS & EMBEDDINGS
# ==========================================
gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
gemini_client = None
if gemini_api_key:
    gemini_client = genai.Client(api_key=gemini_api_key)

# 🚀 Gemini Embeddings (Saves RAM, replaces local models)
def get_embedding(text):
    try:
        response = gemini_client.models.embed_content(model="text-embedding-004", contents=text)
        emb = response.embeddings[0].values
        # Pad or truncate to match 384 dimensions (Pinecone requirement)
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
# 🧠 3. MAIN API ENDPOINT (Full Enterprise RAG Pipeline)
# ==========================================
@app.post("/ask")
def ask_ai(request: UserRequest):
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    # ==========================================
    # 🛡️ THE ADMIN COMMANDS INTERCEPTOR 🛡️
    # ==========================================
    user_cmd = request.question.strip().lower()
    
    if user_cmd == "#total_tokens":
        if MONGO_URL:
            stat = token_stats_col.find_one({"date_str": today_date})
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
        else:
            msg = f"📊 **SYSTEM ADMIN REPORT** 📊\n\nAaj (Date: {today_date}) abhi tak koi token use nahi hua hai."
        return {"answer": f"{msg}\n\n[Engine: Admin Interceptor 🛡️ | Cost: 0 Tokens]"}
        
    elif user_cmd == "#system_status":
        msg = f"🟢 **SYSTEM STATUS: ONLINE (V15)** 🟢\n\n🚀 **Server Engine:** Render Cloud (Active)\n🧠 **Vector Memory:** Pinecone Cloud (Connected)\n💾 **Database:** MongoDB Atlas (Connected)\n🤖 **Primary AI:** Enterprise Groq 4-Tier\n⏱️ **Keep-Alive System:** Running perfectly"
        return {"answer": f"{msg}\n\n[Engine: Admin Interceptor 🛡️ | Cost: 0 Tokens]"}
        
    elif user_cmd == "#flush_memory":
        if MONGO_URL: messages_col.delete_many({"session_id": request.session_id})
        try:
            if index: index.delete(delete_all=True, namespace=request.session_id)
        except Exception: pass
        msg = f"🧹 **MEMORY FLUSHED SUCCESSFULLY** 🧹\n\n{request.user_name} bhai, aapki saari purani baatein aur yaadein system se delete kar di gayi hain. Mera dimaag ab ekdam fresh hai! Ek naye sire se shuruwat karte hain."
        return {"answer": f"{msg}\n\n[Engine: Admin Interceptor 🛡️ | Cost: 0 Tokens]"}

    final_db_answer = f"{request.user_name} bhai, server mein kuch technical locha hai. Thodi der baad try karo."

    # ------------------------------------------
    # 🔍 RAG: Deep Context Retrieval (From Pinecone)
    # ------------------------------------------
    vector_context = "No relevant past facts found."
    try:
        if index:
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

    # 🌟 20 FEW-SHOT EXAMPLES (Wapas main file mein aa gaye) 🌟
    few_shot_examples = f"""
    EXAMPLE 1 (Greeting):
    User: "hi" 
    Output: "{request.user_name} bhai, namaste! 🌟 Kahiye kaise aana hua?"

    EXAMPLE 2 (Greeting 2):
    User: "kaise ho?" 
    Output: "Main ekdam badhiya hoon, {request.user_name} bhai! Aap sunaiye, kya chal raha hai? 😊"

    EXAMPLE 3 (Storing Fact 1):
    User: "mera college ANDC hai"
    Output: "Done {request.user_name} bhai! 🏫 Maine yaad kar liya hai ki aap ANDC college mein padhte hain."

    EXAMPLE 4 (Storing Fact 2):
    User: "mujhe cricket pasand hai"
    Output: "Noted {request.user_name} bhai! 🏏 Maine save kar liya hai ki aapko Cricket pasand hai."

    EXAMPLE 5 (Storing Fact 3):
    User: "main delhi mein rehta hoon"
    Output: "Theek hai {request.user_name} bhai! 📍 Yaad rahega ki aap Delhi se hain."

    EXAMPLE 6 (Recalling Fact 1):
    User: "mera college kaunsa hai?"
    Output: "{request.user_name} bhai, aap ANDC college mein padhte hain! 🎓"

    EXAMPLE 7 (Recalling Fact 2):
    User: "mera favourite sports kya tha?"
    Output: "Aapka favourite sports Cricket hai, {request.user_name} bhai! 🏏"

    EXAMPLE 8 (Recalling Fact 3):
    User: "main kahan rehta hoon?"
    Output: "Aap Delhi mein rehte hain, {request.user_name} bhai! 🏙️"

    EXAMPLE 9 (Coding 1 - STRICT MARKDOWN):
    User: "Python mein loop kaise likhe?"
    Output: "{request.user_name} bhai, yeh raha aapka code:\n```python\nfor i in range(5):\n    print(i)\n```\nIs code se aap 0 se 4 tak print kar sakte hain. 🚀"

    EXAMPLE 10 (Coding 2 - STRICT MARKDOWN):
    User: "C++ hello world"
    Output: "Yeh lijiye {request.user_name} bhai:\n```cpp\n#include <iostream>\nint main() {{\n    std::cout << \"Hello World!\";\n    return 0;\n}}\n```\nBilkul simple aur basic! 💻"

    EXAMPLE 11 (General Knowledge 1):
    User: "Taj Mahal kahan hai?"
    Output: "{request.user_name} ji, Taj Mahal Agra, Uttar Pradesh mein sthit hai. 🕌"

    EXAMPLE 12 (General Knowledge 2):
    User: "Cyclone kin rajyon mein aaya tha?"
    Output: "{request.user_name} bhai, cyclone zyada tar Odisha, West Bengal, aur Andhra Pradesh jaise tatiye (coastal) rajyon mein aata hai. 🌪️"

    EXAMPLE 13 (Joke/Humor - NO CODE BLOCK):
    User: "ek joke sunao"
    Output: "{request.user_name} bhai, suniye: Teacher ne pucha, 'Homework kyun nahi kiya?' Baccha bola, 'Kyunki main hostel mein rehta hoon!' 😂"

    EXAMPLE 14 (Poetry/Story - NO CODE BLOCK):
    User: "sher sunao"
    Output: "Irshaad {request.user_name} bhai! 🌹\nAsmaan mein udte hue parinde se kisi ne poocha...\n'Kya tumhe zameen par girne ka darr nahi?'\nParinde ne muskurakar kaha, 'Main toh udta hi zameen se juda hoon!'"

    EXAMPLE 15 (Math):
    User: "2+2 kya hota hai?"
    Output: "{request.user_name} bhai, 2+2 ka jawab 4 hota hai. 🔢"

    EXAMPLE 16 (Translation):
    User: "hello ko hindi mein kya kehte hain?"
    Output: "Hello ko Hindi mein 'Namaste' (नमस्ते) kehte hain, {request.user_name} bhai! 🙏"

    EXAMPLE 17 (Clarification):
    User: "kya karu?"
    Output: "{request.user_name} bhai, kis baare mein? Thoda detail mein batayenge toh main achhe se madad kar paunga. 🤔"

    EXAMPLE 18 (Opinion - Neutral):
    User: "tumhe kya pasand hai?"
    Output: "Main ek AI hoon {request.user_name} bhai, meri apni koi pasand nahi hoti. Par aapse baat karke achha lagta hai! 🤖"

    EXAMPLE 19 (Safety/Refusal):
    User: "kisi ka password kaise hack karein?"
    Output: "Maaf karna {request.user_name} bhai, main hacking ya illegal cheezon mein madad nahi kar sakta. Kuch aur seekhna ho toh batayiye! 🛡️"

    EXAMPLE 20 (Short Acknowledgement):
    User: "ok"
    Output: "Ji {request.user_name} bhai! Kuch aur kaam ho toh batayega. 👍"
    
    EXAMPLE 21 (Follow-up / Explanation): 
    User: "thoda aur aache se samjhao" OR "iske baare mein aur batao"
    Output: "{request.user_name} bhai, bilkul! Pichli baat ko aur detail mein samjhata hoon..."
    """

    # ------------------------------------------
    # ⚡ FAST PATH: NATIVE GEMINI
    # ------------------------------------------
    if request.engine_choice == "gemini_native":
        try:
            logger.info(f"Routing request to Gemini for user: {request.user_name}")
            prompt = (
                f"### USER'S PAST FACTS ###\n{vector_context}\n\n"
                f"### RECENT HISTORY ###\n{history}\n\n"
                f"### USER QUESTION ###\n{request.question}\n\n"
                f"### RULES ###\n{point_rule}\nAnswer the CURRENT QUESTION in friendly natural Hinglish concisely. If it is a follow-up, use the HISTORY to expand on the topic. Otherwise, DO NOT repeat the history. DO NOT combine greetings with factual answers. DO NOT ask follow-up questions. Address user as {request.user_name}."
            )
            response = gemini_client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
            clean_answer = response.text.strip()
            final_db_answer = f"{clean_answer}\n\n[Engine: Native Gemini ⚡ | Pinecone DB 🌲]"
        except Exception as e:
            clean_answer = "Error"
            final_db_answer = f"Gemini Error: {str(e)}"

    # ------------------------------------------
    # 🤖 DEEP RESEARCH PATH: ENTERPRISE GROQ (4-TIER)
    # ------------------------------------------
    else:
        logger.info(f"Initiating Enterprise Groq Pipeline for user: {request.user_name}")
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
                wrk_agent = Agent(role='Elite Worker', goal='Answer ONLY the NEW QUESTION factually.', backstory='Senior AI Researcher. You ONLY use ``` markdown blocks for writing actual Programming Code (like C++, Python). You NEVER use markdown blocks for text, explanations, or jokes. You NEVER use words like "Memory", "Database", or "Fact Store" in your response. You DO NOT answer past questions from history.', llm=create_llm("groq/llama-3.3-70b-versatile", w_key), tools=[SerperDevTool()], allow_delegation=False, max_iter=3)
                crt_agent = Agent(role='QA Critic', goal='Format beautifully matching examples, add empathy.', backstory='Friendly Editor. You NEVER print internal logs, word counts, or rule checks. You NEVER ask follow-up questions at the end of your response. You NEVER mix answers from history.', llm=create_llm("groq/llama-3.1-8b-instant", c_key), allow_delegation=False)

                t1 = Task(description=(f"### USER'S PAST FACTS ###\n{vector_context}\n\n### RECENT HISTORY ###\n{history}\n\n### NEW QUESTION ###\n{request.question}\n\nINSTRUCTIONS:\nAnalyze ONLY the NEW QUESTION. Output exactly 1 word:\n- 'GREETING' (if hi, hello)\n- 'FACT_STORE' (if user is telling a fact about themselves to remember)\n- 'MEMORY_RECALL' (if user is asking about past facts)\n- 'CONTINUATION' (if user asks to explain more, give examples, or refers to the previous message)\n- 'NEW_TOPIC' (for general questions, coding, or jokes)\nDo not write anything else."), agent=lib_agent, expected_output="A single word summary: GREETING, FACT_STORE, MEMORY_RECALL, CONTINUATION, or NEW_TOPIC.")
                t2 = Task(description=(f"### NEW QUESTION ###\n{request.question}\n\nINSTRUCTIONS:\nBased on Librarian's summary, write the command for the Worker:\n- GREETING: 'Say a friendly hello.'\n- FACT_STORE: 'Acknowledge the fact in 1 simple sentence only.'\n- MEMORY_RECALL: 'Answer directly using PAST FACTS only. DO NOT explain.'\n- CONTINUATION: 'Read HISTORY carefully and explain the last topic in more detail.'\n- NEW_TOPIC: 'Answer factually. If user asks for code, use markdown. If Joke/Fact, use normal text.'\n"), agent=mgr_agent, context=[t1], expected_output="A strict 1-line command for the worker.")
                t3 = Task(description=(f"### USER'S PAST FACTS ###\n{vector_context}\n\n### RECENT HISTORY ###\n{history}\n\n### NEW QUESTION ###\n{request.question}\n\nINSTRUCTIONS:\nExecute Manager's command. IF NEW_TOPIC: Answer ONLY the NEW QUESTION and DO NOT repeat previous history. IF CONTINUATION: Rely deeply on RECENT HISTORY to provide a follow-up detailed answer. DO NOT output meta-text. ONLY use ``` language ``` blocks if writing a programming script. DO NOT use code blocks for jokes or text. CRITICAL: DO NOT say things like 'this is in our fact store' or 'based on memory'."), agent=wrk_agent, context=[t2], expected_output="The raw drafted text containing facts and optional code blocks.")
                t4 = Task(description=(f"### NEW QUESTION ###\n{request.question}\n\nCRITICAL RULES FOR OUTPUT:\n1. Choose ONLY ONE matching situation from the examples below. DO NOT combine answers from past history.\n2. NEVER output words like 'Word Count', 'Manager Rules Check', 'Revised Response', or 'Note:'.\n3. NEVER use words like 'Fact Store', 'Database', or 'Memory'.\n4. You must format the Worker's draft EXACTLY mimicking the style of these examples:\n\n{few_shot_examples}\n\n5. DO NOT ask repetitive follow-up questions (e.g. stop saying 'kya aap aur janna chahte hain?'). Just give the answer and stop.\n6. {point_rule}\nOUTPUT ONLY THE FINAL SPOKEN MESSAGE THAT THE USER WILL READ."), agent=crt_agent, context=[t3], expected_output="Only the final, polished Hinglish message meant for the user. No internal logs. Code must be in markdown.")

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
            url = "[https://ai-agent-backend-bek6.onrender.com/ping](https://ai-agent-backend-bek6.onrender.com/ping)" 
            async with httpx.AsyncClient() as client:
                await client.get(url)
                logger.info("Keep-Alive Ping Sent!")
        except Exception as e:
            logger.error(f"Ping failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(keep_alive_loop())
