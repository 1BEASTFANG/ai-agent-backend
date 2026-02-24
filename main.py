import os
import sys
# 🚀 RENDER IMPORT FIX
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import re
import uuid
import time
import json
import logging
import asyncio 
import httpx   
import certifi 
from pinecone import Pinecone
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient 

# ==========================================
# 🚀 1. ENTERPRISE SETTINGS & LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- MONGODB DATABASE SETUP ---
MONGO_URL = os.getenv("MONGO_URL")
try:
    if MONGO_URL:
        mongo_client = MongoClient(MONGO_URL, tls=True, tlsCAFile=certifi.where(), tlsAllowInvalidCertificates=True)
        db_mongo = mongo_client["ai_assistant_db"]
        messages_col = db_mongo["messages"]
        token_stats_col = db_mongo["token_stats_v2"]
        logger.info("Successfully connected to MongoDB Atlas! 🎉")
except Exception as e: logger.error(f"MongoDB Connection Failed: {e}")

# --- PINECONE DATABASE SETUP ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
index = None
try:
    if PINECONE_API_KEY and PINECONE_HOST:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(host=PINECONE_HOST)
        logger.info("Successfully connected to Pinecone Cloud! 🌲")
except Exception as e: logger.error(f"Pinecone Setup Error: {e}")

app = FastAPI()

# ==========================================
# ⚡ 2. CORE TOOLS (Embeddings & Web Search)
# ==========================================
HF_API_KEY = os.getenv("HF_API_KEY", "").strip()

def get_embedding(text):
    api_url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        response = httpx.post(api_url, headers=headers, json={"inputs": text}, timeout=15.0)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0] if isinstance(data[0], list) else data
        return [0.0] * 384 
    except Exception: return [0.0] * 384

# 🚀 CUSTOM WEB SEARCH (Replaces CrewAI Serper Tool)
def search_web(query):
    serper_key = os.getenv("SERPER_API_KEY", "").strip()
    if not serper_key: return "No internet access (API Key missing)."
    try:
        res = httpx.post("https://google.serper.dev/search", headers={"X-API-KEY": serper_key}, json={"q": query}, timeout=10.0)
        if res.status_code == 200:
            results = res.json().get("organic", [])
            return "\n".join([f"- {r.get('title')}: {r.get('snippet')}" for r in results[:3]])
    except Exception: return "Search failed."
    return "No recent data found."

# 🚀 PINECONE ROUTER
def get_dynamic_example(question: str, user_name: str) -> str:
    try:
        if index:
            res = index.query(vector=get_embedding(question), top_k=1, include_metadata=True, namespace="few-shot-examples")
            if res and res.get('matches'):
                return res['matches'][0]['metadata']['template'].format(q=question, user_name=user_name)
    except Exception: pass
    return f"Output: 'Ji {user_name} bhai! Iska jawab yeh raha...'"

# ==========================================
# 🧠 3. DIRECT GROQ API ENGINE (Zero CrewAI Overhead)
# ==========================================
def get_groq_keys(role):
    start, end = (1, 6) if role == "librarian" else ((6, 11) if role in ["manager", "critic"] else (11, 51))
    return [k for k in [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(start, end)] if k]

def direct_groq_call(prompt, role, keys):
    model_name = "llama-3.3-70b-versatile" if role == "worker" else "llama-3.1-8b-instant"
    for key in keys:
        try:
            res = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=30.0
            )
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"], data["usage"]["total_tokens"]
        except Exception: continue
    return "Error: System busy", 0

class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str
    engine_choice: str = "groq_4_tier" 
    is_point_wise: bool = False 

# ==========================================
# 🏭 4. MAIN API ENDPOINT (The Custom Pipeline)
# ==========================================
@app.post("/ask")
def ask_ai(request: UserRequest):
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    today_date = datetime.now().strftime("%Y-%m-%d")
    user_cmd = request.question.strip().lower()
    
    # --- 🛡️ ADMIN INTERCEPTOR ---
    if user_cmd == "#total_tokens":
        stat = token_stats_col.find_one({"date_str": today_date}) if MONGO_URL else None
        if stat:
            msg = (f"📊 **SYSTEM ADMIN REPORT (V17.1 Custom)** 📊\n\n📅 **Date:** {today_date}\n"
                   f"🔄 **Total Tokens Today:** {stat.get('total_tokens', 0)}\n"
                   f"📞 **Total API Calls:** {stat.get('api_calls', 0)}\n"
                   f"🧠 Worker (70B): {stat.get('worker_tokens', 0)} tokens\n"
                   f"🕵️‍♂️ Critic/Lib (8B): {stat.get('critic_tokens', 0)} tokens")
        else: msg = "Aaj abhi tak koi token use nahi hua hai."
        return {"answer": f"{msg}\n\n[Engine: Admin Interceptor 🛡️]"}
        
    elif user_cmd == "#system_status":
        return {"answer": f"🟢 **SYSTEM STATUS: ONLINE (V17.1 Fixed Critic Pipeline)** 🟢\n\n🚀 Direct Groq Engine: Active (CrewAI Removed)\n🧠 Memory: Pinecone Mega-Vault\n💾 DB: MongoDB\n[Engine: Admin 🛡️]"}
        
    elif user_cmd == "#flush_memory":
        if MONGO_URL: messages_col.delete_many({"session_id": request.session_id})
        try:
            if index: index.delete(delete_all=True, namespace=request.session_id)
        except Exception: pass
        return {"answer": f"🧹 **MEMORY FLUSHED** 🧹\n\n{request.user_name} bhai, saari yaadein delete ho gayi hain!"}

    # --- 🔍 CONTEXT GATHERING ---
    vector_context = "No past facts."
    try:
        if index:
            results = index.query(vector=get_embedding(request.question), top_k=2, include_metadata=True, namespace=request.session_id)
            if results and results.get('matches'): vector_context = "\n".join([m['metadata']['text'] for m in results['matches']])
    except Exception: pass

    history = ""
    if MONGO_URL:
        past = list(messages_col.find({"session_id": request.session_id}).sort("_id", -1).limit(3))
        history = "\n".join([f"U: {m['user_query']}\nA: {re.sub(r'\[Engine:.*?\]', '', m['ai_response']).strip()}" for m in reversed(past)])

    dynamic_example = get_dynamic_example(request.question, request.user_name)
    point_rule = "Use clean bullet points." if request.is_point_wise else "Use concise paragraphs."

    # ==========================================
    # 🏭 THE CUSTOM FAST PIPELINE (2-STEP)
    # ==========================================
    lib_keys, wrk_keys, crt_keys = get_groq_keys("librarian"), get_groq_keys("worker"), get_groq_keys("critic")
    total_tokens, w_tok, c_tok = 0, 0, 0
    
    # STEP 1: LIBRARIAN (Do we need internet?)
    lib_prompt = f"Question: {request.question}\nDoes this require current internet search? Reply only YES or NO."
    need_search, l_tok = direct_groq_call(lib_prompt, "librarian", lib_keys)
    c_tok += l_tok
    
    web_data = ""
    if "YES" in str(need_search).upper():
        web_data = f"Web Search Info:\n{search_web(request.question)}"

        # STEP 2: WORKER (Generate Raw Answer - 70B)
    wrk_prompt = (f"Facts from Database: {vector_context}\nChat History: {history}\n{web_data}\n\n"
                  f"User's Question: {request.question}\n"
                  f"Task: You are an expert AI. Answer the user's question factually and accurately.\n"
                  f"- If 'Facts from Database' or 'Web Search Info' have the answer, use them.\n"
                  f"- If they are empty or don't have the answer, USE YOUR OWN INTERNAL KNOWLEDGE to answer fully.\n"
                  f"- Do not add greetings or styling, just give the raw factual answer.")

    # 🚀 STEP 3: CRITIC (FIXED PROMPT - Strict separation of Style and Facts + Emojis)
    crt_prompt = (f"Task: Rewrite the RAW_ANSWER to match the conversational STYLE of the EXAMPLE_TONE.\n\n"
                  f"RAW_ANSWER (Keep these facts 100% intact, do not alter the truth): \n'{raw_answer}'\n\n"
                  f"EXAMPLE_TONE (Use this personality and formatting, but DO NOT copy its text/facts): \n{dynamic_example}\n\n"
                  f"CRITICAL RULES:\n"
                  f"1. DO NOT change the facts from the RAW_ANSWER. If the raw answer states a specific detail (like 'Lenovo'), you MUST include that detail accurately.\n"
                  f"2. Add a greeting similar to the EXAMPLE_TONE.\n"
                  f"3. Generously sprinkle contextually relevant EMOJIS throughout your response to make it engaging.\n"
                  f"4. {point_rule}\n"
                  f"Output ONLY the final translated conversational message.")
    final_answer, c_tok_step = direct_groq_call(crt_prompt, "critic", crt_keys)
    c_tok += c_tok_step
    
    total_tokens = w_tok + c_tok
    clean_answer = re.sub(r'(?i)(Word Count|Note:|Validation|Task:).*', '', str(final_answer), flags=re.DOTALL).strip()

    # --- 💾 UPDATE DB & PINECONE ---
    if MONGO_URL and total_tokens > 0:
        token_stats_col.update_one({"date_str": today_date}, {"$inc": {"total_tokens": total_tokens, "api_calls": 1, "worker_tokens": w_tok, "critic_tokens": c_tok}}, upsert=True)
        messages_col.insert_one({"session_id": request.session_id, "user_query": request.question, "ai_response": f"{clean_answer}\n\n[Engine: V17.1 Direct API ⚡]", "timestamp": current_time})

    if index and clean_answer and "Error" not in clean_answer:
        try: index.upsert(vectors=[{"id": str(uuid.uuid4()), "values": get_embedding(f"Q: {request.question} A: {clean_answer}"), "metadata": {"text": f"User: {request.question}\nAI: {clean_answer}"}}], namespace=request.session_id)
        except Exception: pass

    final_db_answer = f"{clean_answer}\n\n[Engine: V17.1 Direct API ⚡ | Tokens: {total_tokens}]"
    return {"answer": final_db_answer}

# ==========================================
# 🚀 5. KEEP-ALIVE SYSTEM (Anti-Sleep)
# ==========================================
@app.api_route("/", methods=["GET", "HEAD"])
def home(): return {"status": "V17.1 Custom Pipeline Active"}

@app.get("/ping")
def ping(): return {"status": "Main jag raha hoon bhai!"}

async def keep_alive_loop():
    while True:
        await asyncio.sleep(14 * 60) 
        try:
            async with httpx.AsyncClient() as client:
                await client.get("https://ai-agent-backend-bek6.onrender.com/ping")
                logger.info("Ping Sent!")
        except Exception as e: logger.error(f"Ping failed: {e}")

@app.on_event("startup")
async def startup_event(): asyncio.create_task(keep_alive_loop())
