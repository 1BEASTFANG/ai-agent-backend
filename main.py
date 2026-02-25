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

MONGO_URL = os.getenv("MONGO_URL")
try:
    if MONGO_URL:
        mongo_client = MongoClient(MONGO_URL, tls=True, tlsCAFile=certifi.where(), tlsAllowInvalidCertificates=True)
        db_mongo = mongo_client["ai_assistant_db"]
        messages_col = db_mongo["messages"]
        token_stats_col = db_mongo["token_stats_v2"]
        logger.info("Successfully connected to MongoDB Atlas! 🎉")
except Exception as e: logger.error(f"MongoDB Connection Failed: {e}")

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
# ⚡ 2. CORE TOOLS
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

def search_web(query):
    serper_key = os.getenv("SERPER_API_KEY", "").strip()
    if not serper_key: return "No internet access."
    try:
        res = httpx.post("https://google.serper.dev/search", headers={"X-API-KEY": serper_key}, json={"q": query}, timeout=10.0)
        if res.status_code == 200:
            results = res.json().get("organic", [])
            return "\n".join([f"- {r.get('title')}: {r.get('snippet')}" for r in results[:3]])
    except Exception: return "Search failed."
    return "No recent data found."

def get_dynamic_examples(question: str, user_name: str) -> str:
    try:
        if index:
            res = index.query(vector=get_embedding(question), top_k=2, include_metadata=True, namespace="few-shot-examples")
            if res and res.get('matches'):
                examples = []
                for i, match in enumerate(res['matches']):
                    ex_text = match['metadata']['template'].format(q=question, user_name=user_name)
                    examples.append(f"[STYLE EXAMPLE {i+1}]\n{ex_text}")
                return "\n\n".join(examples)
    except Exception: pass
    return f"Output: 'Ji {user_name} bhai! Iska jawab yeh raha...'"

# ==========================================
# 🧠 3. DIRECT GROQ API ENGINE
# ==========================================
def get_groq_keys(role):
    if role == "librarian": start, end = 1, 5
    else: start, end = 5, 51
    return [k for k in [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(start, end)] if k]

def direct_groq_call(prompt, role, keys):
    if role == "fast_core": model_name = "llama-3.1-8b-instant" 
    elif role == "worker": model_name = "llama-3.3-70b-versatile" 
    else: model_name = "llama-3.1-8b-instant" 
    
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
# 🏭 4. MAIN API ENDPOINT (V19.4 Fix Parroting)
# ==========================================
@app.post("/ask")
def ask_ai(request: UserRequest):
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    today_date = datetime.now().strftime("%Y-%m-%d")
    user_cmd = request.question.strip().lower()
    
    if user_cmd == "#total_tokens":
        stat = token_stats_col.find_one({"date_str": today_date}) if MONGO_URL else None
        if stat:
            msg = (f"📊 **SYSTEM ADMIN REPORT (V19.4)** 📊\n\n📅 **Date:** {today_date}\n"
                   f"🔄 **Total Tokens Today:** {stat.get('total_tokens', 0)}\n"
                   f"📞 **Total API Calls:** {stat.get('api_calls', 0)}\n"
                   f"🧠 **Deep Core:** {stat.get('deep_core_tokens', 0)} tokens\n"
                   f"⚡ **Fast Core:** {stat.get('fast_core_tokens', 0)} tokens")
        else: msg = "Aaj abhi tak koi token use nahi hua hai."
        return {"answer": f"{msg}\n\n[Engine: Admin 🛡️]"}
        
    elif user_cmd == "#system_status":
        return {"answer": f"🟢 **SYSTEM STATUS: ONLINE (V19.4 Strict Anti-Parrot)** 🟢\n\n🚀 Engines: Fast Mode (Casual) + Deep Core (Structured)\n🧠 Memory: Pinecone Vault\n[Engine: Admin 🛡️]"}
        
    elif user_cmd == "#flush_memory":
        if MONGO_URL: messages_col.delete_many({"session_id": request.session_id})
        try:
            if index: index.delete(delete_all=True, namespace=request.session_id)
        except Exception: pass
        return {"answer": f"🧹 **MEMORY FLUSHED** 🧹\n\n{request.user_name} bhai, saari yaadein delete ho gayi hain!"}

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

    total_tokens, w_tok, c_tok = 0, 0, 0
    engine_used = ""
    footer_msg = ""

    # ==========================================
    # ⚡ CORE 1: FAST MODE (Strictly Natural Chat)
    # ==========================================
    if request.engine_choice == "gemini_native":
        fast_keys = get_groq_keys("fast_core")
        
        # 🚀 FIX: Removed examples entirely. Forced natural chat.
        fast_prompt = (
            f"System: You are {request.user_name}'s fast, casual, and friendly AI assistant.\n\n"
            f"=== MEMORY ===\n{vector_context}\n\n"
            f"=== CONVERSATION ===\n"
            f"History: {history}\n"
            f"User: {request.question}\n\n"
            f"CRITICAL RULES:\n"
            f"1. Reply naturally like a friend in Hinglish.\n"
            f"2. Keep it SHORT (1-3 sentences max).\n"
            f"3. STRICTLY NO bullet points, NO headings, and NO lists.\n"
            f"4. Just answer the user's current message directly.\n"
            f"AI Response:"
        )
        
        raw_answer, c_tok = direct_groq_call(fast_prompt, "fast_core", fast_keys)
        total_tokens = c_tok
        clean_answer = re.sub(r'(?i)(Word Count|Note:|Validation|Task:|AI Response:).*', '', str(raw_answer), flags=re.DOTALL).strip()
        
        footer_msg = f"[V19.4 Fast Core ⚡ | Tokens: {total_tokens} | Switch to 'Groq 4-Tier' for Deep Info]"

    # ==========================================
    # 🧠 CORE 2: DEEP RESEARCH MODE (Smart Formatting)
    # ==========================================
    else:
        lib_keys, wrk_keys = get_groq_keys("librarian"), get_groq_keys("worker")
        
        lib_prompt = f"Question: {request.question}\nDoes this require current internet search? Reply only YES or NO."
        need_search, l_tok = direct_groq_call(lib_prompt, "librarian", lib_keys)
        c_tok += l_tok
        
        web_data = ""
        if "YES" in str(need_search).upper():
            web_data = f"Web Search Info:\n{search_web(request.question)}"

        dynamic_examples = get_dynamic_examples(request.question, request.user_name)

        # 🚀 FIX: Extreme warning added to NOT copy the examples' content.
        master_prompt = (
            f"System: You are {request.user_name}'s highly intelligent AI assistant.\n\n"
            f"=== REAL DATA (USE THIS TO ANSWER) ===\nMemory: {vector_context}\nWeb Search: {web_data}\n\n"
            f"=== STYLE TEMPLATES (FORMAT ONLY) ===\n{dynamic_examples}\n"
            f"⚠️ WARNING: DO NOT talk about the topics in the STYLE TEMPLATES (like Chai, Gaming, or Cricket) unless the user explicitly asks about them!\n\n"
            f"=== CONVERSATION ===\n"
            f"History: {history}\n"
            f"User: {request.question}\n\n"
            f"RULES:\n"
            f"1. Answer the user's specific question using REAL DATA.\n"
            f"2. Write in normal paragraphs. ONLY use bullet points if you are listing 3 or more distinct items/steps.\n"
            f"3. Be friendly in Hinglish with emojis.\n"
            f"AI Response:"
        )
                         
        raw_answer, w_tok = direct_groq_call(master_prompt, "worker", wrk_keys)
        total_tokens = w_tok + c_tok
        
        clean_answer = re.sub(r'(?i)(Word Count|Note:|Validation|Task:|AI Response:|Here is the response).*', '', str(raw_answer), flags=re.DOTALL).strip()
        if clean_answer.startswith("'") and clean_answer.endswith("'"): clean_answer = clean_answer[1:-1].strip()
        if clean_answer.startswith('"') and clean_answer.endswith('"'): clean_answer = clean_answer[1:-1].strip()
        
        footer_msg = f"[V19.4 Deep Core 🧠 | Tokens: {total_tokens}]"

    # --- 💾 DB UPDATE ---
    if MONGO_URL and total_tokens > 0:
        db_updates = {"total_tokens": total_tokens, "api_calls": 1}
        if request.engine_choice == "gemini_native":
            db_updates["fast_core_tokens"] = total_tokens
        else:
            db_updates["deep_core_tokens"] = total_tokens
            
        token_stats_col.update_one({"date_str": today_date}, {"$inc": db_updates}, upsert=True)
        messages_col.insert_one({"session_id": request.session_id, "user_query": request.question, "ai_response": f"{clean_answer}\n\n{footer_msg}", "timestamp": current_time})

    if index and clean_answer and "Error" not in clean_answer and "Output:" not in clean_answer:
        try: index.upsert(vectors=[{"id": str(uuid.uuid4()), "values": get_embedding(f"Q: {request.question} A: {clean_answer}"), "metadata": {"text": f"User: {request.question}\nAI: {clean_answer}"}}], namespace=request.session_id)
        except Exception: pass

    return {"answer": f"{clean_answer}\n\n{footer_msg}"}

# ==========================================
# 🚀 5. KEEP-ALIVE
# ==========================================
@app.api_route("/", methods=["GET", "HEAD"])
def home(): return {"status": "V19.4 Strict Anti-Parrot Pipeline Active"}

@app.get("/ping")
def ping(): return {"status": "Main jag raha hoon bhai!"}

async def keep_alive_loop():
    while True:
        await asyncio.sleep(14 * 60) 
        try:
            async with httpx.AsyncClient() as client:
                await client.get("https://ai-agent-backend-bek6.onrender.com/ping")
        except Exception: pass

@app.on_event("startup")
async def startup_event(): asyncio.create_task(keep_alive_loop())
