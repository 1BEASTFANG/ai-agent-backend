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
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse # 🚀 NAYA IMPORT STREAMING KE LIYE
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

# ==========================================
# 🌐 ADVANCED WEB SEARCH ENGINE
# ==========================================
def get_serper_keys():
    keys = [
        os.getenv("SERPER_API_KEY_1", "").strip(),
        os.getenv("SERPER_API_KEY_2", "").strip(),
        os.getenv("SERPER_API_KEY_3", "").strip() 
    ]
    return [k for k in keys if k]

def search_web(query):
    keys = get_serper_keys()
    if not keys: return "No internet access (API keys missing)."
    
    for i, key in enumerate(keys):
        try:
            res = httpx.post(
                "https://google.serper.dev/search", 
                headers={"X-API-KEY": key}, 
                json={"q": query}, 
                timeout=10.0
            )
            if res.status_code == 200:
                results = res.json().get("organic", [])
                return "\n".join([f"- {r.get('title')}: {r.get('snippet')}" for r in results[:3]])
            else:
                logger.warning(f"Serper Key {i+1} failed with status {res.status_code}. Trying next...")
                continue 
        except Exception as e:
            logger.warning(f"Serper Key {i+1} threw an error: {e}. Trying next...")
            continue 
            
    return "Internet search is currently unavailable due to heavy traffic."

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
# 🧠 3. DIRECT GROQ API ENGINES (Normal & Stream)
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

# 🚀 NAYA: STREAMING GENERATOR FOR CHATGPT STYLE TYPING
async def async_stream_groq(prompt, role, keys):
    model_name = "llama-3.1-8b-instant" if role == "fast_core" else "llama-3.3-70b-versatile"
    for key in keys:
        try:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "stream": True}
            
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", "https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20.0) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str == "[DONE]": return
                                try:
                                    chunk = json.loads(data_str)
                                    content = chunk["choices"][0].get("delta", {}).get("content", "")
                                    if content: yield content # 🚀 Yahan ek-ek word nikalega!
                                except Exception: pass
                        return
        except Exception: continue
    yield "Error: Server overloaded."

class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str
    engine_choice: str = "groq_4_tier" 
    is_point_wise: bool = False 

# ==========================================
# ⚡ BACKGROUND TASK (Saves Time)
# ==========================================
def save_memory_background(session_id, question, clean_answer, total_tokens, engine_choice, today_date, current_time):
    # 1. MongoDB Update
    if MONGO_URL and total_tokens > 0:
        try:
            db_updates = {"total_tokens": total_tokens, "api_calls": 1}
            if engine_choice == "gemini_native": db_updates["fast_core_tokens"] = total_tokens
            else: db_updates["deep_core_tokens"] = total_tokens
                
            token_stats_col.update_one({"date_str": today_date}, {"$inc": db_updates}, upsert=True)
            messages_col.insert_one({"session_id": session_id, "user_query": question, "ai_response": clean_answer, "timestamp": current_time})
        except Exception as e: logger.error(f"Mongo Background Error: {e}")

    # 2. Pinecone Vector Save
    if index and clean_answer and "Error" not in clean_answer and "Output:" not in clean_answer:
        try: 
            index.upsert(vectors=[{"id": str(uuid.uuid4()), "values": get_embedding(f"Q: {question} A: {clean_answer}"), "metadata": {"text": f"User: {question}\nAI: {clean_answer}"}}], namespace=session_id)
        except Exception as e: logger.error(f"Pinecone Background Error: {e}")


# ==========================================
# 🏭 4A. MAIN NORMAL API ENDPOINT (Aapki existing app ke liye)
# ==========================================
@app.post("/ask")
def ask_ai(request: UserRequest, bg_tasks: BackgroundTasks):
    start_time = time.time() # 🚀 RESPONSE TIMER START
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    today_date = datetime.now().strftime("%Y-%m-%d")
    user_cmd = request.question.strip().lower()
    
    if user_cmd == "#total_tokens":
        stat = token_stats_col.find_one({"date_str": today_date}) if MONGO_URL else None
        if stat:
            msg = (f"📊 **SYSTEM ADMIN REPORT (V19.5)** 📊\n\n📅 **Date:** {today_date}\n"
                   f"🔄 **Total Tokens Today:** {stat.get('total_tokens', 0)}\n"
                   f"📞 **Total API Calls:** {stat.get('api_calls', 0)}\n"
                   f"🧠 **Deep Core:** {stat.get('deep_core_tokens', 0)} tokens\n"
                   f"⚡ **Fast Core:** {stat.get('fast_core_tokens', 0)} tokens")
        else: msg = "Aaj abhi tak koi token use nahi hua hai."
        return {"answer": f"{msg}\n\n[Engine: Admin 🛡️]"}
        
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
    clean_answer = ""
    footer_msg = ""

    if request.engine_choice == "gemini_native":
        fast_keys = get_groq_keys("fast_core")
        fast_prompt = f"System: You are {request.user_name}'s fast AI assistant.\n\n=== MEMORY ===\n{vector_context}\n\n=== CONVERSATION ===\nHistory: {history}\nUser: {request.question}\n\nCRITICAL RULES:\n1. Reply naturally like a friend in Hinglish.\n2. Keep it SHORT.\nAI Response:"
        
        raw_answer, c_tok = direct_groq_call(fast_prompt, "fast_core", fast_keys)
        total_tokens = c_tok
        clean_answer = re.sub(r'(?i)(Word Count|Note:|Validation|Task:|AI Response:).*', '', str(raw_answer), flags=re.DOTALL).strip()
        elapsed_time = round(time.time() - start_time, 2)
        footer_msg = f"[V19.5 Fast Core ⚡ | ⏱️ {elapsed_time}s | ⚙️ {total_tokens} ]"

    else:
        lib_keys, wrk_keys = get_groq_keys("librarian"), get_groq_keys("worker")
        lib_prompt = f"Analyze this question carefully: '{request.question}'\nDoes this require checking REAL-TIME internet data? Reply YES or NO."
        need_search, l_tok = direct_groq_call(lib_prompt, "librarian", lib_keys)
        c_tok += l_tok
        
        web_data = ""
        if "YES" in str(need_search).upper():
            web_data = f"Web Search Info:\n{search_web(request.question)}"

        dynamic_examples = get_dynamic_examples(request.question, request.user_name)

        master_prompt = (
            f"System: You are {request.user_name}'s highly intelligent AI.\n\n"
            f"=== REAL DATA ===\nMemory: {vector_context}\nWeb Search: {web_data}\n\n"
            f"=== CONVERSATION ===\nHistory: {history}\nUser: {request.question}\n\n"
            f"RULES:\n1. Use real data.\n2. Be friendly in Hinglish with emojis.\nAI Response:"
        )
                          
        raw_answer, w_tok = direct_groq_call(master_prompt, "worker", wrk_keys)
        total_tokens = w_tok + c_tok
        clean_answer = re.sub(r'(?i)(Word Count|Note:|Validation|Task:|AI Response:|Here is the response).*', '', str(raw_answer), flags=re.DOTALL).strip()
        elapsed_time = round(time.time() - start_time, 2)
        footer_msg = f"[V19.5 Deep Core 🧠 | ⏱️ {elapsed_time}s | ⚙️ {total_tokens} ]"

    bg_tasks.add_task(save_memory_background, request.session_id, request.question, f"{clean_answer}\n\n{footer_msg}", total_tokens, request.engine_choice, today_date, current_time)
    return {"answer": f"{clean_answer}\n\n{footer_msg}"}

# ==========================================
# 🏭 4B. NAYA STREAMING ENDPOINT (Flutter App ke naye feature ke liye)
# ==========================================
@app.post("/ask_stream")
async def ask_ai_stream(request: UserRequest):
    start_time = time.time()
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    today_date = datetime.now().strftime("%Y-%m-%d")

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

    # Generator jo word by word answer dega
    async def response_generator():
        full_answer = ""
        footer_msg = ""
        estimated_tokens = 0
        
        if request.engine_choice == "gemini_native":
            fast_keys = get_groq_keys("fast_core")
            fast_prompt = f"System: You are {request.user_name}'s fast AI assistant.\n\n=== MEMORY ===\n{vector_context}\n\n=== CONVERSATION ===\nHistory: {history}\nUser: {request.question}\n\nRULES:\n1. Reply naturally in Hinglish.\n2. Keep it SHORT.\nAI Response:"
            
            async for chunk in async_stream_groq(fast_prompt, "fast_core", fast_keys):
                full_answer += chunk
                yield chunk
            
            estimated_tokens = len(full_answer) // 4
            elapsed_time = round(time.time() - start_time, 2)
            footer_msg = f"\n\n[V20 Stream ⚡ | ⏱️ {elapsed_time}s | ⚙️ ~{estimated_tokens}]"
            yield footer_msg
            
        else:
            lib_keys, wrk_keys = get_groq_keys("librarian"), get_groq_keys("worker")
            lib_prompt = f"Analyze: '{request.question}'. Requires checking internet? YES or NO."
            need_search, _ = direct_groq_call(lib_prompt, "librarian", lib_keys)
            
            web_data = ""
            if "YES" in str(need_search).upper():
                web_data = f"Web Search Info:\n{search_web(request.question)}"

            master_prompt = (
                f"System: You are {request.user_name}'s highly intelligent AI.\n\n"
                f"=== DATA ===\nMemory: {vector_context}\nWeb Search: {web_data}\n\n"
                f"=== CONVERSATION ===\nHistory: {history}\nUser: {request.question}\n\n"
                f"RULES:\n1. Use real data.\n2. Be friendly in Hinglish with emojis.\nAI Response:"
            )
            
            async for chunk in async_stream_groq(master_prompt, "worker", wrk_keys):
                full_answer += chunk
                yield chunk
                
            estimated_tokens = len(full_answer) // 4
            elapsed_time = round(time.time() - start_time, 2)
            footer_msg = f"\n\n[V20 Stream 🧠 | ⏱️ {elapsed_time}s | ⚙️ ~{estimated_tokens}]"
            yield footer_msg

        # Stream khatam hone ke baad DB me save karega (Parde ke peeche)
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, save_memory_background, request.session_id, request.question, full_answer + footer_msg, estimated_tokens, request.engine_choice, today_date, current_time)

    # Returning the stream
    return StreamingResponse(response_generator(), media_type="text/plain")

# ==========================================
# 🚀 5. KEEP-ALIVE
# ==========================================
@app.api_route("/", methods=["GET", "HEAD"])
def home(): return {"status": "V20.0 Live & Streaming Engine Active"}

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
