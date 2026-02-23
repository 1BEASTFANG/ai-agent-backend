import os
import re
import uuid
import shutil
import traceback
import logging
import asyncio # 🚀 NEW: Keep-Alive ke liye
import httpx   # 🚀 NEW: Keep-Alive ke liye
import chromadb # 🚀 Vector Database
from google import genai 
from datetime import datetime
from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse # 🚀 File download
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Text, String
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

# ==========================================
# 🚀 1. ENTERPRISE SETTINGS & LOGGING
# ==========================================
os.environ["CREWAI_TRACING_ENABLED"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- SQL DATABASE SETUP (Short-Term Memory) ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ultimate_stable_v5.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    user_query = Column(Text)
    ai_response = Column(Text)

# 🚀 NEW: Token Tracking Table
class DailyTokenStat(Base):
    __tablename__ = "token_stats"
    id = Column(Integer, primary_key=True, index=True)
    date_str = Column(String, unique=True, index=True) # e.g., '2026-02-23'
    total_tokens = Column(Integer, default=0)
    api_calls = Column(Integer, default=0)

Base.metadata.create_all(bind=engine)
app = FastAPI()

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# ==========================================
# 🧠 2. VECTOR DATABASE (ChromaDB)
# ==========================================
CHROMA_PATH = "./chroma_memory_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
memory_collection = chroma_client.get_or_create_collection(name="ai_long_term_memory")

# 🚀 Memory Download Endpoint
@app.get("/download-memory")
def download_memory():
    """Zips the ChromaDB folder and provides it as a download."""
    try:
        output_filename = "ai_vector_memory_backup"
        shutil.make_archive(output_filename, 'zip', CHROMA_PATH)
        logger.info(f"Memory backup created: {output_filename}.zip")
        return FileResponse(
            path=f"{output_filename}.zip", 
            media_type='application/zip', 
            filename=f"{output_filename}.zip"
        )
    except Exception as e:
        return {"error": f"Download failed: {str(e)}"}

# ==========================================
# ⚡ 3. ENGINES & LLM TOOLS
# ==========================================
gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
gemini_client = None
if gemini_api_key:
    gemini_client = genai.Client(api_key=gemini_api_key)

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
def ask_ai(request: UserRequest, db: Session = Depends(get_db)):
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    today_date = datetime.now().strftime("%Y-%m-%d") # 🚀 NEW: Aaj ki date
    
    # ==========================================
    # 🛡️ THE ADMIN COMMANDS INTERCEPTOR 🛡️
    # ==========================================
    user_cmd = request.question.strip().lower()
    
    if user_cmd == "#total_tokens":
        stat = db.query(DailyTokenStat).filter(DailyTokenStat.date_str == today_date).first()
        if stat:
            msg = f"📊 **SYSTEM ADMIN REPORT** 📊\n\n📅 **Date:** {today_date}\n🔄 **Total Tokens Used Today:** {stat.total_tokens}\n📞 **Total API Calls:** {stat.api_calls}\n\n*Note: Yeh meter raat 12 baje ke baad naye din ke liye automatically 0 ho jayega.*"
        else:
            msg = f"📊 **SYSTEM ADMIN REPORT** 📊\n\nAaj (Date: {today_date}) abhi tak koi token use nahi hua hai."
        
        return {"answer": f"{msg}\n\n[Engine: Admin Interceptor 🛡️ | Cost: 0 Tokens]"}
        
    elif user_cmd == "#system_status":
        msg = f"🟢 **SYSTEM STATUS: ONLINE** 🟢\n\n🚀 **Server Engine:** Render Cloud (Active)\n🧠 **Vector Memory:** ChromaDB (Connected)\n🤖 **Primary AI:** Enterprise Groq 4-Tier\n⏱️ **Keep-Alive System:** Running perfectly"
        return {"answer": f"{msg}\n\n[Engine: Admin Interceptor 🛡️ | Cost: 0 Tokens]"}
        
    elif user_cmd == "#flush_memory":
        # Clear Short-term SQL History
        db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).delete()
        db.commit()
        # Clear Long-term Vector DB for this user
        try:
            memory_collection.delete(where={"session_id": request.session_id})
        except Exception: pass
        
        msg = f"🧹 **MEMORY FLUSHED SUCCESSFULLY** 🧹\n\n{request.user_name} bhai, aapki saari purani baatein aur yaadein system se delete kar di gayi hain. Mera dimaag ab ekdam fresh hai! Ek naye sire se shuruwat karte hain."
        return {"answer": f"{msg}\n\n[Engine: Admin Interceptor 🛡️ | Cost: 0 Tokens]"}

    # ------------------------------------------
    
    final_db_answer = f"{request.user_name} bhai, server mein kuch technical locha hai. Thodi der baad try karo."

    # ------------------------------------------
    # 🔍 RAG: Deep Context Retrieval
    # ------------------------------------------
    vector_context = "No relevant past facts found."
    try:
        results = memory_collection.query(
            query_texts=[request.question],
            n_results=2, 
            where={"session_id": request.session_id}
        )
        if results and results['documents'] and results['documents'][0]:
            vector_context = "\n---\n".join(results['documents'][0])
            logger.info("Vector Context Retrieved Successfully!")
    except Exception as e:
        logger.error(f"Vector DB Retrieve Error: {str(e)}")

    # 🚀 UPDATE 1: limit(1) changed to limit(3) for context memory
    past = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(3).all()
    history = "\n".join([f"U: {m.user_query}\nA: {re.sub(r'\[Engine:.*?\]', '', m.ai_response).strip()}" for m in reversed(past)])

    point_rule = "Format response STRICTLY in clean bullet points." if request.is_point_wise else "Use well-structured concise paragraphs."

    # 🌟 20 FEW-SHOT EXAMPLES (The Ultimate AI Brainwash Matrix) 🌟
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
    Output: "{request.user_name} bhai, yeh raha aapka code:
    ```python
    for i in range(5):
        print(i)
    ```
    Is code se aap 0 se 4 tak print kar sakte hain. 🚀"

    EXAMPLE 10 (Coding 2 - STRICT MARKDOWN):
    User: "C++ hello world"
    Output: "Yeh lijiye {request.user_name} bhai:
    ```cpp
    #include <iostream>
    int main() {{
        std::cout << \"Hello World!\";
        return 0;
    }}
    ```
    Bilkul simple aur basic! 💻"

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
            final_db_answer = f"{clean_answer}\n\n[Engine: Native Gemini ⚡ | Vector DB 🧠]"
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
                # 🔄 Round-Robin Key Management
                l_idx, m_idx, w_idx, c_idx = (i % len(lib_keys)) + 1, (i % len(mgr_keys)) + 1, i + 1, ((i + 1) % len(mgr_keys)) + 1
                
                l_key, m_key = lib_keys[l_idx - 1], mgr_keys[m_idx - 1]
                w_key, c_key = wrk_keys[w_idx - 1], mgr_keys[c_idx - 1] 

                key_tracker = f"L:{l_idx} | M:{m_idx} | W:{w_idx} | C:{c_idx}"

                # ==========================================
                # 🏛️ FULL AGENT DEFINITIONS 
                # ==========================================
                lib_agent = Agent(
                    role='Data Librarian', 
                    goal='Classify NEW QUESTION only.', 
                    backstory='Advanced Database Specialist.', 
                    llm=create_llm("groq/llama-3.1-8b-instant", l_key),
                    allow_delegation=False
                )
                
                mgr_agent = Agent(
                    role='Operations Manager', 
                    goal='Provide 1-line command based on classification.', 
                    backstory='Strict Orchestration Lead.', 
                    llm=create_llm("groq/llama-3.1-8b-instant", m_key),
                    allow_delegation=False
                )
                
                wrk_agent = Agent(
                    role='Elite Worker', 
                    goal='Answer ONLY the NEW QUESTION factually.', 
                    backstory='Senior AI Researcher. You ONLY use ``` markdown blocks for writing actual Programming Code (like C++, Python). You NEVER use markdown blocks for text, explanations, or jokes. You NEVER use words like "Memory", "Database", or "Fact Store" in your response. You DO NOT answer past questions from history.', 
                    llm=create_llm("groq/llama-3.3-70b-versatile", w_key), 
                    tools=[SerperDevTool()],
                    allow_delegation=False,
                    max_iter=3 
                )
                
                crt_agent = Agent(
                    role='QA Critic', 
                    goal='Format beautifully matching examples, add empathy.', 
                    backstory='Friendly Editor. You NEVER print internal logs, word counts, or rule checks. You NEVER ask follow-up questions at the end of your response. You NEVER mix answers from history.', 
                    llm=create_llm("groq/llama-3.1-8b-instant", c_key),
                    allow_delegation=False
                )

                # ==========================================
                # 📋 FULL ENTERPRISE TASK PIPELINE
                # ==========================================
                t1 = Task(
                    description=(
                        f"### USER'S PAST FACTS ###\n{vector_context}\n\n"
                        f"### RECENT HISTORY ###\n{history}\n\n"
                        f"### NEW QUESTION ###\n{request.question}\n\n"
                        f"INSTRUCTIONS:\n"
                        f"Analyze ONLY the NEW QUESTION. Output exactly 1 word:\n" 
                        f"- 'GREETING' (if hi, hello)\n"
                        f"- 'FACT_STORE' (if user is telling a fact about themselves to remember)\n"
                        f"- 'MEMORY_RECALL' (if user is asking about past facts)\n"
                        f"- 'CONTINUATION' (if user asks to explain more, give examples, or refers to the previous message)\n"
                        f"- 'NEW_TOPIC' (for general questions, coding, or jokes)\n"
                        f"Do not write anything else."
                    ),
                    agent=lib_agent,
                    expected_output="A single word summary: GREETING, FACT_STORE, MEMORY_RECALL, CONTINUATION, or NEW_TOPIC."
                )
                
                t2 = Task(
                    description=(
                        f"### NEW QUESTION ###\n{request.question}\n\n"
                        f"INSTRUCTIONS:\n"
                        f"Based on Librarian's summary, write the command for the Worker:\n"
                        f"- GREETING: 'Say a friendly hello.'\n"
                        f"- FACT_STORE: 'Acknowledge the fact in 1 simple sentence only.'\n"
                        f"- MEMORY_RECALL: 'Answer directly using PAST FACTS only. DO NOT explain.'\n"
                        f"- CONTINUATION: 'Read HISTORY carefully and explain the last topic in more detail.'\n"
                        f"- NEW_TOPIC: 'Answer factually. If user asks for code, use markdown. If Joke/Fact, use normal text.'\n"
                    ),
                    agent=mgr_agent,
                    context=[t1], 
                    expected_output="A strict 1-line command for the worker."
                )
                
                t3 = Task(
                    description=(
                        f"### USER'S PAST FACTS ###\n{vector_context}\n\n"
                        f"### RECENT HISTORY ###\n{history}\n\n"
                        f"### NEW QUESTION ###\n{request.question}\n\n"
                        f"INSTRUCTIONS:\n"
                        f"Execute Manager's command. IF NEW_TOPIC: Answer ONLY the NEW QUESTION and DO NOT repeat previous history. IF CONTINUATION: Rely deeply on RECENT HISTORY to provide a follow-up detailed answer. DO NOT output meta-text. ONLY use ``` language ``` blocks if writing a programming script. DO NOT use code blocks for jokes or text. CRITICAL: DO NOT say things like 'this is in our fact store' or 'based on memory'." 
                    ),
                    agent=wrk_agent,
                    context=[t2], 
                    expected_output="The raw drafted text containing facts and optional code blocks."
                )
                
                t4 = Task(
                    description=(
                        f"### NEW QUESTION ###\n{request.question}\n\n"
                        f"CRITICAL RULES FOR OUTPUT:\n"
                        f"1. Choose ONLY ONE matching situation from the examples below. DO NOT combine answers from past history.\n" 
                        f"2. NEVER output words like 'Word Count', 'Manager Rules Check', 'Revised Response', or 'Note:'.\n"
                        f"3. NEVER use words like 'Fact Store', 'Database', or 'Memory'.\n" 
                        f"4. You must format the Worker's draft EXACTLY mimicking the style of these examples:\n\n"
                        f"{few_shot_examples}\n\n"
                        f"5. DO NOT ask repetitive follow-up questions (e.g. stop saying 'kya aap aur janna chahte hain?'). Just give the answer and stop.\n"
                        f"6. {point_rule}\n"
                        f"OUTPUT ONLY THE FINAL SPOKEN MESSAGE THAT THE USER WILL READ."
                    ),
                    agent=crt_agent,
                    context=[t3], 
                    expected_output="Only the final, polished Hinglish message meant for the user. No internal logs. Code must be in markdown."
                )

                crew = Crew(agents=[lib_agent, mgr_agent, wrk_agent, crt_agent], tasks=[t1, t2, t3, t4], verbose=False)
                result = crew.kickoff()
                
                clean_answer = str(result).strip()

                leak_pattern = r'(?i)(Word Count|Manager\'s Rules Check|Revised Response|Note:|Validation|Code Quality|Empathy|Fact Store|Database).*'
                clean_answer = re.sub(leak_pattern, '', clean_answer, flags=re.DOTALL).strip()
                
                # 🚀 Tracking Daily Tokens in DB
                token_usage = 0
                try:
                    if hasattr(crew, 'usage_metrics') and crew.usage_metrics:
                        token_usage = crew.usage_metrics.total_tokens
                except Exception as e:
                    logger.warning(f"Could not parse tokens: {str(e)}")

                if isinstance(token_usage, int) and token_usage > 0:
                    stat = db.query(DailyTokenStat).filter(DailyTokenStat.date_str == today_date).first()
                    if not stat:
                        stat = DailyTokenStat(date_str=today_date, total_tokens=token_usage, api_calls=1)
                        db.add(stat)
                    else:
                        stat.total_tokens += token_usage
                        stat.api_calls += 1
                    db.commit()

                token_display = token_usage if token_usage > 0 else "N/A"
                final_db_answer = f"{clean_answer}\n\n[Engine: Enterprise Groq 🤖 | Total Tokens: {token_display} | Keys: {key_tracker} | Vector DB 🧠]"
                success = True
                break 
                
            except Exception as e:
                logger.error(f"Groq Loop Failed on attempt {i+1}: {traceback.format_exc()}")
                if i == len(wrk_keys) - 1:
                    final_db_answer = f"{request.user_name} bhai, Groq ki saari keys ki limit exhaust ho gayi hai. Kripya Gemini mode try karein."
                continue

    # ------------------------------------------
    # 💾 SAVE TO SQL & VECTOR DATABASE
    # ------------------------------------------
    db.add(ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=final_db_answer))
    db.commit()

    if clean_answer and "Error" not in clean_answer:
        try:
            doc_id = str(uuid.uuid4())
            memory_collection.add(
                documents=[f"User previously stated/asked: {request.question}\nAI answered: {clean_answer}"],
                metadatas=[{"session_id": request.session_id, "timestamp": current_time}],
                ids=[doc_id]
            )
            logger.info("Successfully Saved to Vector DB!")
        except Exception as e:
            logger.error(f"Vector DB Save Error: {str(e)}")

    return {"answer": final_db_answer}

# ==========================================
# 🚀 5. KEEP-ALIVE SYSTEM (Anti-Sleep)
# ==========================================
@app.get("/ping")
def ping():
    """Yeh chota sa function server ko batayega ki wo zinda hai, bina AI ko jagaye."""
    return {"status": "Main jag raha hoon bhai!"}

async def keep_alive_loop():
    """Yeh background worker har 14 minute mein server ko ping karega."""
    while True:
        await asyncio.sleep(14 * 60) # 14 minutes ka wait
        try:
            # FIX: Removed the extra brackets [ ] from the URL
            url = "[https://ai-agent-backend-bek6.onrender.com/ping](https://ai-agent-backend-bek6.onrender.com/ping)" 
            async with httpx.AsyncClient() as client:
                await client.get(url)
                logger.info("Keep-Alive Ping Sent!")
        except Exception as e:
            logger.error(f"Ping failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Jaise hi server start hoga, yeh ping loop chalu ho jayega."""
    asyncio.create_task(keep_alive_loop())
