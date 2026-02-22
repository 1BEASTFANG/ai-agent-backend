import os
import re
import uuid
import shutil
import traceback
import logging
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

    # Short-term SQL History
    past = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(1).all()
    history = "\n".join([f"U: {m.user_query}\nA: {re.sub(r'\[Engine:.*?\]', '', m.ai_response).strip()}" for m in reversed(past)])

    point_rule = "Format response STRICTLY in clean bullet points." if request.is_point_wise else "Use well-structured concise paragraphs."

    # 🌟 EXPANDED FEW-SHOT EXAMPLES (The Ultimate AI Brainwash) 🌟
    few_shot_examples = f"""
    EXAMPLE 1 (Greeting):
    User: "hi" or "hello"
    Output: "{request.user_name} bhai, namaste! 🌟 Kahiye, main aapki kya madad kar sakta hoon?"

    EXAMPLE 2 (Storing a Fact):
    User: "Mera college ANDC hai"
    Output: "Done {request.user_name} bhai! 🏫 Maine yaad kar liya hai ki aap ANDC college mein padhte hain."

    EXAMPLE 3 (Recalling a Fact):
    User: "Mera college kaunsa hai?"
    Output: "{request.user_name} bhai, aap ANDC college mein padhte hain! 🎓"

    EXAMPLE 4 (Coding Question - STRICT MARKDOWN):
    User: "Python mein loop kaise likhe?"
    Output: "{request.user_name} bhai, yeh raha aapka code:
    ```python
    for i in range(5):
        print(i)
    ```
    Is code se aap 0 se 4 tak print kar sakte hain. 🚀"

    EXAMPLE 5 (General Fact):
    User: "Taj Mahal kahan hai?"
    Output: "{request.user_name} ji, Taj Mahal Agra, Uttar Pradesh mein sthit hai. 🕌"
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
                f"### RULES ###\n{point_rule}\nAnswer in friendly natural Hinglish concisely. DO NOT ask follow-up questions. Address user as {request.user_name}."
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
                    goal='Combine memory and recent history to classify query accurately.', 
                    backstory='Advanced Database Specialist.', 
                    llm=create_llm("groq/llama-3.1-8b-instant", l_key),
                    allow_delegation=False
                )
                
                mgr_agent = Agent(
                    role='Operations Manager', 
                    goal='Provide strictly formatted action plans without extra text.', 
                    backstory='Strict Orchestration Lead.', 
                    llm=create_llm("groq/llama-3.1-8b-instant", m_key),
                    allow_delegation=False
                )
                
                wrk_agent = Agent(
                    role='Elite Worker', 
                    goal='Execute the manager\'s plan factually.', 
                    backstory='Senior AI Researcher. You use tools only when necessary. You ALWAYS put code in ``` markdown blocks.', 
                    llm=create_llm("groq/llama-3.3-70b-versatile", w_key), 
                    tools=[SerperDevTool()],
                    allow_delegation=False,
                    max_iter=3 
                )
                
                crt_agent = Agent(
                    role='QA Critic', 
                    goal='Format beautifully matching examples, add empathy.', 
                    backstory='Friendly Editor. You NEVER print internal logs. You PRESERVE code blocks perfectly.', 
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
                        f"Analyze the NEW QUESTION. Output exactly one of these three summaries:\n"
                        f"1. 'GREETING' (if the user is just saying hi, hello, etc.)\n"
                        f"2. 'MEMORY' (if it asks about past facts or personal info)\n"
                        f"3. 'NEW_TOPIC' (if it is a new factual or general question)\n"
                        f"Do not write anything else."
                    ),
                    agent=lib_agent,
                    expected_output="A single word summary: GREETING, MEMORY, or NEW_TOPIC."
                )
                
                t2 = Task(
                    description=(
                        f"### NEW QUESTION ###\n{request.question}\n\n"
                        f"INSTRUCTIONS:\n"
                        f"Based on Librarian's summary, write the command for the Worker:\n"
                        f"- If GREETING: Command = 'DO NOT use Search. Say a friendly hello.'\n"
                        f"- If MEMORY: Command = 'Answer using PAST FACTS only. Keep it short. DO NOT ask follow-up questions.'\n"
                        f"- Otherwise: Command = 'Answer factually concisely. Use web search if necessary. Wrap code in ``` blocks.'\n"
                    ),
                    agent=mgr_agent,
                    context=[t1], 
                    expected_output="A strict 1-line command for the worker."
                )
                
                t3 = Task(
                    description=(
                        f"### USER'S PAST FACTS ###\n{vector_context}\n\n"
                        f"### NEW QUESTION ###\n{request.question}\n\n"
                        f"INSTRUCTIONS:\n"
                        f"Execute the Manager's command exactly. Draft the response. Do not output meta-text. ALWAYS put code snippets in ``` language ``` markdown blocks."
                    ),
                    agent=wrk_agent,
                    context=[t2], 
                    expected_output="The raw drafted text containing facts and markdown code blocks."
                )
                
                t4 = Task(
                    description=(
                        f"### NEW QUESTION ###\n{request.question}\n\n"
                        f"CRITICAL RULES FOR OUTPUT:\n"
                        f"1. NEVER output words like 'Word Count', 'Manager Rules Check', 'Revised Response', or 'Note:'.\n"
                        f"2. You must format the Worker's draft EXACTLY in the style of these examples:\n\n"
                        f"{few_shot_examples}\n\n"
                        f"3. {point_rule}\n"
                        f"4. PRESERVE all markdown code blocks (```) perfectly.\n"
                        f"5. DO NOT ask repetitive follow-up questions. Be concise.\n"
                        f"OUTPUT ONLY THE FINAL SPOKEN MESSAGE THAT THE USER WILL READ."
                    ),
                    agent=crt_agent,
                    context=[t3], 
                    expected_output="Only the final, polished Hinglish message meant for the user. No internal logs. Code must be in markdown."
                )

                crew = Crew(agents=[lib_agent, mgr_agent, wrk_agent, crt_agent], tasks=[t1, t2, t3, t4], verbose=False)
                result = crew.kickoff()
                
                clean_answer = str(result).strip()

                # 🚀 THE SAFETY NET: Strip out any leaked meta-text generated by the Critic
                leak_pattern = r'(?i)(Word Count|Manager\'s Rules Check|Revised Response|Note:|Validation).*'
                clean_answer = re.sub(leak_pattern, '', clean_answer, flags=re.DOTALL).strip()
                
                # Safely extract token usage
                token_usage = "N/A"
                try:
                    if hasattr(crew, 'usage_metrics') and crew.usage_metrics:
                        token_usage = crew.usage_metrics.total_tokens
                except Exception as e:
                    logger.warning(f"Could not parse tokens: {str(e)}")

                final_db_answer = f"{clean_answer}\n\n[Engine: Enterprise Groq 🤖 | Total Tokens: {token_usage} | Keys: {key_tracker} | Vector DB 🧠]"
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
    # 1. Save to SQL (Short term)
    db.add(ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=final_db_answer))
    db.commit()

    # 2. Save to Vector DB (Long term deep memory)
    if clean_answer and "Error" not in clean_answer:
        try:
            doc_id = str(uuid.uuid4())
            memory_collection.add(
                # Storing it cleanly so AI understands it's a past fact when retrieved
                documents=[f"User previously stated/asked: {request.question}\nAI answered: {clean_answer}"],
                metadatas=[{"session_id": request.session_id, "timestamp": current_time}],
                ids=[doc_id]
            )
            logger.info("Successfully Saved to Vector DB!")
        except Exception as e:
            logger.error(f"Vector DB Save Error: {str(e)}")

    return {"answer": final_db_answer}
