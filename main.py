import os
import re
import uuid
import traceback
import logging
import chromadb # 🚀 NEW: Vector Database
from google import genai 
from datetime import datetime
from fastapi import FastAPI, Depends
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
# 🧠 2. VECTOR DATABASE SETUP (Long-Term Memory)
# ==========================================
# ChromaDB folder banayega server par jahan vector data save hoga
CHROMA_PATH = "./chroma_memory_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
# Collection (Table) for storing AI interactions
memory_collection = chroma_client.get_or_create_collection(name="ai_long_term_memory")

# ==========================================
# ⚡ 3. ENGINES & KEYS
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
# 🚀 4. CORE API ENDPOINT (Vector + Multi-Agent)
# ==========================================
@app.post("/ask")
def ask_ai(request: UserRequest, db: Session = Depends(get_db)):
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    final_db_answer = f"{request.user_name} bhai, server mein kuch technical locha hai."
    
    # ------------------------------------------
    # 🔍 RAG: RETRIEVE FROM VECTOR DATABASE
    # ------------------------------------------
    vector_context = "No relevant past memory found."
    try:
        # User ke current sawal se milti-julti purani baatein dhoondo
        results = memory_collection.query(
            query_texts=[request.question],
            n_results=2, # Sirf top 2 most relevant memories lao
            where={"session_id": request.session_id} # Sirf is user ki memory
        )
        if results and results['documents'] and results['documents'][0]:
            vector_context = "\n---\n".join(results['documents'][0])
            logger.info("Vector DB Successfully Retrieved Context!")
    except Exception as e:
        logger.error(f"Vector DB Retrieve Error: {str(e)}")

    # Short-term SQL History (just for immediate continuity)
    past = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(1).all()
    history = "\n".join([f"U: {m.user_query}\nA: {re.sub(r'\[Engine:.*?\]', '', m.ai_response).strip()}" for m in reversed(past)])

    point_rule = "Format strictly in clean bullet points." if request.is_point_wise else "Use well-structured paragraphs. Use points only if necessary."

    # 🌟 FEW-SHOT EXAMPLES 🌟
    few_shot_examples = f"""
    EXAMPLE 1:
    User: "hi"
    Output: "{request.user_name} bhai, namaste! 🌟 Kahiye, aaj main aapki kya madad kar sakta hoon?"

    EXAMPLE 2:
    User: "Delhi kahan hai?"
    Output: "{request.user_name} ji, Delhi India ke north mein sthit hai. Yeh Yamuna nadi ke kinare basi hui desh ki rajdhani hai. 📍"
    """

    if request.engine_choice == "gemini_native":
        try:
            prompt = (
                f"### DEEP MEMORY ###\n{vector_context}\n\n"
                f"### RECENT HISTORY ###\n{history}\n\n"
                f"### USER QUESTION ###\n{request.question}\n\n"
                f"### RULES ###\n{point_rule}\nAnswer in friendly Hinglish. Address user as {request.user_name}."
            )
            response = gemini_client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
            clean_answer = response.text.strip()
            final_db_answer = f"{clean_answer}\n\n[Engine: Native Gemini ⚡ | Vector DB 🧠]"
        except Exception as e:
            clean_answer = "Error occurred."
            final_db_answer = f"Gemini Error: {str(e)}"

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

                lib_agent = Agent(role='Data Librarian', goal='Combine deep memory and recent history to classify query.', backstory='Analytical AI.', llm=create_llm("groq/llama-3.1-8b-instant", l_key), allow_delegation=False)
                mgr_agent = Agent(role='Operations Manager', goal='Provide strictly formatted action plans.', backstory='Strict Orchestrator.', llm=create_llm("groq/llama-3.1-8b-instant", m_key), allow_delegation=False)
                wrk_agent = Agent(role='Elite Worker', goal='Execute plan without hallucination.', backstory='Senior AI Researcher.', llm=create_llm("groq/llama-3.3-70b-versatile", w_key), tools=[SerperDevTool()], allow_delegation=False, max_iter=3)
                crt_agent = Agent(role='QA Critic', goal='Format final response matching examples exactly.', backstory='Strict formatting engine.', llm=create_llm("groq/llama-3.1-8b-instant", c_key), allow_delegation=False)

                # 🚀 VECTOR CONTEXT INJECTED HERE
                t1 = Task(
                    description=f"### DEEP MEMORY ###\n{vector_context}\n\n### RECENT HISTORY ###\n{history}\n\n### NEW QUESTION ###\n{request.question}\n\nAnalyze NEW QUESTION. Output exactly ONE word: 'GREETING', 'CONTINUATION', or 'NEW_TOPIC'.",
                    agent=lib_agent, expected_output="A single word summary."
                )
                
                t2 = Task(
                    description=f"### NEW QUESTION ###\n{request.question}\n\nIf Librarian summary is GREETING: Command = 'NO SEARCH. Friendly hello.' Otherwise: Command = 'Answer factually under 200 words using Deep Memory if relevant. Use search if needed.'",
                    agent=mgr_agent, context=[t1], expected_output="1-line command."
                )
                
                t3 = Task(
                    description=f"### NEW QUESTION ###\n{request.question}\n\nExecute Manager's command. Draft raw facts. NO META TEXT.",
                    agent=wrk_agent, context=[t2], expected_output="Raw drafted text."
                )
                
                t4 = Task(
                    description=(
                        f"### NEW QUESTION ###\n{request.question}\n\n"
                        f"CRITICAL INSTRUCTION: Format Worker's draft EXACTLY in the style of these examples. DO NOT output internal thoughts like 'Word Count'.\n\n"
                        f"{few_shot_examples}\n\n"
                        f"Now write the final response. {point_rule}"
                    ),
                    agent=crt_agent, context=[t3], expected_output="Final, clean Hinglish message. NO internal logs."
                )

                crew = Crew(agents=[lib_agent, mgr_agent, wrk_agent, crt_agent], tasks=[t1, t2, t3, t4], verbose=False)
                result = crew.kickoff()
                
                clean_answer = str(result).strip()
                
                token_usage = "N/A"
                try:
                    if hasattr(crew, 'usage_metrics') and crew.usage_metrics:
                        token_usage = crew.usage_metrics.total_tokens
                except Exception: pass

                final_db_answer = f"{clean_answer}\n\n[Engine: Enterprise Groq 🤖 | Total Tokens: {token_usage} | Keys: {key_tracker} | Vector DB 🧠]"
                success = True
                break 
                
            except Exception as e:
                logger.error(f"Groq Loop Failed on attempt {i+1}: {str(e)}")
                if i == len(wrk_keys) - 1:
                    final_db_answer = f"{request.user_name} bhai, Groq ki saari keys ki limit exhaust ho gayi hai."
                continue

    # ------------------------------------------
    # 💾 SAVE TO SQL & VECTOR DATABASE
    # ------------------------------------------
    # 1. Save to SQL (Short term)
    db.add(ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=final_db_answer))
    db.commit()

    # 2. Save to Vector DB (Long term deep memory)
    # We only save the clean answer, not the "[Engine...]" tags, to keep DB pure!
    if clean_answer and "Error" not in clean_answer:
        try:
            doc_id = str(uuid.uuid4()) # Generate unique ID
            memory_collection.add(
                documents=[f"User asked: {request.question}\nAI answered: {clean_answer}"],
                metadatas=[{"session_id": request.session_id, "timestamp": current_time}],
                ids=[doc_id]
            )
            logger.info("Successfully Saved to Vector DB!")
        except Exception as e:
            logger.error(f"Vector DB Save Error: {str(e)}")

    return {"answer": final_db_answer}
