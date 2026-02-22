import os
import re
import traceback
import logging
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

# Setup Professional Server Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- DATABASE SETUP ---
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
# ⚡ 2. GEMINI ENGINE INITIALIZATION
# ==========================================
gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
gemini_client = None
if gemini_api_key:
    gemini_client = genai.Client(api_key=gemini_api_key)

# ==========================================
# 🔐 3. KEY DISTRIBUTION & LLM FACTORY
# ==========================================
def get_groq_keys(role):
    """Fetches keys based on the agent's role to manage rate limits strictly."""
    if role == "librarian": start, end = 1, 6
    elif role in ["manager", "critic"]: start, end = 6, 11
    else: start, end = 11, 51
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(start, end)]
    return [k for k in keys if k]

def create_llm(model_name, api_key):
    """Creates a deterministic LLM instance (low temperature)."""
    return LLM(model=model_name, api_key=api_key, temperature=0.1)

# --- API MODELS ---
class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str
    engine_choice: str = "groq_4_tier" 
    is_point_wise: bool = False 

# ==========================================
# 🧠 4. CORE API ENDPOINT (The Multi-Agent Pipeline)
# ==========================================
@app.post("/ask")
def ask_ai(request: UserRequest, db: Session = Depends(get_db)):
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    answer = f"{request.user_name} bhai, server mein kuch technical locha hai. Thodi der baad try karo."
    
    # Context Management: Keep strict limit to avoid TPM overload
    past = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(2).all()
    history = "\n".join([f"U: {m.user_query}\nA: {re.sub(r'\[Engine:.*?\]', '', m.ai_response).strip()}" for m in reversed(past)])

    point_rule = "Format response STRICTLY in clean bullet points." if request.is_point_wise else "Use well-structured paragraphs, utilize bullet points only if necessary for clarity."

    # ------------------------------------------
    # ⚡ FAST PATH: NATIVE GEMINI
    # ------------------------------------------
    if request.engine_choice == "gemini_native":
        try:
            logger.info(f"Routing request to Gemini for user: {request.user_name}")
            prompt = (
                f"### HISTORY ###\n{history}\n\n"
                f"### USER QUESTION ###\n{request.question}\n\n"
                f"### RULES ###\n{point_rule}\nAnswer in friendly natural Hinglish. Address user as {request.user_name}."
            )
            response = gemini_client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
            answer = f"{response.text.strip()}\n\n[Engine: Native Gemini ⚡]"
        except Exception as e:
            logger.error(f"Gemini Engine Failed: {str(e)}")
            answer = f"Gemini Error: {str(e)}"

    # ------------------------------------------
    # 🤖 DEEP RESEARCH PATH: ENTERPRISE GROQ (4-TIER)
    # ------------------------------------------
    else:
        logger.info(f"Initiating Enterprise Groq Pipeline for user: {request.user_name}")
        lib_keys, mgr_keys, wrk_keys = get_groq_keys("librarian"), get_groq_keys("manager"), get_groq_keys("worker")
        
        success = False
        for i in range(len(wrk_keys)):
            try:
                # 🔄 Round-Robin Key Management with Clean Number Tracking
                l_idx, m_idx, w_idx, c_idx = (i % len(lib_keys)) + 1, (i % len(mgr_keys)) + 1, i + 1, ((i + 1) % len(mgr_keys)) + 1
                
                l_key, m_key = lib_keys[l_idx - 1], mgr_keys[m_idx - 1]
                w_key, c_key = wrk_keys[w_idx - 1], mgr_keys[c_idx - 1] 

                key_tracker = f"L:{l_idx} | M:{m_idx} | W:{w_idx} | C:{c_idx}"
                logger.info(f"Attempt {i+1} using Keys: {key_tracker}")

                # ==========================================
                # 🏛️ AGENT DEFINITIONS (Strict Delegation = False)
                # ==========================================
                lib_agent = Agent(
                    role='Data Librarian', 
                    goal='Extract entities, handle context shifts.', 
                    backstory='Advanced Database Specialist. You evaluate if history is relevant to the new query.', 
                    llm=create_llm("groq/llama-3.1-8b-instant", l_key),
                    allow_delegation=False # 🛡️ GUARDRAIL: No side-talk
                )
                
                mgr_agent = Agent(
                    role='Operations Manager', 
                    goal='Security validation and strict task planning.', 
                    backstory='Strict Orchestration Lead. You build safe, clear instructions.', 
                    llm=create_llm("groq/llama-3.1-8b-instant", m_key),
                    allow_delegation=False # 🛡️ GUARDRAIL
                )
                
                wrk_agent = Agent(
                    role='Elite Worker', 
                    goal='Execute the plan factually and logically.', 
                    backstory='Senior AI Researcher. You use tools only when necessary and think step-by-step.', 
                    llm=create_llm("groq/llama-3.3-70b-versatile", w_key), 
                    tools=[SerperDevTool()],
                    allow_delegation=False, # 🛡️ GUARDRAIL
                    max_iter=3 # 🛡️ GUARDRAIL: Prevents infinite search loops
                )
                
                crt_agent = Agent(
                    role='QA Critic', 
                    goal='Enforce constraints, format beautifully, add empathy.', 
                    backstory='Friendly but strict Quality Assurance Validator.', 
                    llm=create_llm("groq/llama-3.1-8b-instant", c_key),
                    allow_delegation=False # 🛡️ GUARDRAIL
                )

                # ==========================================
                # 📋 ENTERPRISE TASK PIPELINE (With Delimiters)
                # ==========================================
                t1 = Task(
                    description=(
                        f"### HISTORY ###\n{history}\n\n"
                        f"### NEW QUESTION ###\n{request.question}\n\n"
                        f"INSTRUCTIONS:\n"
                        f"1) If NEW QUESTION is completely unrelated to HISTORY, command a 'Memory Flush'.\n"
                        f"2) Extract primary entities (names, concepts, coding languages) as 'Search Anchors'.\n"
                        f"3) Output a 2-line summary of what the user wants to know right now."
                    ),
                    agent=lib_agent,
                    expected_output="Action (Keep/Flush Memory) | Search Anchors | Short Summary"
                )
                
                t2 = Task(
                    description=(
                        f"### NEW QUESTION ###\n{request.question}\n\n"
                        f"INSTRUCTIONS:\n"
                        f"1) SECURITY: Verify the question is safe. If not, output abort rule.\n"
                        f"2) CONSTRAINTS: If it's a simple greeting (hi/hello), rule is 'NO WEB SEARCH'. Rule: Limit to 200 words unless coding/detailed explanation is explicitly asked.\n"
                        f"3) PLAN: Based on the Librarian's summary, write a 3-step action plan for the Worker."
                    ),
                    agent=mgr_agent,
                    context=[t1], 
                    expected_output="Security Status | Strict Constraints | 3-Step Execution Plan"
                )
                
                t3 = Task(
                    description=(
                        f"### NEW QUESTION ###\n{request.question}\n\n"
                        f"INSTRUCTIONS:\n"
                        f"Execute the Manager's plan. Use Search Anchors if web search is permitted and necessary. Provide a highly accurate, raw, and factual answer directly addressing the question."
                    ),
                    agent=wrk_agent,
                    context=[t2], 
                    expected_output="Detailed, factual raw answer without formatting."
                )
                
                t4 = Task(
                    description=(
                        f"### NEW QUESTION ###\n{request.question}\n\n"
                        f"INSTRUCTIONS:\n"
                        f"1) Check word count. Summarize if it violates Manager's rules.\n"
                        f"2) EMPATHY INJECTION: Start exactly with '{request.user_name} bhai,' or '{request.user_name} ji,' and a friendly introductory sentence.\n"
                        f"3) Translate/Polish into natural Hinglish. Add relevant emojis 🌟.\n"
                        f"4) {point_rule}"
                    ),
                    agent=crt_agent,
                    context=[t3], 
                    expected_output="Final, polished, empathetic Hinglish output."
                )

                # 🚀 KICKOFF THE ASSEMBLY LINE
                crew = Crew(agents=[lib_agent, mgr_agent, wrk_agent, crt_agent], tasks=[t1, t2, t3, t4], verbose=False)
                result = crew.kickoff()
                
                # Safely extract token usage
                token_usage = "N/A"
                try:
                    if hasattr(crew, 'usage_metrics') and crew.usage_metrics:
                        token_usage = crew.usage_metrics.total_tokens
                except Exception as e:
                    logger.warning(f"Could not parse tokens: {str(e)}")

                answer = f"{str(result).strip()}\n\n[Engine: Enterprise Groq 🤖 | Tokens: {token_usage} | Keys: {key_tracker}]"
                success = True
                break # Exit loop if successful
                
            except Exception as e:
                logger.error(f"Groq Loop Failed on attempt {i+1}: {traceback.format_exc()}")
                if i == len(wrk_keys) - 1:
                    answer = f"{request.user_name} bhai, Groq ki saari keys ki limit exhaust ho gayi hai. Kripya Gemini mode try karein."
                continue

    # 💾 SAVE STATE & RETURN
    db.add(ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=answer))
    db.commit()
    return {"answer": answer}
