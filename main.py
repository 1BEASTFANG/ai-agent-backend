import os
import re
import json
import traceback
from google import genai 
from datetime import datetime
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Text, String
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

# CrewAI Imports
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

# 🚀 Settings
os.environ["CREWAI_TRACING_ENABLED"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"

# --- DATABASE ---
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

# --- ENGINE 2: NATIVE GEMINI SETUP ---
gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
gemini_client = None
if gemini_api_key:
    gemini_client = genai.Client(api_key=gemini_api_key)

# 🚀 Keys Loading Logic
def get_groq_keys(role="worker"):
    if role == "librarian": start, end = 1, 6
    elif role in ["manager", "critic"]: start, end = 6, 11
    else: start, end = 11, 51
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(start, end)]
    return [k for k in keys if k]

def create_llm(api_key):
    # Tokens per minute (TPM) safe rakhne ke liye 8b-instant model
    return LLM(model="groq/llama-3.1-8b-instant", api_key=api_key, temperature=0.2)

# --- REQUEST MODEL ---
class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str
    engine_choice: str = "groq_4_tier" 
    is_point_wise: bool = False 

# ==========================================
# 🚀 MAIN API ENDPOINT
# ==========================================
@app.post("/ask")
def ask_ai(request: UserRequest, db: Session = Depends(get_db)):
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    
    # 🚀 History limit 2 (TPM Balance ke liye)
    past = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(2).all()
    history = "\n".join([f"U: {m.user_query}\nA: {re.sub(r'\[Engine:.*?\]', '', m.ai_response).strip()}" for m in reversed(past)])

    point_style = ""
    if request.is_point_wise:
        point_style = "STRICT RULE: Provide the response ONLY in bullet points. No paragraphs."

    # ⚡ ENGINE 2: GEMINI FLASH
    if request.engine_choice == "gemini_native":
        try:
            response = gemini_client.models.generate_content(
                model='gemini-1.5-flash', 
                contents=f"{point_style}\nContext: {history}\nUser: {request.question}\nAnswer in natural Hinglish."
            )
            answer = f"{response.text.strip()}\n\n[Engine: Native Gemini ⚡]"
        except Exception as e:
            answer = f"Gemini Error: {str(e)}"

    # 🤖 ENGINE 1: GROQ 4-TIER (REACTIVE LOOP)
    else:
        lib_keys = get_groq_keys("librarian")
        mgr_keys = get_groq_keys("manager")
        wrk_keys = get_groq_keys("worker")
        crt_keys = get_groq_keys("critic")

        success = False
        # 🚀 Sabhi keys par loop chalega agar error aaye
        for i in range(len(wrk_keys)):
            try:
                # Modulo (%) se baki roles ki keys repeat hongi agar worker keys zyada hain
                l_key = lib_keys[i % len(lib_keys)]
                m_key = mgr_keys[i % len(mgr_keys)]
                w_key = wrk_keys[i]
                c_key = crt_keys[i % len(crt_keys)]

                lib_agent = Agent(role='Librarian', goal='Analyze history.', backstory='Expert researcher.', llm=create_llm(l_key))
                mgr_agent = Agent(role='Manager', goal='Plan response.', backstory='Senior Planner.', llm=create_llm(m_key))
                wrk_agent = Agent(role='Worker', goal='Research & Write.', backstory='CS Expert.', llm=create_llm(w_key), tools=[SerperDevTool()])
                # 🚀 CRITIC: Ye worker ke response ko polish karega
                crt_agent = Agent(role='Critic', goal='Polish & Review.', backstory='Professional Editor.', llm=create_llm(c_key))

                t1 = Task(description=f"Context from history: {history}", agent=lib_agent, expected_output="Key context.")
                t2 = Task(description=f"Answer strategy for: {request.question}", agent=mgr_agent, expected_output="Strategy.")
                t3 = Task(description=f"Draft Hinglish response.", agent=wrk_agent, expected_output="Raw draft.")
                # 🚀 CRITIC TASK: Polishing Step
                t4 = Task(description=f"Polish draft for natural Hinglish. {point_style}", agent=crt_agent, expected_output="Final clean text.")

                crew = Crew(agents=[lib_agent, mgr_agent, wrk_agent, crt_agent], tasks=[t1, t2, t3, t4], verbose=False)
                answer = f"{str(crew.kickoff()).strip()}\n\n[Engine: Groq 4-Tier 🤖]"
                success = True
                break 
            except Exception as e:
                print(f"Key {i+1} failed: {str(e)}")
                if i == len(wrk_keys) - 1:
                    answer = "Agent Error: Sabhi keys exhaust ho gayi hain. Gemini try karein."
                continue

    db.add(ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=answer))
    db.commit()
    return {"answer": answer}
