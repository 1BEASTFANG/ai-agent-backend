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

# --- ENGINE 2: GEMINI (FIXED TO FLASH) ---
gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
gemini_client = None
if gemini_api_key:
    gemini_client = genai.Client(api_key=gemini_api_key)

# ==========================================
# 🚀 Nikhil's Exact Key Distribution Logic
# ==========================================
def get_groq_keys(role):
    # Librarian: 1 to 5
    if role == "librarian": start, end = 1, 6
    # Manager & Critic: 6 to 10 (Critic using same as manager)
    elif role in ["manager", "critic"]: start, end = 6, 11
    # Worker: 11 to 51
    else: start, end = 11, 51
    
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(start, end)]
    return [k for k in keys if k]

def create_llm(api_key):
    return LLM(model="groq/llama-3.1-8b-instant", api_key=api_key, temperature=0.1)

class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str
    engine_choice: str = "groq_4_tier" 
    is_point_wise: bool = False 

@app.post("/ask")
def ask_ai(request: UserRequest, db: Session = Depends(get_db)):
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    
    # History limit strictly 2 to save tokens
    past = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(2).all()
    history = "\n".join([f"U: {m.user_query}\nA: {re.sub(r'\[Engine:.*?\]', '', m.ai_response).strip()}" for m in reversed(past)])

    point_rule = "Provide answer in bullet points." if request.is_point_wise else "Provide natural paragraph."

    # ------------------------------------------
    # ⚡ ENGINE 2: GEMINI (Using FLASH strictly)
    # ------------------------------------------
    if request.engine_choice == "gemini_native":
        try:
            # Strictly using flash to avoid 404
            response = gemini_client.models.generate_content(
                model='gemini-1.5-flash', 
                contents=f"History: {history}\nUser: {request.question}\nRule: {point_rule}\nAnswer in natural Hinglish."
            )
            answer = f"{response.text.strip()}\n\n[Engine: Native Gemini ⚡]"
        except Exception as e:
            answer = f"Gemini Error: {str(e)} (Agar 404 aaye toh Google ne free tier block kiya hai, Groq use karein)"

    # ------------------------------------------
    # 🤖 ENGINE 1: GROQ 4-TIER (Nikhil's Architecture)
    # ------------------------------------------
    else:
        lib_keys = get_groq_keys("librarian")
        mgr_keys = get_groq_keys("manager") # Manager and Critic share this
        wrk_keys = get_groq_keys("worker")
        
        success = False
        # 🚀 Worker ki keys par loop chalega. Agar limit hit hui toh agli try karega.
        for i in range(len(wrk_keys)):
            try:
                l_key = lib_keys[i % len(lib_keys)]
                m_key = mgr_keys[i % len(mgr_keys)]
                w_key = wrk_keys[i]
                c_key = mgr_keys[(i + 1) % len(mgr_keys)] # Critic uses a different key from the manager pool

                # 1. Agents Defined
                lib_agent = Agent(role='Librarian', goal='Summarize prompt and history.', backstory='Context analyzer.', llm=create_llm(l_key))
                mgr_agent = Agent(role='Manager', goal='Define rules and persona.', backstory='Prompt engineer.', llm=create_llm(m_key))
                wrk_agent = Agent(role='Worker', goal='Execute task.', backstory='Executor.', llm=create_llm(w_key), tools=[SerperDevTool()])
                crt_agent = Agent(role='Critic', goal='Polish and fix errors.', backstory='Strict Editor.', llm=create_llm(c_key))

                # 2. Strict Tasks with Context passing (TOKEN SAVER)
                t1 = Task(
                    description=f"History: {history}\nUser prompt: {request.question}\nSummarize the user's intent.",
                    agent=lib_agent,
                    expected_output="Short summary of what user wants."
                )
                
                t2 = Task(
                    description=f"Create strict rules and a persona for the worker based on the summary.",
                    agent=mgr_agent,
                    context=[t1], # 🚀 Sirf t1 ka output lega, history nahi
                    expected_output="Persona and Rules for Worker."
                )
                
                t3 = Task(
                    description=f"Execute the task using the persona and rules provided. Search web if needed.",
                    agent=wrk_agent,
                    context=[t2], # 🚀 Sirf Manager ki rules lega
                    expected_output="Raw drafted answer."
                )
                
                t4 = Task(
                    description=f"Polish the drafted answer. Fix mistakes. {point_rule} Answer in natural Hinglish.",
                    agent=crt_agent,
                    context=[t3], # 🚀 Sirf Worker ka draft lega
                    expected_output="Final polished Hinglish answer."
                )

                crew = Crew(agents=[lib_agent, mgr_agent, wrk_agent, crt_agent], tasks=[t1, t2, t3, t4], verbose=False)
                answer = f"{str(crew.kickoff()).strip()}\n\n[Engine: Groq 4-Tier 🤖 (Nikhil's Arc)]"
                success = True
                break 
                
            except Exception as e:
                print(f"Key Attempt {i+1} failed: {str(e)}")
                if i == len(wrk_keys) - 1:
                    answer = "Bhai, Groq ki saari keys ki limit exhaust ho gayi hai. Thodi der baad try karein."
                continue

    db.add(ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=answer))
    db.commit()
    return {"answer": answer}
