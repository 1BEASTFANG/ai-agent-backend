import os
import re
import json
import traceback
# 🚀 FIXED 1: Naya Google GenAI import (Deprecation warning hatane ke liye)
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

# --- ENGINE 2: NATIVE GEMINI SETUP (UPDATED) ---
gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
gemini_client = None
if gemini_api_key:
    # 🚀 FIXED 2: Naye SDK ke hisab se Client initialization
    gemini_client = genai.Client(api_key=gemini_api_key)

# --- ENGINE 1: GROQ KEY FACTORY ---
def get_groq_llm(role="worker", index=0):
    if role == "librarian":
        start, end = 1, 6
        model = "groq/llama-3.1-8b-instant"
    elif role in ["manager", "critic"]:
        start, end = 6, 11
        model = "groq/llama-3.1-8b-instant"
    else: 
        start, end = 11, 51
        # 🚀 FIXED 3: Heavy model (70b) hata kar fast model lagaya taaki Rate Limit hit na ho
        model = "groq/llama-3.1-8b-instant"

    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(start, end)]
    valid = [k for k in keys if k]
    if not valid: raise ValueError(f"Pool {role} has no keys!")
    return LLM(model=model, api_key=valid[index % len(valid)], temperature=0.2)

# --- REQUEST MODEL ---
class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str
    engine_choice: str = "gemini_native"
    is_point_wise: bool = False 

# ==========================================
# 🚀 MAIN API ENDPOINT
# ==========================================
@app.post("/ask")
def ask_ai(request: UserRequest, db: Session = Depends(get_db)):
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    
    # 1. History Context
    past = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(6).all()
    history = "\n".join([f"User: {m.user_query}\nAI: {re.sub(r'\[Engine:.*?\]', '', m.ai_response).strip()}" for m in reversed(past)])

    # 🚀 Point-wise Instruction
    point_style = ""
    if request.is_point_wise:
        point_style = "STRICT RULE: Provide the entire response ONLY in bullet points. Do not use long paragraphs."

    # ------------------------------------------
    # ⚡ ENGINE 2: PURE NATIVE GEMINI (UPDATED)
    # ------------------------------------------
    if request.engine_choice == "gemini_native":
        prompt = f"""
        {point_style}
        Time: {current_time}
        Context: {history}
        User {request.user_name}: {request.question}
        Answer in natural Hinglish directly.
        """
        try:
            if not gemini_client:
                raise ValueError("Gemini API Key missing in environment!")
            
            # 🚀 FIXED 4: Naye SDK ka Text Generation syntax
            response = gemini_client.models.generate_content(
                model='gemini-1.5-pro',
                contents=prompt
            )
            answer = f"{response.text.strip()}\n\n[Engine: Native Gemini ⚡]"
        except Exception as e:
            answer = f"Gemini Engine Error: {str(e)}"

    # ------------------------------------------
    # 🤖 ENGINE 1: GROQ 4-TIER AGENTS
    # ------------------------------------------
    else:
        try:
            lib_agent = Agent(role='Librarian', goal='Find context.', backstory='Expert researcher.', llm=get_groq_llm("librarian"))
            mgr_agent = Agent(role='Manager', goal='Plan answer.', backstory=f'Senior Strategist. {point_style}', llm=get_groq_llm("manager"))
            wrk_agent = Agent(role='Worker', goal='Write answer.', backstory='CS Expert.', llm=get_groq_llm("worker"), tools=[SerperDevTool()])
            crt_agent = Agent(role='Critic', goal='Fix errors.', backstory=f'Perfectionist Editor. {point_style}', llm=get_groq_llm("critic"))

            t1 = Task(description=f"Filter history: {history}", agent=lib_agent, expected_output="Key context.")
            t2 = Task(description=f"Plan Hinglish response for: {request.question}", agent=mgr_agent, expected_output="Strategy.")
            t3 = Task(description=f"Execute search and write draft.", agent=wrk_agent, expected_output="Draft answer.")
            
            final_task_desc = f"Review and polish. {point_style} Remove narration."
            t4 = Task(description=final_task_desc, agent=crt_agent, expected_output="Final clean text.")

            crew = Crew(agents=[lib_agent, mgr_agent, wrk_agent, crt_agent], tasks=[t1, t2, t3, t4], verbose=False)
            answer = f"{str(crew.kickoff()).strip()}\n\n[Engine: Groq 4-Tier 🤖]"
            
        except Exception as e:
            answer = f"Agent Error: {str(e)}"

    # 3. Save & Return
    db.add(ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=answer))
    db.commit()
    return {"answer": answer}
