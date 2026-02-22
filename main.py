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
    
    # 🚀 TPM FIX: History limit sirf 1 rakhi hai taaki 6000 limit cross na ho
    past = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(1).all()
    history = "\n".join([f"U: {m.user_query}\nA: {re.sub(r'\[Engine:.*?\]', '', m.ai_response).strip()}" for m in reversed(past)])

    point_style = "Answer in points." if request.is_point_wise else "Answer naturally."

    # ⚡ ENGINE 2: GEMINI FLASH (FIXED 404)
    if request.engine_choice == "gemini_native":
        try:
            # 🚀 FIXED: gemini-1.5-flash use kiya 404 error hatane ke liye
            response = gemini_client.models.generate_content(
                model='gemini-1.5-flash', 
                contents=f"Context: {history}\nUser: {request.question}\n{point_style}"
            )
            answer = f"{response.text.strip()}\n\n[Engine: Native Gemini ⚡]"
        except Exception as e:
            answer = f"Gemini Error: {str(e)}"

    # 🤖 ENGINE 1: GROQ (REACTIVE KEY LOOP)
    else:
        wrk_keys = get_groq_keys("worker")
        lib_keys = get_groq_keys("librarian")
        mgr_keys = get_groq_keys("manager")
        crt_keys = get_groq_keys("critic")
        
        success = False
        for i in range(len(wrk_keys)):
            try:
                # 🚀 Loop with multiple keys
                l_key, m_key, w_key, c_key = lib_keys[i%len(lib_keys)], mgr_keys[i%len(mgr_keys)], wrk_keys[i], crt_keys[i%len(crt_keys)]
                
                l_llm = LLM(model="groq/llama-3.1-8b-instant", api_key=l_key)
                m_llm = LLM(model="groq/llama-3.1-8b-instant", api_key=m_key)
                w_llm = LLM(model="groq/llama-3.1-8b-instant", api_key=w_key)
                c_llm = LLM(model="groq/llama-3.1-8b-instant", api_key=c_key)

                lib = Agent(role='Librarian', goal='Analyze context.', backstory='Context Expert.', llm=l_llm)
                mgr = Agent(role='Manager', goal='Plan answer.', backstory='Strategy expert.', llm=m_llm)
                wrk = Agent(role='Worker', goal='Research.', backstory='CS Expert.', llm=w_llm, tools=[SerperDevTool()])
                crt = Agent(role='Critic', goal='Polish.', backstory='Editor.', llm=c_llm)

                t1 = Task(description=f"History: {history}", agent=lib, expected_output="Context.")
                t2 = Task(description=f"Plan for: {request.question}", agent=mgr, expected_output="Plan.")
                t3 = Task(description=f"Draft answer.", agent=wrk, expected_output="Draft.")
                t4 = Task(description=f"Final Review. {point_style}", agent=crt, expected_output="Final Text.")

                crew = Crew(agents=[lib, mgr, wrk, crt], tasks=[t1, t2, t3, t4], verbose=False)
                answer = f"{str(crew.kickoff()).strip()}\n\n[Engine: Groq 4-Tier 🤖]"
                success = True
                break
            except Exception as e:
                print(f"Key {i} failed: {str(e)}")
                if i == len(wrk_keys)-1: answer = "All keys failed."
                continue

    db.add(ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=answer))
    db.commit()
    return {"answer": answer}
