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
# 🚀 Key Distribution Logic
# ==========================================
def get_groq_keys(role):
    if role == "librarian": start, end = 1, 6
    elif role in ["manager", "critic"]: start, end = 6, 11
    else: start, end = 11, 51
    
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(start, end)]
    return [k for k in keys if k]

# 🚀 NAYA: Ab har agent ka model alag ho sakta hai
def create_llm(model_name, api_key):
    return LLM(model=model_name, api_key=api_key, temperature=0.2)

class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str
    engine_choice: str = "groq_4_tier" 
    is_point_wise: bool = False 

@app.post("/ask")
def ask_ai(request: UserRequest, db: Session = Depends(get_db)):
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    
    # 🚀 SAFETY FIX: Agar koi fatal error aaye toh default answer taiyar rahe
    answer = f"{request.user_name} bhai, server mein kuch technical locha hai. Thodi der baad try karo."
    
    # History limit strictly 2 
    past = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(2).all()
    history = "\n".join([f"U: {m.user_query}\nA: {re.sub(r'\[Engine:.*?\]', '', m.ai_response).strip()}" for m in reversed(past)])

    point_rule = "Use bullet points strictly." if request.is_point_wise else "Use well-structured points naturally."

    # ------------------------------------------
    # ⚡ ENGINE 2: GEMINI
    # ------------------------------------------
    if request.engine_choice == "gemini_native":
        try:
            response = gemini_client.models.generate_content(
                model='gemini-1.5-flash', 
                contents=f"History: {history}\nUser: {request.question}\nRule: {point_rule}\nAnswer in friendly natural Hinglish. Address the user as {request.user_name}."
            )
            answer = f"{response.text.strip()}\n\n[Engine: Native Gemini ⚡]"
        except Exception as e:
            answer = f"Gemini Error: {str(e)}"

    # ------------------------------------------
    # 🤖 ENGINE 1: GROQ 4-TIER (The Empathic Architecture)
    # ------------------------------------------
    else:
        lib_keys = get_groq_keys("librarian")
        mgr_keys = get_groq_keys("manager") 
        wrk_keys = get_groq_keys("worker")
        
        success = False
        for i in range(len(wrk_keys)):
            try:
                l_key = lib_keys[i % len(lib_keys)]
                m_key = mgr_keys[i % len(mgr_keys)]
                w_key = wrk_keys[i]
                c_key = mgr_keys[(i + 1) % len(mgr_keys)] 

                # 1. Agents Defined (Worker gets the 70B Brain)
                lib_agent = Agent(role='Librarian', goal='Analyze memory and topic shift.', backstory='Smart context manager.', llm=create_llm("groq/llama-3.1-8b-instant", l_key))
                mgr_agent = Agent(role='Manager', goal='Define strict rules.', backstory='Strict Prompt Engineer.', llm=create_llm("groq/llama-3.1-8b-instant", m_key))
                # 🚀 WORKER is now 70B for high-quality logic
                wrk_agent = Agent(role='Worker', goal='Execute task flawlessly.', backstory='Senior CS Expert.', llm=create_llm("groq/llama-3.3-70b-versatile", w_key), tools=[SerperDevTool()])
                crt_agent = Agent(role='Critic', goal='Polish, format, and add empathy.', backstory='Friendly, empathetic editor.', llm=create_llm("groq/llama-3.1-8b-instant", c_key))

                # 2. Tasks with NEW Instructions
                
                # 🚀 LIBRARIAN: Memory Flush Logic
                t1 = Task(
                    description=f"User prompt: '{request.question}'. History: '{history}'. Compare the new prompt with history. If the new prompt is a completely different topic, IGNORE the history entirely (flush memory). If related, keep the context. Output ONLY a short summary of what the user wants right now.",
                    agent=lib_agent,
                    expected_output="Short summary of current user intent."
                )
                
                # 🚀 MANAGER: Word Limit Logic
                t2 = Task(
                    description=f"Create rules for the worker based on the summary. RULE 1: Keep the answer under 200 words UNLESS the user explicitly asks for a detailed explanation or code. RULE 2: Set a professional persona.",
                    agent=mgr_agent,
                    context=[t1], 
                    expected_output="Rules for Worker."
                )
                
                t3 = Task(
                    description=f"Execute the task using the rules provided. Provide accurate information.",
                    agent=wrk_agent,
                    context=[t2], 
                    expected_output="Raw detailed answer."
                )
                
                # 🚀 CRITIC: Empathy, Formatting, and Emojis
                t4 = Task(
                    description=f"Review the worker's draft. 1) Polish it into natural Hinglish. 2) Structure it in clean bullet points. 3) Add relevant emojis 🌟. 4) VERY IMPORTANT: Start the response by addressing the user '{request.user_name}' empathetically. Example: '{request.user_name} bhai, ye bahut badhiya sawal hai!' or if correcting, '{request.user_name}, mujhe lagta hai yahan thodi galti hai, aaiye theek karte hain'. 5) Keep it concise as per rules. {point_rule}",
                    agent=crt_agent,
                    context=[t3], 
                    expected_output="Final empathetic and formatted Hinglish answer."
                )

                

                crew = Crew(agents=[lib_agent, mgr_agent, wrk_agent, crt_agent], tasks=[t1, t2, t3, t4], verbose=False)
                answer = f"{str(crew.kickoff()).strip()}\n\n[Engine: Groq Pro 4-Tier 🤖]"
                success = True
                break 
                
            except Exception as e:
                print(f"Key Attempt {i+1} failed: {str(e)}")
                if i == len(wrk_keys) - 1:
                    answer = f"{request.user_name} bhai, Groq ki saari keys ki limit exhaust ho gayi hai. Thodi der baad try karein ya Gemini par switch karein."
                continue

    db.add(ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=answer))
    db.commit()
    return {"answer": answer}
