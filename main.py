import os
import re
import json
import traceback
import google.generativeai as genai
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Text, String
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

# CrewAI Imports (Sirf Groq ke liye use honge)
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool

# 🚀 Settings for Stability
os.environ["CREWAI_TRACING_ENABLED"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ultimate_dual_engine.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

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

# --- NATIVE GEMINI SETUP (No CrewAI here) ---
gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    # Using the robust pro model
    native_gemini_model = genai.GenerativeModel('gemini-1.5-pro')
else:
    print("WARNING: GEMINI_API_KEY is missing!")

# --- GROQ CREWAI SETUP (For Engine 2) ---
search_tool = SerperDevTool()

def get_groq_llm(model_type="worker", index=0):
    # Librarian uses keys 1-5, Worker uses keys 6-50
    start, end = (1, 6) if model_type == "librarian" else (6, 51)
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(start, end)]
    valid_keys = [k for k in keys if k]
    
    if not valid_keys:
        raise ValueError(f"No valid Groq keys found for {model_type}!")
        
    actual_key = valid_keys[index % len(valid_keys)]
    model_name = "groq/llama-3.1-8b-instant" if model_type == "librarian" else "groq/llama-3.3-70b-versatile"
    
    return LLM(model=model_name, api_key=actual_key, temperature=0.3)

# --- REQUEST MODEL ---
class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str
    engine_choice: str = "gemini_native"  # Default is Gemini (Fast & Stable)

# ==========================================
# 🚀 MAIN API ENDPOINT
# ==========================================
@app.post("/ask")
def ask_ai(request: UserRequest, db: Session = Depends(get_db)):
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    
    # 1. Fetch Chat History
    past_msgs = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(6).all()
    history_str = ""
    if past_msgs:
        for m in reversed(past_msgs):
            # Clean HTML or backend tags from history
            clean_ai = re.sub(r'\[Engine:.*?\]', '', m.ai_response).strip()
            history_str += f"User: {m.user_query}\nAI: {clean_ai}\n\n"

    answer = ""

    # ==========================================
    # ⚡ ENGINE 1: NATIVE GEMINI (100% Stable)
    # ==========================================
    if request.engine_choice == "gemini_native":
        if not gemini_api_key:
            return {"answer": "Bhai, Gemini API Key missing hai server par!"}
            
        # Create a direct, powerful prompt
        prompt = f"""
        You are an expert AI assistant helping '{request.user_name}'.
        Current Date and Time: {current_time}.
        
        CRITICAL RULES:
        1. Answer in natural Hinglish (mix of Hindi and English).
        2. Be direct. Do not say "I am thinking" or "Here is the answer".
        3. Use the context below if it relates to the user's question.
        
        --- Past Conversation Context ---
        {history_str if history_str else "No past context. This is a new conversation."}
        ---------------------------------
        
        User's New Question: {request.question}
        """
        
        try:
            response = native_gemini_model.generate_content(prompt)
            answer = f"{response.text.strip()}\n\n[Engine: Native Gemini ⚡]"
        except Exception as e:
            print(f"Gemini Native Error: {e}")
            answer = f"Bhai, Gemini server par kuch issue aaya hai: {str(e)}"

    # ==========================================
    # 🤖 ENGINE 2: GROQ + CREWAI (Agentic Mode)
    # ==========================================
    else:
        try:
            wrk_llm = get_groq_llm("worker", 0)
            
            backstory = (
                f"You are helping {request.user_name}. Today is {current_time}. "
                "CRITICAL: Never narrate actions. Give direct answers in Hinglish only. "
                "Use the search tool ONLY if you need real-time facts, otherwise answer from memory."
            )
            
            worker = Agent(
                role='Expert Python & CS Assistant',
                goal='Provide accurate answers without narration.',
                backstory=backstory,
                llm=wrk_llm,
                tools=[search_tool],
                max_iter=3,
                verbose=False
            )
            
            task_desc = f"Context:\n{history_str}\n\nQuestion: {request.question}\nAnswer directly."
            task = Task(description=task_desc, expected_output="Clean Hinglish response", agent=worker)
            
            raw_res = str(Crew(agents=[worker], tasks=[task]).kickoff())
            
            # Clean up JSON leaks and tags
            clean_res = re.sub(r'```json.*?```', '', raw_res, flags=re.DOTALL)
            clean_res = re.sub(r'\{.*?\}', '', clean_res)
            answer = f"{clean_res.strip()}\n\n[Engine: Groq Agents 🤖]"
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg:
                answer = "Bhai, Groq ki Rate Limit hit ho gayi hai (100k tokens over). Kripya UI se 'Gemini Mode' switch karein!"
            else:
                answer = f"CrewAI/Groq Error: {str(e)}. Please switch to Gemini Mode."
            print(traceback.format_exc())

    # 3. Save to Database
    new_entry = ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=answer)
    db.add(new_entry)
    db.commit()
    
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "Bulletproof Dual-Engine AI Backend is Live!"}
