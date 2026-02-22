import os
import re
import json
from datetime import datetime
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Text, String
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool

# 🚀 FIX: CrewAI settings
os.environ["CREWAI_TRACING_ENABLED"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_history_v4.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, default="default_user")
    user_query = Column(Text)
    ai_response = Column(Text)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String, unique=True, index=True)
    persona = Column(Text, default="A friendly user. Adjust tone over time.")

Base.metadata.create_all(bind=engine)
app = FastAPI()

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- SEARCH TOOL ---
class MySearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Use this for real-time factual info from the web like news or prices."
    def _run(self, query: str) -> str:
        return SerperDevTool().run(search_query=str(query))

search_tool = MySearchTool()

# ==========================================
# 🚀 THE LLM POOLS (Hybrid Setup)
# ==========================================

# 🧠 PREMIUM (Gemini 1.5 Pro) - Manager aur Critic ke liye
def get_premium_llm():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key: raise ValueError("GEMINI_API_KEY missing!")
    return LLM(model="gemini/gemini-1.5-pro", api_key=key, temperature=0.2)

# 📚 LIBRARIAN (Groq 8B) - Keys 1 to 5
def get_librarian_llm(index):
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 6)]
    valid = [k for k in keys if k]
    return LLM(model="groq/llama-3.1-8b-instant", api_key=valid[index % len(valid) if valid else 0], temperature=0.1)

# 👷 WORKER (Groq 70B) - Keys 6 to 50
def get_worker_llm(index):
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(6, 51)]
    valid = [k for k in keys if k]
    return LLM(model="groq/llama-3.3-70b-versatile", api_key=valid[index % len(valid) if valid else 0], temperature=0.3)

class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str

def extract_json(response_text):
    start = response_text.find('{')
    end = response_text.rfind('}')
    if start != -1 and end != -1:
        try: return json.loads(response_text[start:end+1])
        except: return None
    return None

# --- MAIN API ENDPOINT ---
@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    past_messages = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(10).all()
    user_profile = db.query(UserProfile).filter(UserProfile.user_name == request.user_name).first() or UserProfile(user_name=request.user_name)
    
    history_str = ""
    if past_messages:
        for i, m in enumerate(reversed(past_messages)):
            clean_resp = re.sub(r'<.*?>', '', m.ai_response).split('\n\n[')[0].strip()
            history_str += f"[{i+1}] User: {m.user_query}\nAgent: {clean_resp}\n"

    current_time_str = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")

    # 📚 PHASE 1: LIBRARIAN (Groq 8B - Context Filtering)
    librarian_prompt = f"Time: {current_time_str}\nHistory:\n{history_str}\nQuery: {request.question}\nTask: Extract related context. Return JSON: {{'relevant_history': 'string'}}"
    librarian_data = {"relevant_history": ""}
    try:
        lib_res = str(get_librarian_llm(0).call(messages=[{"role": "user", "content": librarian_prompt}]))
        parsed = extract_json(lib_res)
        if parsed: librarian_data = parsed
    except: pass

    # 👔 PHASE 2: MANAGER (Gemini 1.5 Pro - Strategy)
    manager_prompt = f"""
    Analyze the user intent like a senior strategist.
    Current Time: {current_time_str}
    Context: {librarian_data['relevant_history']}
    New Query: {request.question}
    Persona: {user_profile.persona}

    Task:
    1. Categorize topic strictly.
    2. Create a perfect 'effective_query' for the worker.
    3. Write elite 'worker_instructions': Hinglish only, NO permission seeking, NO narration.
    4. Update persona only for style preferences (e.g., 'keep it short').

    RETURN JSON: {{'category': 'string', 'effective_query': 'string', 'worker_instructions': 'string', 'updated_user_persona': 'string'}}
    """
    manager_data = {"category": "general", "effective_query": request.question, "worker_instructions": "Be direct and helpful."}
    try:
        mgr_res = str(get_premium_llm().call(messages=[{"role": "user", "content": manager_prompt}]))
        parsed = extract_json(mgr_res)
        if parsed: 
            manager_data = parsed
            user_profile.persona = manager_data.get("updated_user_persona", user_profile.persona)
            db.add(user_profile)
            db.commit()
    except: pass

    # 👷 PHASE 3: WORKER (Groq 70B - Execution)
    final_context = f"[Relevant Context: {librarian_data['relevant_history']}]" if librarian_data['relevant_history'] else "[No history context]"
    backstory = (
        f"You are talking to {request.user_name}. Topic: {manager_data['category']}. "
        f"Today: {current_time_str}. Hinglish only. NO NARRATION (Don't say 'Searching...'). "
        f"RULES: {manager_data['worker_instructions']}"
    )
    worker_agent = Agent(role='Executor', goal='Best answer.', backstory=backstory, tools=[search_tool], llm=get_worker_llm(0), verbose=False)
    worker_task = Task(description=f"Task: {manager_data['effective_query']}\n{final_context}", expected_output="Clean Hinglish response.", agent=worker_agent)
    
    raw_answer = str(Crew(agents=[worker_agent], tasks=[worker_task]).kickoff())

    # 🕵️‍♂️ PHASE 4: THE CRITIC (Gemini 1.5 Pro - Quality Control)
    critic_prompt = f"""
    User Intent: {request.question}
    Worker Response: {raw_answer}
    Task: Fix any 'Searching...' narration, ensure natural Hinglish, remove JSON leaks, and polish for a Gemini-like expert tone.
    Return ONLY the final polished text for the user.
    """
    final_answer = raw_answer
    try:
        final_answer = str(get_premium_llm().call(messages=[{"role": "user", "content": critic_prompt}]))
    except: pass

    # Final Safety Clean
    final_answer = re.sub(r'```json.*?```', '', final_answer, flags=re.DOTALL)
    final_answer = final_answer.replace('{', '').replace('}', '').strip()
    
    final_response = f"{final_answer}\n\n[M: Premium | W: 70B | C: Premium]"
    
    new_entry = ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=final_response)
    db.add(new_entry)
    db.commit()
    return {"answer": final_response}

@app.get("/")
def root(): return {"message": "Hybrid 4-Tier Hierarchical AI Backend is Live!"}
