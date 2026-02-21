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

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_history_v3.db")
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

# 🚀 NAYI TABLE: User ki pasand yaad rakhne ke liye
class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String, unique=True, index=True)
    persona = Column(Text, default="A new user. Learn their preferences over time.")

Base.metadata.create_all(bind=engine)
app = FastAPI()

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- SEARCH TOOL ---
class MySearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Use this for real-time factual info from the web."
    def _run(self, query: str) -> str:
        return SerperDevTool().run(search_query=str(query))

search_tool = MySearchTool()

# ==========================================
# 🚀 HELPER FUNCTIONS: MANAGER & WORKER POOLS
# ==========================================

def get_manager_llm(key_index):
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 6)]
    valid_keys = [k for k in keys if k]
    if not valid_keys: raise ValueError("Manager ke liye API Keys (1-5) missing hain!")
    
    return LLM(
        model="groq/llama-3.1-8b-instant", 
        api_key=valid_keys[key_index % len(valid_keys)],
        base_url="https://api.groq.com/openai/v1",
        temperature=0.1 
    )

def get_total_manager_keys():
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 6)]
    return len([k for k in keys if k])

def get_worker_llm(key_index):
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(6, 51)]
    valid_keys = [k for k in keys if k]
    if not valid_keys: raise ValueError("Worker ke liye API Keys (6-50) missing hain!")
    
    return LLM(
        model="groq/llama-3.3-70b-versatile", 
        api_key=valid_keys[key_index % len(valid_keys)],
        base_url="https://api.groq.com/openai/v1",
        temperature=0.4 
    )

def get_total_worker_keys():
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(6, 51)]
    return len([k for k in keys if k])

class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str

# --- MAIN API ENDPOINT ---
@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    # 1. Purani history nikalo (Last 3 messages)
    past_messages = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(3).all()
    
    # 🚀 NAYA: Database se User ka Profile nikalo
    user_profile = db.query(UserProfile).filter(UserProfile.user_name == request.user_name).first()
    if not user_profile:
        user_profile = UserProfile(user_name=request.user_name)
        db.add(user_profile)
        db.commit()
        db.refresh(user_profile)
    
    history_str = ""
    if past_messages:
        for m in reversed(past_messages):
            clean_ai_resp = re.sub(r'<.*?>', '', m.ai_response).split('\n\n[')[0].strip()
            history_str += f"User: {m.user_query}\nAgent: {clean_ai_resp}\n"

    # ==========================================
    # 🚀 PHASE 1: THE MANAGER AI (With Dynamic Persona)
    # ==========================================
    manager_prompt = f"""
    Analyze the user's query and update their long-term persona if needed.
    
    Chat History:
    {history_str if history_str else "No history yet."}
    
    Current User Persona (What we know about their preferences so far):
    "{user_profile.persona}"
    
    New User Query: "{request.question}"
    
    Task:
    1. Check if the New Query is a continuation of the Chat History.
    2. Determine the formatting rules the responding AI should follow.
    3. Update the 'Current User Persona' if the user expresses a new preference (e.g., they ask for short answers, code only, specific languages, etc.). Keep it as a concise summary.
    
    RETURN STRICTLY A VALID JSON OBJECT WITH NO OTHER TEXT:
    {{
        "is_new_topic": true or false,
        "category": "string",
        "worker_instructions": "string (Specific rules for the responding AI for this query)",
        "updated_user_persona": "string (The updated long-term preferences of this user)"
    }}
    """
    
    total_manager_keys = get_total_manager_keys()
    manager_data = None
    manager_used_key = 1

    for i in range(total_manager_keys if total_manager_keys > 0 else 1):
        try:
            manager_llm = get_manager_llm(i)
            manager_response = manager_llm.call(messages=[{"role": "user", "content": manager_prompt}])
            json_str = re.sub(r"```json|```", "", manager_response).strip()
            manager_data = json.loads(json_str)
            manager_used_key = i + 1 
            break 
        except Exception as e:
            print(f"DEBUG: Manager Error with Key {i+1}: {str(e)}")
            continue

    if not manager_data:
        manager_data = {
            "is_new_topic": False, 
            "category": "general", 
            "worker_instructions": "Provide a helpful response.",
            "updated_user_persona": user_profile.persona
        }
        manager_used_key = "Failsafe"

    # 🚀 NAYA: User ki nayi pasand Database mein hamesha ke liye save karein
    if manager_data.get("updated_user_persona"):
        user_profile.persona = manager_data.get("updated_user_persona")
        db.commit()

    # ==========================================
    # 🚀 PHASE 2: MEMORY MANAGEMENT 
    # ==========================================
    final_history = history_str
    if manager_data.get("is_new_topic") == True:
        final_history = "[System Alert: User switched to a completely new topic. Ignore the context of previous messages and focus ONLY on the new query.]\n"

    # ==========================================
    # 🚀 PHASE 3: THE WORKER AI (Guided by Persona)
    # ==========================================
    answer = "Bhai, saari keys busy hain. Thoda wait kar le."
    total_worker_keys = get_total_worker_keys()

    for i in range(total_worker_keys if total_worker_keys > 0 else 1):
        try:
            worker_llm = get_worker_llm(i)
            
            backstory_text = (
                f"You are a highly capable AI expert talking to your friend '{request.user_name}'. "
                "CRITICAL RULES: "
                f"1. NAME: Always call the user '{request.user_name}'. "
                "2. NO NARRATION: Never narrate your actions. Just give the answer. "
                "3. SEARCH RULE: Use internet_search for factual data or recommendations. "
                "4. LANGUAGE: Natural Hinglish. "
                "5. CODE: Triple backticks (```). "
                f"\n👤 USER PERSONA (Strictly adapt your tone and style to this): {user_profile.persona}"
                f"\n🔥 MANAGER'S STRICT INSTRUCTIONS FOR THIS PROMPT: {manager_data.get('worker_instructions')} 🔥"
                f"\n--- Chat History ---\n{final_history}\n-------------------"
            )
            
            smart_agent = Agent(
                role='Expert Responder',
                goal=f'Provide the perfect response tailored exactly to the User Persona.',
                backstory=backstory_text,
                tools=[search_tool],
                llm=worker_llm,
                verbose=False
            )
            
            task = Task(
                description=f"User asks: {request.question}. Formulate your response based strictly on the Manager's Instructions and the User Persona.",
                expected_output="A helpful Hinglish response tailored to the user's exact needs and preferences.",
                agent=smart_agent
            )
            
            raw_answer = str(Crew(agents=[smart_agent], tasks=[task]).kickoff())
            
            if raw_answer and not raw_answer.startswith("Agent stopped"):
                clean_answer = re.sub(r'<.*?>', '', raw_answer)
                clean_answer = re.sub(r'function=.*?>', '', clean_answer)
                
                approx_tokens = int((len(backstory_text) + len(clean_answer)) / 4) 
                answer = f"{clean_answer}\n\n[M-Key: {manager_used_key} | W-Key: {i+6} | Est. Tokens: {approx_tokens}]"
                break 

        except Exception as e:
            print(f"DEBUG: Worker Error with Key {i+6}: {str(e)}")
            continue

    new_entry = ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=answer)
    db.add(new_entry)
    db.commit()
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "Dynamic Persona AI Backend is Live!"}
