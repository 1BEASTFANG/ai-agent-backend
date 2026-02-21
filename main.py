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

# 🚀 FIX: CrewAI ka tracing aur timeout prompts band
os.environ["CREWAI_TRACING_ENABLED"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"

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

# --- TOOLS ---
class MySearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Use this for real-time factual info from the web like news, prices, or specs."
    def _run(self, query: str) -> str:
        return SerperDevTool().run(search_query=str(query))

search_tool = MySearchTool()

# ==========================================
# 🚀 HELPER FUNCTIONS: 3 AI POOLS
# ==========================================

def get_librarian_llm(key_index):
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 3)]
    valid_keys = [k for k in keys if k]
    return LLM(model="groq/llama-3.1-8b-instant", api_key=valid_keys[key_index % len(valid_keys) if valid_keys else 0], base_url="https://api.groq.com/openai/v1", temperature=0.1)

def get_manager_llm(key_index):
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(3, 6)]
    valid_keys = [k for k in keys if k]
    return LLM(model="groq/llama-3.1-8b-instant", api_key=valid_keys[key_index % len(valid_keys) if valid_keys else 0], base_url="https://api.groq.com/openai/v1", temperature=0.1)

def get_worker_llm(key_index):
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(6, 51)]
    valid_keys = [k for k in keys if k]
    return LLM(model="groq/llama-3.3-70b-versatile", api_key=valid_keys[key_index % len(valid_keys) if valid_keys else 0], base_url="https://api.groq.com/openai/v1", temperature=0.3)

class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str

def extract_json(response_text):
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}')
    if start_idx != -1 and end_idx != -1:
        try: return json.loads(response_text[start_idx:end_idx+1])
        except: return None
    return None

# --- MAIN API ENDPOINT ---
@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    past_messages = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(10).all()
    
    user_profile = db.query(UserProfile).filter(UserProfile.user_name == request.user_name).first()
    if not user_profile:
        user_profile = UserProfile(user_name=request.user_name)
        db.add(user_profile)
        db.commit()
        db.refresh(user_profile)
    
    history_str = ""
    if past_messages:
        for i, m in enumerate(reversed(past_messages)):
            clean_ai_resp = re.sub(r'<.*?>', '', m.ai_response).split('\n\n[')[0].strip()
            history_str += f"[{i+1}] User: {m.user_query}\nAgent: {clean_ai_resp}\n"

    current_time_str = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")

    # ==========================================
    # 🚀 PHASE 1: THE LIBRARIAN AI (With Duplication Check)
    # ==========================================
    librarian_prompt = f"""
    Current Time: {current_time_str}
    Chat History: {history_str}
    New Query: "{request.question}"
    
    Task:
    1. Extract ONLY related context.
    2. IMPORTANT: If user asks for joke/story, note PREVIOUS ones from history so Worker doesn't repeat.
    3. Update Persona only for stylistic preferences.
    
    RETURN JSON: {{"relevant_history": "string", "updated_user_persona": "string"}}
    """
    librarian_data = {"relevant_history": "", "updated_user_persona": user_profile.persona}
    for i in range(2):
        try:
            lib_res = str(get_librarian_llm(i).call(messages=[{"role": "user", "content": librarian_prompt}]))
            parsed = extract_json(lib_res)
            if parsed: 
                librarian_data = parsed
                break
        except: continue

    if librarian_data.get("updated_user_persona"):
        user_profile.persona = librarian_data.get("updated_user_persona")
        db.commit()

    # ==========================================
    # 🚀 PHASE 2: THE MANAGER AI (Strategy & Language Lock)
    # ==========================================
    manager_prompt = f"""
    Current Time: {current_time_str}
    Relevant History: "{librarian_data.get('relevant_history', '')}"
    New Query: "{request.question}"
    
    Task:
    1. CATEGORIZE topic.
    2. EFFECTIVE QUERY: Rewrite short queries (e.g. 'haa', 'karo') into full descriptive intents.
    3. LANGUAGE RULE: Enforce Hinglish (50% Hindi, 50% English). No long pure Hindi blocks.
    4. ACTION: Tell Worker NO permission seeking. Just provide results.
    
    RETURN JSON: {{"category": "string", "effective_query": "string", "worker_instructions": "string"}}
    """
    manager_data = {"category": "general", "effective_query": request.question, "worker_instructions": "Be direct and use Hinglish."}
    for i in range(3):
        try:
            mgr_res = str(get_manager_llm(i).call(messages=[{"role": "user", "content": manager_prompt}]))
            parsed = extract_json(mgr_res)
            if parsed:
                manager_data = parsed
                manager_used_key = i + 3
                break
        except: continue

    # ==========================================
    # 🚀 PHASE 3: THE WORKER AI (Anti-Narration Execution)
    # ==========================================
    filtered_history = librarian_data.get("relevant_history", "")
    final_history = f"[Context:\n{filtered_history}]" if filtered_history.strip() else "[No past context.]"
    actual_intent = manager_data.get("effective_query", request.question)
    query_category = manager_data.get("category", "general")

    answer = "Bhai, saari keys busy hain. Thoda wait kar le."
    
    for i in range(45):
        try:
            worker_llm = get_worker_llm(i)
            backstory_text = (
                f"You are {request.user_name}'s expert AI. Today: {current_time_str}. "
                "CRITICAL RULES: "
                "1. NO NARRATION: Never say 'I am searching' or 'Can I tell you?'. Provide results IMMEDIATELY. "
                "2. NO REPETITION: Do not repeat jokes/stories found in context. "
                "3. LANGUAGE: Natural Hinglish (Mix of Hindi/English). Avoid pure Hindi lectures. "
                "4. DIRECT ACTION: If intent is 'haa' or 'batao', show the info NOW. "
                f"TOPIC: '{query_category}'. "
                f"\n👤 PERSONA: {user_profile.persona}"
                f"\n🔥 MANAGER RULES: {manager_data.get('worker_instructions')} 🔥"
                f"\n--- Context ---\n{final_history}\n-------------------"
            )
            
            smart_agent = Agent(role='Expert Responder', goal='Final answer without narration.', backstory=backstory_text, tools=[search_tool], llm=worker_llm, max_iter=4, verbose=False)
            task = Task(description=f"User intent: {actual_intent}. Answer directly.", expected_output="Clean Hinglish response.", agent=smart_agent)
            raw_answer = str(Crew(agents=[smart_agent], tasks=[task]).kickoff())
            
            if raw_answer and not raw_answer.startswith("Agent stopped"):
                clean_answer = re.sub(r'```json.*?```', '', raw_answer, flags=re.DOTALL)
                clean_answer = re.sub(r'\{"query".*?\}', '', clean_answer)
                clean_answer = re.sub(r'<.*?>', '', clean_answer).strip()
                
                approx_tokens = int((len(backstory_text) + len(clean_answer)) / 4) 
                answer = f"{clean_answer}\n\n[L: 1/2 | M: {manager_data.get('category')} | W: {i+6} | Tok: {approx_tokens}]"
                break 
        except Exception as e:
            print(f"DEBUG: Worker Error Key {i+6}: {str(e)}")
            continue

    new_entry = ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=answer)
    db.add(new_entry)
    db.commit()
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "3-Tier Hierarchical AI Backend is Live!"}
