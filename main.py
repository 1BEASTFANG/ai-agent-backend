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

# 🚀 FIX 1: CrewAI ka 20 second wala prompt hamesha ke liye band
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

# --- SEARCH TOOL ---
class MySearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Use this for real-time factual info from the web like news, prices, or specs."
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
    return LLM(model="groq/llama-3.1-8b-instant", api_key=valid_keys[key_index % len(valid_keys)], base_url="https://api.groq.com/openai/v1", temperature=0.1)

def get_total_manager_keys():
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 6)]
    return len([k for k in keys if k])

def get_worker_llm(key_index):
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(6, 51)]
    valid_keys = [k for k in keys if k]
    if not valid_keys: raise ValueError("Worker ke liye API Keys (6-50) missing hain!")
    return LLM(model="groq/llama-3.3-70b-versatile", api_key=valid_keys[key_index % len(valid_keys)], base_url="https://api.groq.com/openai/v1", temperature=0.3)

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

    # 🚀 NAYA: Current Time System ko dena taaki search na karna pade
    current_time_str = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")

    # ==========================================
    # 🚀 PHASE 1: THE STRICT MANAGER AI
    # ==========================================
    manager_prompt = f"""
    Analyze the New User Query and the Chat History.
    
    Current Date & Time: {current_time_str}
    Chat History:
    {history_str if history_str else "No history yet."}
    Current User Persona: "{user_profile.persona}"
    New User Query: "{request.question}"
    
    Task:
    1. CATEGORIZE: Identify the topic of the query.
    2. EFFECTIVE QUERY: If the New Query is just "Haa", "ok karo", "batao", look at the Chat History to see what the user actually wants and REWRITE it as a full question (e.g., "Give me the list of phones under 10000").
    3. RELEVANT HISTORY: Extract ONLY past messages that share the SAME category.
    4. PERSONA UPDATE: Update ONLY stylistic preferences (e.g., "short answers"). Do NOT add topics to the persona.
    
    RETURN STRICTLY A VALID JSON OBJECT WITH NO OTHER TEXT:
    {{
        "category": "string",
        "effective_query": "string (Crucial: Must be a complete descriptive question)",
        "relevant_history": "string",
        "worker_instructions": "string",
        "updated_user_persona": "string"
    }}
    """
    
    total_manager_keys = get_total_manager_keys()
    manager_data = None
    manager_used_key = 1

    for i in range(total_manager_keys if total_manager_keys > 0 else 1):
        try:
            manager_llm = get_manager_llm(i)
            manager_response = str(manager_llm.call(messages=[{"role": "user", "content": manager_prompt}]))
            
            start_idx = manager_response.find('{')
            end_idx = manager_response.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = manager_response[start_idx:end_idx+1]
                manager_data = json.loads(json_str)
                manager_used_key = i + 1 
                break 
            else:
                raise ValueError("Valid JSON not found")
        except Exception as e:
            print(f"DEBUG: Manager Error Key {i+1}: {str(e)}")
            continue

    if not manager_data:
        manager_data = {
            "category": "general",
            "effective_query": request.question,
            "relevant_history": "", 
            "worker_instructions": "Provide a direct response.",
            "updated_user_persona": user_profile.persona
        }
        manager_used_key = "Failsafe"

    if manager_data.get("updated_user_persona"):
        user_profile.persona = manager_data.get("updated_user_persona")
        db.commit()

    # ==========================================
    # 🚀 PHASE 2: MEMORY MANAGEMENT 
    # ==========================================
    filtered_history = manager_data.get("relevant_history", "")
    final_history = f"[Relevant Past Context:\n{filtered_history}]" if filtered_history.strip() else "[No past context.]"

    actual_intent = manager_data.get("effective_query", request.question)
    query_category = manager_data.get("category", "general")

    # ==========================================
    # 🚀 PHASE 3: THE WORKER AI (Anti-Narration Rules)
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
                f"2. CURRENT TIME: Today is {current_time_str}. DO NOT use the search tool if the user asks for the date, day, or time. You already know it! "
                "3. NO NARRATION / NO PERMISSION: NEVER say 'Main internet search karunga' or 'Main tool use karta hoon'. DO NOT ask for permission. Just do the search silently and give the final result immediately. "
                "4. DIRECT ACTION: If the user says 'haa batao' or 'ok karo', it means you failed to provide info earlier. Provide the actual information immediately without any excuses. "
                "5. NO TOOL LEAKS: NEVER output raw JSON or code like {\"query\": \"...\"}. "
                f"6. TOPIC FOCUS: The current topic is '{query_category}'. "
                "7. LANGUAGE: Natural Hinglish. "
                f"\n👤 USER PERSONA: {user_profile.persona}"
                f"\n🔥 MANAGER'S INSTRUCTIONS: {manager_data.get('worker_instructions')} 🔥"
                f"\n--- Context ---\n{final_history}\n-------------------"
            )
            
            smart_agent = Agent(
                role='Expert Responder',
                goal='Provide the final, direct answer immediately. Never narrate actions or ask for permission.',
                backstory=backstory_text,
                tools=[search_tool],
                llm=worker_llm,
                max_iter=4, 
                verbose=False
            )
            
            task = Task(
                description=f"User's actual intent: {actual_intent}. Provide the final, direct answer. Do NOT tell the user what you are doing.",
                expected_output="A clean, helpful Hinglish response.",
                agent=smart_agent
            )
            
            raw_answer = str(Crew(agents=[smart_agent], tasks=[task]).kickoff())
            
            if raw_answer and not raw_answer.startswith("Agent stopped"):
                # Safety strip for JSON leaks
                clean_answer = re.sub(r'```json.*?```', '', raw_answer, flags=re.DOTALL)
                clean_answer = re.sub(r'\{"query".*?\}', '', clean_answer)
                clean_answer = re.sub(r'<.*?>', '', clean_answer).strip()
                
                approx_tokens = int((len(backstory_text) + len(clean_answer)) / 4) 
                context_status = "Used" if filtered_history.strip() else "Hidden"
                answer = f"{clean_answer}\n\n[M-Key: {manager_used_key} | W-Key: {i+6} | Ctx: {context_status} | Tok: {approx_tokens}]"
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
ey {i+6}: {str(e)}")
            continue

    new_entry = ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=answer)
    db.add(new_entry)
    db.commit()
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "Dynamic Persona AI Backend is Live!"}
