import os
import re
from datetime import datetime
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Text, String
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 🚀 Training data import
try:
    from training_data import TRAIN_DATA
except ImportError:
    TRAIN_DATA = [("hello", "general"), ("code python", "coding")]

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

# --- HELPER FUNCTIONS ---
def get_groq_llm(key_index):
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    valid_keys = [k for k in keys if k]
    if not valid_keys: raise ValueError("API Keys missing hain!")
    
    # 🚀 MODEL UPGRADED TO 70B
    return LLM(
        model="groq/llama-3.3-70b-versatile", 
        api_key=valid_keys[key_index % len(valid_keys)],
        base_url="https://api.groq.com/openai/v1",
        temperature=0.4 
    )

def get_total_valid_keys():
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    return len([k for k in keys if k])

# --- ML ROUTER ---
texts, labels = zip(*TRAIN_DATA)
ml_router = make_pipeline(TfidfVectorizer(), MultinomialNB())
ml_router.fit(texts, labels)

def detect_category(text):
    return ml_router.predict([text.lower()])[0]

class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str

# --- MAIN API ENDPOINT ---
@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    past_messages = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(4).all()
    
    # 🚀 CONFIRMATION LOGIC
    user_q_lower = request.question.lower().strip()
    is_confirmation = user_q_lower in ["haa", "ha", "yes", "ji", "ok", "theek", "batao", "bataiye"]
    current_category = detect_category(request.question)
    
    history_str = ""
    if past_messages:
        last_msg_category = detect_category(past_messages[0].user_query)
        # Topic switch logic: Confirmation words don't trigger topic switch alert
        if not is_confirmation and current_category != last_msg_category and current_category != 'general':
            history_str = "[System Alert: User switched the topic. Forget previous memory.]\n"
        else:
            for m in reversed(past_messages):
                clean_ai_resp = re.sub(r'<.*?>', '', m.ai_response).split('\n\n[Key:')[0].strip()
                history_str += f"User: {m.user_query}\nAgent: {clean_ai_resp}\n"

    answer = "Bhai, saari keys busy hain. Thoda wait kar le."
    total_keys = get_total_valid_keys()

    for i in range(total_keys if total_keys > 0 else 1):
        try:
            current_llm = get_groq_llm(i)
            
            backstory_text = (
                f"You are a highly capable AI expert talking to your friend '{request.user_name}'. "
                "CRITICAL RULES: "
                f"1. NAME: Always call the user '{request.user_name}'. "
                "2. NO NARRATION: Just give the direct answer without saying 'I am searching'. "
                "3. SEARCH RULE: Always use internet_search for recommendations or current events. "
                "4. NO GIVING UP: Be persistent. Summarize web info if perfect answer isn't found. "
                "5. LANGUAGE: Natural Hinglish. "
                "6. CODE: Triple backticks (```). "
                f"\n--- Chat History ---\n{history_str}\n-------------------"
            )
            
            smart_agent = Agent(
                role='Expert Researcher',
                goal=f'Provide detailed, fact-checked Hinglish answers to {request.user_name}.',
                backstory=backstory_text,
                tools=[search_tool],
                llm=current_llm,
                verbose=False
            )
            
            # 🚀 TASK LOGIC FOR CONFIRMATIONS
            effective_query = request.question
            if is_confirmation and past_messages:
                effective_query = f"Provide full details for: {past_messages[0].user_query}"

            task = Task(
                description=f"User asks: {effective_query}. Provide a high-quality, research-based response.",
                expected_output="A helpful Hinglish response.",
                agent=smart_agent
            )
            
            raw_answer = str(Crew(agents=[smart_agent], tasks=[task]).kickoff())
            
            if raw_answer and not raw_answer.startswith("Agent stopped"):
                clean_answer = re.sub(r'<.*?>', '', raw_answer)
                clean_answer = re.sub(r'function=.*?>', '', clean_answer)
                approx_tokens = int((len(backstory_text) + len(clean_answer)) / 4) 
                answer = f"{clean_answer}\n\n[Key: {i+1} | Est. Tokens: {approx_tokens}]"
                break 

        except Exception as e:
            print(f"DEBUG: Error with Key {i+1}: {str(e)}")
            continue

    new_entry = ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=answer)
    db.add(new_entry)
    db.commit()
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "Nikhil's AI Backend is Live!"}
