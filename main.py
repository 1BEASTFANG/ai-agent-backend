import os
import re  # 🚀 Regex cleaning ke liye zaroori hai
from datetime import datetime
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Text, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
# Baaki saare imports ke theek niche isey lagayein:
from training_data import TRAIN_DATA

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

def get_groq_llm(key_index):
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    valid_keys = [k for k in keys if k]
    if not valid_keys: raise ValueError("Render par API Keys missing hain!")
    
    return LLM(
        model="groq/llama-3.1-8b-instant", 
        api_key=valid_keys[key_index % len(valid_keys)],
        base_url="https://api.groq.com/openai/v1",
        temperature=0.3  # 🚀 MAGIC WAND: Isse AI overacting aur fake words banana band kar dega
    )

def get_total_valid_keys():
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    return len([k for k in keys if k])

class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str

# --- MASSIVE ML ROUTER DATASET (Ultra-High Accuracy) ---


texts, labels = zip(*TRAIN_DATA)
ml_router = make_pipeline(TfidfVectorizer(), MultinomialNB())
ml_router.fit(texts, labels)

def detect_category(text):
    return ml_router.predict([text.lower()])[0]

def is_similar(current_q, past_q):
    words_current = set(current_q.lower().split())
    words_past = set(past_q.lower().split())
    if not words_current or not words_past: return False
    overlap = len(words_current.intersection(words_past))
    return (overlap / len(words_current)) >= 0.50

@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    past_messages = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(4).all()
    current_category = detect_category(request.question)
    history_str = ""
    
    if past_messages:
        last_msg_category = detect_category(past_messages[0].user_query)
        
        # 🚀 SMART MEMORY MANAGER: Topic badalte hi Memory Delete!
        if current_category != last_msg_category and current_category != 'general':
            history_str = "[System Alert: User switched the topic. Forget all previous chat memory and focus ONLY on the new query.]\n"
        else:
            for m in reversed(past_messages):
                clean_ai_resp = re.sub(r'<.*?>', '', m.ai_response).split('\n\n[Key:')[0].strip()
                history_str += f"User: {m.user_query}\nAgent: {clean_ai_resp}\n"

    answer = "Bhai, saari keys busy hain. Thoda wait kar le."
    current_date = datetime.now().strftime("%Y-%m-%d")
    total_keys = get_total_valid_keys()

    for i in range(total_keys if total_keys > 0 else 1):
        try:
            current_llm = get_groq_llm(i)
            
            backstory_text = (
                f"You are a smart, highly accurate, and helpful AI assistant talking to your friend {request.user_name}. "
                "CRITICAL RULES FOR YOUR RESPONSE: "
                "1. LANGUAGE: You MUST reply in natural, everyday 'Hinglish' (Hindi spoken in daily life, written in the English alphabet). "
                "2. TONE: Be friendly, clear, and direct. Do NOT use highly formal pure Hindi words (avoid words like 'uplabdh', 'vishal', 'shasit'). "
                "3. NO FAKE SLANG: Do NOT invent weird words (like 'khanos' or 'nayaan'). Speak simply. "
                "4. MATH & LOGIC: If asked a math calculation, solve it step-by-step using standard English math terms (like 'multiply', 'subtract', 'BODMAS rule'). Do NOT translate math terms into weird Hindi. "
                f"5. ADDRESSING: Politely address the user by their name ({request.user_name}) in a natural way. "
                "6. CLEAN OUTPUT: Never output <function> tags, XML, or internal JSON. "
                f"\n--- Chat History ---\n{history_str}\n-------------------"
            )
            
            smart_agent = Agent(
                role='AI Assistant',
                goal=f'Provide highly accurate and natural Hinglish answers to {request.user_name}.',
                backstory=backstory_text,
                tools=[search_tool],
                llm=current_llm,
                verbose=False
            )
            
            task_desc = f"User ({request.user_name}) asks: {request.question}. Provide a smart, logically correct, and natural Hinglish response."
            task = Task(description=task_desc, expected_output="A clean, logical Hinglish response.", agent=smart_agent)
            
            raw_answer = str(Crew(agents=[smart_agent], tasks=[task]).kickoff())
            
            if raw_answer and not raw_answer.startswith("Agent stopped"):
                clean_answer = re.sub(r'<.*?>', '', raw_answer)
                clean_answer = re.sub(r'function=.*?>', '', clean_answer)
                clean_answer = clean_answer.replace('{ "code":', '').replace('}', '').strip()
                
                approx_tokens = int((len(backstory_text) + len(task_desc) + len(clean_answer)) / 4) 
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
    return {"message": "Multi-Tenant AI is Live!"}
