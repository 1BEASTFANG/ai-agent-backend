import os
from datetime import datetime
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_history.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
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

# --- KEY ROTATION LOGIC ---
def get_groq_llm(key_index):
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    valid_keys = [k for k in keys if k]
    if not valid_keys:
        raise ValueError("Render par API Keys missing hain!")
    selected_key = valid_keys[key_index % len(valid_keys)]
    return LLM(model="groq/llama-3.1-8b-instant", api_key=selected_key)

def get_total_valid_keys():
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    return len([k for k in keys if k])

class UserRequest(BaseModel):
    question: str

# --- 🚀 MEGA TOPIC ROUTER (Highly Expanded) ---
def detect_category(text):
    text = text.lower()
    
    # 1. Data Science & Analysis
    data_words = ['data', 'pandas', 'matplotlib', 'seaborn', 'csv', 'dataset', 'plot', 'graph', 'chart', 'analysis', 'clean', 'visual', 'aggregate', 'calldata', 'diamonds', 'movies', 'dataframe', 'numpy']
    # 2. Core CS, DSA, Graphics & Backend
    coding_words = ['code', 'python', 'c++', 'django', 'error', 'bug', 'script', 'function', 'logic', 'dsa', 'render', 'ubuntu', 'api', 'opengl', 'vulkan', 'imgui', 'linked list', 'doubly linked list', 'node', 'pointer', 'stack', 'queue', 'backend', 'deploy', 'program', 'music player']
    # 3. College, Academics & Presentations
    college_words = ['college', 'assignment', 'presentation', 'ppt', 'slide', 'sdg', 'sustainable', 'goals', 'physical science', 'physics', 'exam', 'study', 'notes', 'project', 'andc', 'acharya narendra dev']
    # 4. News & Current Affairs
    news_words = ['news', 'aaj', 'khabar', 'match', 'samachar', 'update', 'latest', 'current affairs', 'duniya', 'world', 'india', 'headline', 'summit', 'event', 'today']
    # 5. Location, Weather & Geography
    location_words = ['location', 'kaha', 'kahan', 'delhi', 'weather', 'map', 'distance', 'place', 'city', 'country', 'address', 'mausam', 'direction', 'rasta']
    # 6. Math & Calculations
    math_words = ['math', 'calculate', 'calculation', 'formula', 'solve', 'equation', 'plus', 'minus', 'multiply', 'divide', 'algebra', 'calculus', 'integration', 'derivation', 'matrix', 'maths']
    
    # Priority matching
    if any(word in text for word in data_words): return 'data_science'
    if any(word in text for word in coding_words): return 'coding'
    if any(word in text for word in college_words): return 'college'
    if any(word in text for word in news_words): return 'news'
    if any(word in text for word in location_words): return 'location'
    if any(word in text for word in math_words): return 'math'
    
    return 'general'

@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    past_messages = db.query(ChatMessage).order_by(ChatMessage.id.desc()).limit(4).all()
    
    current_category = detect_category(request.question)
    history_str = ""
    
    if past_messages:
        last_msg_category = detect_category(past_messages[0].user_query)
        
        # SMART DROP: Memory cleared if topic changes!
        if current_category != 'general' and last_msg_category != 'general' and current_category != last_msg_category:
            print(f"INFO: Topic shift ({last_msg_category} -> {current_category}). Memory flushed to save tokens!")
            history_str = "[System: Topic changed by user. Previous context cleared for efficiency.]\n"
        else:
            for m in reversed(past_messages):
                clean_response = m.ai_response.split("\n\n[Key:")[0]
                history_str += f"User: {m.user_query}\nAgent: {clean_response}\n"

    answer = "Bhai, saari keys busy hain. Thoda wait kar le."
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    total_keys = get_total_valid_keys()
    loop_count = total_keys if total_keys > 0 else 1 

    for i in range(loop_count):
        try:
            current_llm = get_groq_llm(i)
            
            backstory_text = (
                f"Date: {current_date}. Aap Nikhil Yadav aur Arvind Kumar ke smart AI dost ho. "
                "TOP 10 SNIPER RULES (FOLLOW STRICTLY): "
                "1. SCRIPT: Use ONLY Roman Hinglish (e.g., 'kya haal hai'). NEVER use Devanagari Hindi. "
                "2. CONCISENESS: For casual chat, reply naturally in just 1 line. "
                "3. TOOL SECRECY: NEVER output internal code like '-function=internet_search>'. Keep it hidden. "
                "4. FACTS/NEWS: Search the web and give 3-4 crisp bullet points. No fluff. "
                "5. CODING: Provide Markdown code ONLY if explicitly asked. Keep comments clear. "
                "6. NO FLUFF: Never say 'I am an AI' or 'I apologize'. Start directly. "
                "7. EXPERT: Maintain high accuracy for CS, Math, and Data Science queries. "
                "8. CONTEXT AWARE: Read chat history. Do not repeat the same phrases. "
                "9. MIRRORING: If user is short, be short. If user asks details, be detailed. "
                "10. NO KEY TAGS: NEVER type '[Key: X]' yourself. The backend handles it. "
                f"\n--- Chat History ---\n{history_str}\n-------------------"
            )
            
            smart_agent = Agent(
                role='Pro AI Mentor',
                goal='To answer flawlessly using the 10 Sniper Rules without showing internal tags.',
                backstory=backstory_text,
                tools=[search_tool],
                llm=current_llm,
                verbose=False
            )
            
            task_desc = f"User: {request.question}. Apply the 10 Sniper Rules. Do NOT leak tool tags."
            task = Task(description=task_desc, expected_output="Clean Hinglish response.", agent=smart_agent)
            
            raw_answer = str(Crew(agents=[smart_agent], tasks=[task]).kickoff())
            
            if raw_answer and not raw_answer.startswith("Agent stopped"):
                 raw_answer = raw_answer.replace("-function=internet_search>", "").strip()
                 
                 total_chars = len(backstory_text) + len(task_desc) + len(raw_answer)
                 approx_tokens = int(total_chars / 4) 
                 
                 answer = f"{raw_answer}\n\n[Key: {i+1} | Est. Tokens: {approx_tokens}]"
                 break 
        except Exception as e:
            print(f"WARN: Key #{i+1} failed. Error: {e}")
            continue

    new_entry = ChatMessage(user_query=request.question, ai_response=answer)
    db.add(new_entry)
    db.commit()
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "Bilingual AI (Mega Micro-Router Edition) is Live!"}
