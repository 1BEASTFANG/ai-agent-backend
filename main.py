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

# --- SMART DATABASE SETUP (Anti-Amnesia) ---
# Agar environment variable mein Postgres URL hai toh wo use karega, warna SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_history.db")
# Render Postgres URLs start with postgres:// but sqlalchemy needs postgresql://
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

# --- SEARCH TOOL SETUP ---
class MySearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Use this for real-time factual info from the web."
    def _run(self, query: str) -> str:
        return SerperDevTool().run(search_query=str(query))

search_tool = MySearchTool()

# --- AUTO-SCALING GROQ ROTATION ---
def get_groq_llm(key_index):
    # Dynamic keys fetcher (Supports unlimited keys if you add them in Render)
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    valid_keys = [k for k in keys if k]
    if not valid_keys:
        raise ValueError("Render par koi bhi Groq API Key nahi milti!")
    
    selected_key = valid_keys[key_index % len(valid_keys)]
    return LLM(model="groq/llama-3.3-70b-versatile", api_key=selected_key)

def get_total_valid_keys():
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    return len([k for k in keys if k])

class UserRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    # 🧠 OPTIMIZED MEMORY: 15 messages for best context without draining tokens
    past_messages = db.query(ChatMessage).order_by(ChatMessage.id.desc()).limit(15).all()
    history_str = "".join([f"User: {m.user_query}\nAgent: {m.ai_response}\n" for m in reversed(past_messages)])

    answer = "Maaf kijiye, saari keys abhi busy hain."
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    total_keys = get_total_valid_keys()
    loop_count = total_keys if total_keys > 0 else 1 

    for i in range(loop_count):
        try:
            current_llm = get_groq_llm(i)
            print(f"INFO: Attempting with Key #{i+1}...")
            
            # --- THE PERFECT HINGLISH "BRO" AGENT ---
            smart_agent = Agent(
                role='Tera Smart AI Yaar',
                goal='Nikhil ko chote, natural jawab dena aur sirf mangne par code dena.',
                backstory=(
                    f"Aaj {current_date} hai. Aap Nikhil Yadav (ANDC CS student) ke personal digital dost ho. "
                    "STRICT BEHAVIORAL RULES: "
                    "1. LANGUAGE: Sirf Roman script (English alphabets) mein HINGLISH (e.g., 'Haan bhai') ya pure English use karo. Devnagari script (हिंदी) STRICTLY BAN hai. "
                    "2. EXTREMELY CONCISE: Normal baaton (hi, hello, kaise ho) ka jawab sirf 1 ya 2 line mein do. 'Main ek AI hoon' jaisa robotic intro kabhi mat do. "
                    "3. NO UNPROMPTED CODE: Jab tak user exactly 'code', 'program' ya 'script' na maange, tab tak galti se bhi code block generate mat karna. "
                    "4. CONTEXT AWARENESS: 'Chat History' ko dhyan se padho. Agar user kisi pichle topic ka code maange, toh history se logic uthao aur bina sawal-jawab kiye turant Markdown (```) mein code do. "
                    "5. MIRRORING: Agar user ka message bohot chota hai, toh tumhara reply bhi chota hona chahiye. "
                    f"\n--- Chat History (Last 15 Messages) ---\n{history_str}\n-------------------"
                ),
                tools=[search_tool],
                llm=current_llm,
                verbose=False
            )
            
            task = Task(
                description=(
                    f"Current query: {request.question}. "
                    "1. Read Chat History to understand the context. "
                    "2. Reply in strict Roman Hinglish or English. "
                    "3. Keep it super brief unless code/details are explicitly requested. No Devnagari."
                ),
                expected_output="A very natural, short Hinglish response OR a Markdown code block if explicitly requested.",
                agent=smart_agent
            )
            
            answer = str(Crew(agents=[smart_agent], tasks=[task]).kickoff())
            if answer and not answer.startswith("Agent stopped"):
                 break 
        except Exception as e:
            print(f"WARN: Key #{i+1} failed. Shifting to next key. Error: {e}")
            continue

    if not answer or answer == "Maaf kijiye, saari keys abhi busy hain.":
         answer = "Bhai, abhi system par load hai aur saari API limits khatam ho chuki hain. Thodi der baad try karna!"

    new_entry = ChatMessage(user_query=request.question, ai_response=answer)
    db.add(new_entry)
    db.commit()
    return {"answer": answer}

@app.get("/")
def root(): return {"message": "Bilingual Auto-Scaling Pro Agent is Ready!"}
