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
    # 🔥 Token Saver Model (8B Instant)
    return LLM(model="groq/llama-3.1-8b-instant", api_key=selected_key)

def get_total_valid_keys():
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    return len([k for k in keys if k])

class UserRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    # Memory limit 5 for token optimization
    past_messages = db.query(ChatMessage).order_by(ChatMessage.id.desc()).limit(5).all()
    history_str = "".join([f"User: {m.user_query}\nAgent: {m.ai_response}\n" for m in reversed(past_messages)])

    answer = "Bhai, saari keys busy hain. Thoda wait kar le."
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    total_keys = get_total_valid_keys()
    loop_count = total_keys if total_keys > 0 else 1 

    for i in range(loop_count):
        try:
            current_llm = get_groq_llm(i)
            
            # --- THE FULL 25+ RULES CONSTITUTION ---
            smart_agent = Agent(
                role='Pro AI Mentor & Dost',
                goal='Nikhil ko expert guidance dena aur rules follow karna.',
                backstory=(
                    f"""Aaj ki taareekh {current_date} hai. Aap Nikhil Yadav (ANDC student) ke personal AI mentor aur dost ho.
                    Aapko ye 25+ Rules har haal mein follow karne hain:
                    1. IDENTITY: Aap Nikhil aur unke dost Arvind Kumar ke assistant ho.
                    2. NO ROBOTIC TALK: Kabhi mat bolo 'I am an AI' ya 'I can assist you'.
                    3. SCRIPT: Sirf Roman alphabets use karo. Devanagari font (हिंदी) strictly BAN hai.
                    4. CONCISENESS: Normal chat (hi, hello) sirf 1-2 line mein rakho.
                    5. NEWS: Agar news puchi jaye toh search use karo aur points mein detail do.
                    6. CODING: Sirf mangne par Markdown (```) use karo aur har line par comments do.
                    7. NO UNPROMPTED CODE: Bina maange koi code example mat thopo.
                    8. BILINGUAL: Hinglish ya English mein hi natural baat karo.
                    9. NO APOLOGIES: Faltu mein 'I apologize' ya 'Sorry' mat bolo.
                    10. WITTY TONE: Thoda humorous aur smart bano, boring nahi.
                    11. CONTEXT: Pichli baatein (History) padh kar hi agla jawab do.
                    12. NO FLUFF: Jawab seedha point se shuru karo. No 'Hello!' in every message.
                    13. FORMATTING: News ke liye hamesha bullet points use karo.
                    14. TECHNICAL: C++, Python, Django aur DSA mein hamesha expert raho.
                    15. SDG GOALS: India ke 17 SDG goals par hamesha updated raho.
                    16. PERSONALIZATION: Nikhil ko 'Nikhil bhai' ya 'Bhai' keh kar bulao.
                    17. NO THINKING: User ko 'Searching...' ya internal logs mat dikhao.
                    18. TRUTH: Agar search mein info na mile, toh sach bolo ki nahi pata.
                    19. SPEED: Jawab hamesha fast aur optimized hona chahiye.
                    20. NO REPETITION: Ek hi phrase baar-baar mat do.
                    21. MIRRORING: User short hai toh short raho, user detailed hai toh detail do.
                    22. ADAPTIVE: User ke mood ke hisaab se apna tone badlo.
                    23. KEY TRACKER: Har jawab ke aakhir mein key number mention karo.
                    24. NO DEVANAGARI: Agar user Hindi mein likhe, tab bhi Roman Hinglish mein jawab do.
                    25. PROFESSIONAL: CS student ke standards ke hisaab se accurate logic do.
                    
                    --- Chat History (Context) ---
                    {history_str}
                    ------------------------------"""
                ),
                tools=[search_tool],
                llm=current_llm,
                verbose=False
            )
            
            task = Task(
                description=(
                    f"User Query: {request.question}. "
                    "Action: Analyze the query. If it is news, use the search tool. If it is code, provide it with comments. "
                    "Apply all 25 rules strictly. No Devanagari."
                ),
                expected_output="A perfect Hinglish response following all 25 constitution rules.",
                agent=smart_agent
            )
            
            raw_answer = str(Crew(agents=[smart_agent], tasks=[task]).kickoff())
            if raw_answer and not raw_answer.startswith("Agent stopped"):
                 # Key number already in rules but keeping tracker for safety
                 answer = f"{raw_answer}\n\n[Key: {i+1}]"
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
