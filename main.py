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

class MySearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Use this for real-time factual info from the web."
    def _run(self, query: str) -> str:
        return SerperDevTool().run(search_query=str(query))

search_tool = MySearchTool()

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

@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    past_messages = db.query(ChatMessage).order_by(ChatMessage.id.desc()).limit(5).all()
    history_str = "".join([f"User: {m.user_query}\nAgent: {m.ai_response}\n" for m in reversed(past_messages)])

    answer = "Bhai, saari keys ki limit khatam hai. Thoda wait kar le."
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    total_keys = get_total_valid_keys()
    loop_count = total_keys if total_keys > 0 else 1 

    for i in range(loop_count):
        try:
            current_llm = get_groq_llm(i)
            
            # --- THE 25+ RULES CONSTITUTION ---
            smart_agent = Agent(
                role='Pro AI Mentor & Dost',
                goal='Nikhil ko expert guidance dena aur natural conversation karna.',
                backstory=(
                    f"Aaj {current_date} hai. Aap Nikhil Yadav (ANDC student) ke personal AI mentor ho. "
                    "Aapko niche diye gaye 25+ Rules har haal mein follow karne hain: \n"
                    "1. IDENTITY: Aap Nikhil aur unke dost Arvind Kumar ke personal assistant ho.\n"
                    "2. NO ROBOTIC TALK: Kabhi mat bolo 'I am an AI' ya 'How can I assist you'.\n"
                    "3. SCRIPT: Sirf Roman alphabets use karo. Devanagari font (हिंदी) strictly BAN hai.\n"
                    "4. CONCISENESS: Normal chat 1-2 line mein rakho.\n"
                    "5. NEWS: Agar news puchi jaye toh search use karo aur points mein brief detail do.\n"
                    "6. CODING: Sirf mangne par Markdown (```) use karo. Har line par // ya # comments MANDATORY hain.\n"
                    "7. NO UNPROMPTED CODE: Bina maange coding example mat do.\n"
                    "8. BILINGUAL: Hinglish ya English mein hi baat karo.\n"
                    "9. NO APOLOGIES: Faltu mein 'I apologize' ya 'Sorry' mat bolo unless major mistake ho.\n"
                    "10. WITTY TONE: Thoda humorous aur smart bano, boring nahi.\n"
                    "11. CONTEXT: Pichli baatein (History) padh kar hi agla jawab do.\n"
                    "12. NO FLUFF: Jawab seedha point se shuru karo. No intro/outro.\n"
                    "13. FORMATTING: News ke liye bullet points use karo.\n"
                    "14. TECHNICAL: C++, Python, Django aur DSA (Music player logic) mein expert raho.\n"
                    "15. SDG GOALS: India ke 17 SDG goals par updated raho.\n"
                    "16. PERSONALIZATION: Nikhil ko 'Nikhil bhai' ya 'Bhai' keh kar sambodhit kar sakte ho.\n"
                    "17. NO THINKING PROCESS: User ko apna internal thought process (e.g. 'Searching the web...') mat dikhao.\n"
                    "18. TRUTH: Agar search mein info na mile, toh guess mat karo. Sach bata do.\n"
                    "19. SPEED: Jawab hamesha fast aur optimized hona chahiye.\n"
                    "20. NO REPETITION: Ek hi baat har message mein repeat mat karo.\n"
                    "21. SUGGESTIONS: Agar koi topic complex ho, toh aakhir mein 1 chota suggestion de sakte ho.\n"
                    "22. ADAPTIVE: User gusse mein hai toh calm raho, user excited hai toh energetic.\n"
                    "23. CLEARITY: Complex topics ko 10-year-old bache ki tarah samjhao.\n"
                    "24. SAFETY: Kisi bhi illegal activity mein help mat karo.\n"
                    "25. KEY TRANSPARENCY: Apne jawab ke aakhir mein key number mention karna mandatory hai.\n"
                    f"\n--- History ---\n{history_str}\n-------------------"
                ),
                tools=[search_tool],
                llm=current_llm,
                verbose=False
            )
            
            task = Task(
                description=(
                    f"User Query: {request.question}. "
                    "Task: Detect if it's general chat, news, or code. Apply the 25 Rules accordingly."
                ),
                expected_output="A perfectly formatted, natural response following all 25 rules.",
                agent=smart_agent
            )
            
            raw_answer = str(Crew(agents=[smart_agent], tasks=[task]).kickoff())
            if raw_answer and not raw_answer.startswith("Agent stopped"):
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
def root(): return {"message": "Bilingual AI (25+ Rules Master Edition) Ready!"}
    description: str = "Use this for real-time factual info from the web."
    def _run(self, query: str) -> str:
        return SerperDevTool().run(search_query=str(query))

search_tool = MySearchTool()

def get_groq_llm(key_index):
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    valid_keys = [k for k in keys if k]
    if not valid_keys:
        raise ValueError("Render par koi bhi Groq API Key nahi milti!")
    
    selected_key = valid_keys[key_index % len(valid_keys)]
    return LLM(model="groq/llama-3.1-8b-instant", api_key=selected_key)

def get_total_valid_keys():
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    return len([k for k in keys if k])

class UserRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    past_messages = db.query(ChatMessage).order_by(ChatMessage.id.desc()).limit(5).all()
    history_str = "".join([f"User: {m.user_query}\nAgent: {m.ai_response}\n" for m in reversed(past_messages)])

    answer = "Maaf kijiye, saari keys abhi busy hain."
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    total_keys = get_total_valid_keys()
    loop_count = total_keys if total_keys > 0 else 1 

    for i in range(loop_count):
        try:
            current_llm = get_groq_llm(i)
            print(f"INFO: Attempting with Key #{i+1}...")
            
            smart_agent = Agent(
                role='Tera Smart AI Yaar',
                goal='Nikhil ko chote, natural jawab dena aur sirf mangne par code dena.',
                backstory=(
                    f"Aaj {current_date} hai. Aap Nikhil Yadav (ANDC CS student) ke personal AI dost ho. "
                    "RULES: 1. Language: Roman Hinglish/English (No Devnagari). 2. Concise: Short 1-2 line reply for general talk. "
                    "3. No code unless asked. 4. Use Chat History for context. "
                    f"\n--- Chat History ---\n{history_str}\n-------------------"
                ),
                tools=[search_tool],
                llm=current_llm,
                verbose=False
            )
            
            task = Task(
                description=f"User query: {request.question}. Reply naturally in Hinglish/English. Be super brief.",
                expected_output="Short natural response.",
                agent=smart_agent
            )
            
            raw_answer = str(Crew(agents=[smart_agent], tasks=[task]).kickoff())
            
            if raw_answer and not raw_answer.startswith("Agent stopped"):
                 # ✨ KEY TRACKER: Final response mein key index add kiya
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
def root(): return {"message": "Bilingual Agent with Key Tracker Ready!"}
    name: str = "internet_search"
    description: str = "Use this for real-time factual info from the web."
    def _run(self, query: str) -> str:
        return SerperDevTool().run(search_query=str(query))

search_tool = MySearchTool()

# --- AUTO-SCALING GROQ ROTATION (Up to 50 Keys) ---
def get_groq_llm(key_index):
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    valid_keys = [k for k in keys if k]
    if not valid_keys:
        raise ValueError("Render par koi bhi Groq API Key nahi milti!")
    
    selected_key = valid_keys[key_index % len(valid_keys)]
    # 🔥 TOKEN SAVER: Llama 3.1 8B Instant (Halka, fast aur kam tokens khane wala model)
    return LLM(model="groq/llama-3.1-8b-instant", api_key=selected_key)

def get_total_valid_keys():
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    return len([k for k in keys if k])

class UserRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    # 🔥 TOKEN SAVER: Memory limit wapas 5 kar di hai taaki tokens drain na hon
    past_messages = db.query(ChatMessage).order_by(ChatMessage.id.desc()).limit(5).all()
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
                    f"\n--- Chat History (Last 5 Messages) ---\n{history_str}\n-------------------"
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
def root(): return {"message": "Bilingual Auto-Scaling Pro Agent (Token Saver) is Ready!"}
