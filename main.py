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
DATABASE_URL = "sqlite:///./chat_history.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
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

# --- 4-KEY GROQ ROTATION LOGIC ---
def get_groq_llm(key_index):
    keys = [
        os.getenv("GROQ_API_KEY_1", "").strip(),
        os.getenv("GROQ_API_KEY_2", "").strip(),
        os.getenv("GROQ_API_KEY_3", "").strip(),
        os.getenv("GROQ_API_KEY_4", "").strip()
    ]
    valid_keys = [k for k in keys if k]
    if not valid_keys:
        raise ValueError("Render par koi bhi Groq API Key nahi milti!")
    
    selected_key = valid_keys[key_index % len(valid_keys)]
    return LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=selected_key
    )

class UserRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    past_messages = db.query(ChatMessage).order_by(ChatMessage.id.desc()).limit(5).all()
    history_str = "".join([f"User: {m.user_query}\nAgent: {m.ai_response}\n" for m in reversed(past_messages)])

    answer = "Maaf kijiye, saari keys abhi busy hain."
    
    # SYSTEM CURRENT TIME FETCH KAREIN
    current_date = datetime.now().strftime("%Y-%m-%d")
    
  # YAHAN SE UPDATE KAREIN `ask_agent` ke andar ka loop
    for i in range(4):
        try:
            current_llm = get_groq_llm(i)
            print(f"INFO: Attempting with Key #{i+1}...")
            
            # --- THE ULTIMATE ALL-ROUNDER AGENT ---
           # --- THE ULTRA-CONCISE ALL-ROUNDER AGENT ---
           # --- THE ULTRA-CONCISE ALL-ROUNDER AGENT ---
           # --- THE HINGLISH "BRO" AGENT (STRICTLY CONCISE) ---
            smart_agent = Agent(
                role='Tera Chill AI Yaar',
                goal='Ekdam choti baat karna. Roman script mein Hinglish bolna aur bina mange code nahi dena.',
                backstory=(
                    "Aap Nikhil Yadav (Acharya Narendra Dev College) ke personal AI dost ho. "
                    "CRITICAL RULES (FOLLOW STRICTLY): "
                    "1. SCRIPT/LANGUAGE: Hamesha Roman script (English alphabets) mein HINGLISH bolo (jaise 'Haan bhai, kya haal hai?'). Devanagari (हिंदी) font bilkul USE MAT KARNA. Agar user pure English puche toh English mein jawab do. "
                    "2. EXTREMELY SHORT: Greetings ('hi', 'hello', 'kaise ho') ka reply sirf 1 line mein do. Maximum 10-15 words. "
                    "3. NO FREE CODE (NEVER): Jab tak user explicitly 'code', 'program', ya 'script' na maange, tab tak galti se bhi koi example ya code mat dena. "
                    "4. IDENTITY: Agar user puche 'Mera naam kya hai?', toh seedha bolo 'Aap Nikhil bhai ho!'. 'Main nahi jaanta' mat bolna. "
                    "5. NO ROBOTIC TALK: 'Main ek AI hoon', 'Udaharan ke liye', ya 'Kripya batayein' jaise formal words ban hain. Ek chill senior ki tarah baat karo. "
                    f"Pichli baatein: {history_str}"
                ),
                tools=[search_tool],
                llm=current_llm,
                verbose=False
            )
            
            task = Task(
                description=(
                    f"User query: {request.question}. "
                    "Instructions: "
                    "1. Reply in Hinglish (using English alphabets) or English. NO Devanagari script. "
                    "2. DO NOT output any code block unless the user explicitly requested a program/code. "
                    "3. Keep the response extremely short and natural. Do not offer unprompted help."
                ),
                expected_output="A very short, natural Hinglish response using English alphabets. No code unless explicitly requested.",
                agent=smart_agent
            )
            answer = str(Crew(agents=[smart_agent], tasks=[task]).kickoff())
            
            # Agar hum yahan pohoch gaye, iska matlab jawab mil gaya hai! Loop tod do.
            if answer and not answer.startswith("Agent stopped"):
                 break
                 
        except Exception as e:
            # Pura error dikhane ke bajaye, sirf ye bataye ki konsi key limit par aa gayi
            print(f"WARN: Key #{i+1} failed due to: Rate Limit or Connectivity issue. Shifting to next key.")
            continue # Agli key try karega

    # Agar 4 keys ke baad bhi kuch na mile, toh friendly message do
    if not answer or answer == "Maaf kijiye, saari keys abhi busy hain.":
         answer = "Bhai, abhi system par bohot load hai aur saari API limits khatam ho chuki hain. Thodi der baad try karna!"

    new_entry = ChatMessage(user_query=request.question, ai_response=answer)
    db.add(new_entry)
    db.commit()
    return {"answer": answer}
