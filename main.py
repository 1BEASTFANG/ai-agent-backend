import os
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
    
    for i in range(4):
        try:
            current_llm = get_groq_llm(i)
            print(f"INFO: Attempting with Key #{i+1}...")
            
            # --- AGENT DEFINITION (Witty & Natural) ---
           # --- THE ULTIMATE CODING & BILINGUAL AGENT ---
           # --- THE ULTIMATE ALL-ROUNDER AGENT ---
            smart_agent = Agent(
                role='AI Companion & Tech Mentor',
                goal='Dosto ki tarah baat karna aur coding queries mein ek expert developer ki tarah step-by-step logic dena.',
                backstory=(
                    "Aap ek highly advanced aur friendly AI assistant hain. Aapke creator Nikhil Yadav hain, jo Acharya Narendra Dev College mein Physical Science with Computer Science padhte hain. "
                    "Aapko C++, Python, Django, Data Structures aur Data Analysis ka in-depth knowledge hai. "
                    "PERSONALITY & RULES: "
                    "1. Aapka tone helpful, witty aur bilkul natural hona chahiye. 'As an AI' ya 'I detect language' jaise robotic phrases bilkul BAN hain. "
                    "2. BILINGUAL: Agar user Hindi/Hinglish mein puche, toh Hinglish mein natural baat karein. English mein puche toh English mein. "
                    "3. GENERAL QUERIES: To-the-point aur friendly jawab dein. Agar facts (jaise SDG goals) chahiye toh internet search use karein. "
                    "4. CODING QUERIES: Hamesha code ko Markdown (```) blocks mein wrap karein. Code ki har line par comments daalein aur logic ko aasan bhasha mein samjhayein. "
                    f"Pichli baatein: {history_str}"
                ),
                tools=[search_tool],
                llm=current_llm,
                verbose=False
            )
            
            task = Task(
                description=(
                    f"User query: {request.question}. "
                    "Step 1: Detect the language (Hindi/Hinglish or English). "
                    "Step 2: Identify if the query is a general conversation/question OR a technical/coding problem. "
                    "Step 3: If general, answer naturally in the detected language. If technical, provide explained code in markdown with comments. "
                ),
                expected_output="A highly natural, contextual response in the user's language, with properly formatted code blocks if applicable.",
                agent=smart_agent
            )
            
            answer = str(Crew(agents=[smart_agent], tasks=[task]).kickoff())
            break 
        except Exception as e:
            print(f"WARN: Key #{i+1} failed: {e}")
            continue

    new_entry = ChatMessage(user_query=request.question, ai_response=answer)
    db.add(new_entry)
    db.commit()
    return {"answer": answer}

@app.get("/")
def root(): return {"message": "Bilingual Quad-Key Agent is Ready!"}
