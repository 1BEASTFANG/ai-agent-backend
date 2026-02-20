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

# --- SEARCH TOOL ---
class MySearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Use this only for factual info."
    def _run(self, query: str) -> str:
        return SerperDevTool().run(search_query=str(query))

search_tool = MySearchTool()

# --- LLM CONFIGURATIONS (Triple Layer) ---

# Layer 1: OpenRouter (Primary)
or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
openrouter_llm = LLM(
    model="openrouter/google/gemini-2.0-flash-exp:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=or_key
)

# Layer 2: Groq (Updated Model for 2026)
groq_key = os.getenv("GROQ_API_KEY", "").strip()
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile", # Updated from decommissioned model
    api_key=groq_key
)

# Layer 3: Direct Google Gemini (The Ultimate Backup)
google_key = os.getenv("GOOGLE_API_KEY", "").strip()
google_llm = LLM(
    model="google/gemini-1.5-flash", 
    api_key=google_key
)

class UserRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    past_messages = db.query(ChatMessage).order_by(ChatMessage.id.desc()).limit(5).all()
    history_str = "".join([f"User: {m.user_query}\nAgent: {m.ai_response}\n" for m in reversed(past_messages)])

    def run_agent(llm_choice, provider_name):
        print(f"INFO: Attempting with {provider_name}...")
        agent = Agent(
            role='Dost Assistant',
            goal='Friendly and factual answers in Hindi.',
            backstory=f"Aap ek digital dost hain. History: {history_str}",
            tools=[search_tool],
            llm=llm_choice
        )
        task = Task(
            description=f"User question: {request.question}. Answer in Hindi.",
            expected_output="Short Hindi response.",
            agent=agent
        )
        return str(Crew(agents=[agent], tasks=[task]).kickoff())

    # --- THE TRIPLE FALLBACK LOGIC ---
    answer = ""
    try:
        answer = run_agent(openrouter_llm, "OpenRouter")
    except Exception:
        try:
            answer = run_agent(groq_llm, "Groq")
        except Exception:
            try:
                answer = run_agent(google_llm, "Google Direct")
            except Exception:
                answer = "Bhai, aaj saare servers thak gaye hain. Thodi der mein try karein!"

    new_entry = ChatMessage(user_query=request.question, ai_response=answer)
    db.add(new_entry)
    db.commit()
    return {"answer": answer}

@app.get("/")
def root(): return {"message": "Triple-Layer AI Agent is Live!"}
