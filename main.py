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
    try:
        yield db
    finally:
        db.close()

# --- SEARCH TOOL SETUP ---
class MySearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Use this only for factual info."
    def _run(self, query: str) -> str:
        # Serper key automatically environment se uthayi jayegi
        return SerperDevTool().run(search_query=str(query))

search_tool = MySearchTool()

# --- LLM CONFIGURATIONS (With Safety Trimming) ---
# Primary: OpenRouter
or_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
openrouter_llm = LLM(
    model="openrouter/google/gemini-2.0-flash-exp:free",
    temperature=0.1,
    base_url="https://openrouter.ai/api/v1",
    api_key=or_api_key
)

# Backup: Groq (Render par GROQ_API_KEY set hai)
groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
groq_llm = LLM(
    model="groq/llama-3.1-70b-versatile",
    temperature=0.1,
    api_key=groq_api_key
)

class UserRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    # 1. Past memory load karna
    past_messages = db.query(ChatMessage).order_by(ChatMessage.id.desc()).limit(5).all()
    history_str = ""
    for msg in reversed(past_messages):
        history_str += f"User: {msg.user_query}\nAgent: {msg.ai_response}\n"

    # Helper function to run the agent
    def execute_with_llm(selected_llm):
        smart_agent = Agent(
            role='Dost Assistant',
            goal='Sahi aur friendly jawab dena.',
            backstory=f"Aap ek assistant hain jo memory use karte hain. History: {history_str}",
            tools=[search_tool],
            llm=selected_llm
        )
        task = Task(
            description=f"User ka naya sawal: {request.question}\nHindi mein jawab dein.",
            expected_output="Direct Hindi response.",
            agent=smart_agent
        )
        crew = Crew(agents=[smart_agent], tasks=[task])
        return str(crew.kickoff())

    # --- FALLBACK LOGIC ---
    try:
        print("INFO: Trying OpenRouter...")
        answer = execute_with_llm(openrouter_llm)
    except Exception as e:
        print(f"ERROR: OpenRouter failed ({e}). Switching to Groq...")
        try:
            answer = execute_with_llm(groq_llm)
        except Exception as e2:
            print(f"CRITICAL: Groq also failed ({e2})")
            answer = "Maaf kijiye Nikhil bhai, mere dono servers abhi thoda busy hain. Thodi der mein try karein!"

    # 2. Database mein save karein
    new_entry = ChatMessage(user_query=request.question, ai_response=answer)
    db.add(new_entry)
    db.commit()

    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "Dost Agent is Ready with Fallback Support!"}
