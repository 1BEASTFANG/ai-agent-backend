import os
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
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

# --- AI AGENT SETUP (NO HARDCODED KEYS) ---
# Note: Keys will be picked from the environment automatically by CrewAI tools
class MySearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Use this only for factual info."
    def _run(self, query: str) -> str:
        return SerperDevTool().run(search_query=str(query))

search_tool = MySearchTool()
my_llm = LLM(model="gemini/gemini-1.5-flash", temperature=0.1, api_version="v1")

class UserRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    past_messages = db.query(ChatMessage).order_by(ChatMessage.id.desc()).limit(5).all()
    history_str = ""
    for msg in reversed(past_messages):
        history_str += f"User: {msg.user_query}\nAgent: {msg.ai_response}\n"

    smart_agent = Agent(
        role='Dost Assistant',
        goal='Sahi aur friendly jawab dena.',
        backstory=f"Aap ek assistant hain jo memory use karte hain. History: {history_str}",
        tools=[search_tool],
        llm=my_llm
    )
    
    task = Task(
        description=f"User ka naya sawal: {request.question}\nHindi mein jawab dein.",
        expected_output="Direct Hindi response.",
        agent=smart_agent
    )
    
    crew = Crew(agents=[smart_agent], tasks=[task])
    answer = str(crew.kickoff())

    new_entry = ChatMessage(user_query=request.question, ai_response=answer)
    db.add(new_entry)
    db.commit()

    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "DB-Powered Agent is Ready!"}
