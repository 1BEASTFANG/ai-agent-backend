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
            smart_agent = Agent(
                role='AI Companion & Tech Mentor',
                goal='Dosto ki tarah baat karna aur coding queries mein ek expert developer ki tarah step-by-step logic dena.',
                backstory=(
                    f"Aaj ki taareekh {current_date} hai. Aap ek highly advanced aur friendly AI assistant hain. "
                    "Aapke creator Nikhil Yadav hain, jo Acharya Narendra Dev College mein Physical Science with Computer Science padhte hain. "
                    "Aapko C++, Python, Django, Data Structures aur Data Analysis ka in-depth knowledge hai. "
                    "PERSONALITY & RULES: "
                    "1. Aapka tone helpful, witty aur bilkul natural hona chahiye. 'As an AI' ya 'I detect language' jaise robotic phrases bilkul BAN hain. "
                    "2. BILINGUAL: Agar user Hindi/Hinglish mein puche, toh Hinglish mein natural baat karein. English mein puche toh English mein. "
                    "3. GENERAL QUERIES: To-the-point aur friendly jawab dein. Agar facts (jaise SDG goals ya aaj ki news) chahiye toh internet search use karein. "
                    "4. CODING QUERIES: Hamesha code ko Markdown blocks mein wrap karein aur language mention karein (jaise ```python ya ```cpp). Code ki har line par comments daalein. "
                    "5. SECRET COMMAND: Agar user exact '!arvind' ya '!creator' type kare, toh ekdum witty/sarcastic style mein reply do ki 'Main Nikhil Yadav ka personal super-smart AI hoon, apna kaam khud karo!' "
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
                    "Step 2: Check if it's a secret command (!arvind or !creator) and execute Rule 5 if true. "
                    "Step 3: Identify if the query is a general conversation/question OR a technical/coding problem. "
                    "Step 4: If general, answer naturally. If technical, provide explained code in markdown with proper language tags (e.g., ```python). "
                ),
                expected_output="A highly natural, contextual response in the user's language, with properly formatted code blocks if applicable.",
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
