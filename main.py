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

# 🔥 NAYA: Machine Learning Libraries for Zero-Token Router
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

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

# --- 🚀 LOCAL MACHINE LEARNING ROUTER (Zero Token Cost) ---
# Humne if-else hata kar model ko example sentences se train kiya hai
TRAIN_DATA = [
    ("pandas dataframe read csv plot chart", "data_science"),
    ("calldata diamonds movies dataset analysis", "data_science"),
    ("c++ code compile error django ubuntu", "coding"),
    ("dsa linked list stack queue logic", "coding"),
    ("acharya narendra dev college andc assignment", "college"),
    ("main kha phadta hoon konsa college", "college"), # ML will learn 'kha phadta' = college
    ("aaj ki latest news batao duniya ki khabar", "news"),
    ("current affairs update samachar", "news"),
    ("delhi ka weather kaisa hai location map", "location"),
    ("kahan par hai distance rasta", "location"),
    ("2 plus 2 calculate formula solve", "math"),
    ("algebra matrix math equation", "math"),
    ("hi hello aur batao kya haal hai", "general"),
    ("kya bol rhe ho", "general")
]

# Training the Local Model in 0.01 seconds
texts, labels = zip(*TRAIN_DATA)
ml_router = make_pipeline(TfidfVectorizer(), MultinomialNB())
ml_router.fit(texts, labels)

def detect_category(text):
    # ML Prediction: Sentence ka context samajh kar category dega
    predicted_category = ml_router.predict([text.lower()])[0]
    return predicted_category

# --- 🚨 DYNAMIC 50% OVERLAP DETECTOR ---
def is_similar(current_q, past_q):
    words_current = set(current_q.lower().split())
    words_past = set(past_q.lower().split())
    if not words_current or not words_past: return False
    overlap = len(words_current.intersection(words_past))
    return (overlap / len(words_current)) >= 0.50

@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    past_messages = db.query(ChatMessage).order_by(ChatMessage.id.desc()).limit(4).all()
    
    current_category = detect_category(request.question)
    history_str = ""
    is_forced_override = False
    
    if len(past_messages) >= 2:
        last_q = past_messages[0].user_query
        second_last_q = past_messages[1].user_query
        current_q = request.question
        
        if is_similar(current_q, last_q) and is_similar(current_q, second_last_q):
            is_forced_override = True

    if past_messages:
        last_msg_category = detect_category(past_messages[0].user_query)
        
        if is_forced_override:
            history_str += "[System Alert: User repeating inputs (Frustrated). Ignore formal categories, preserve memory, give direct solution based on 10 Rules.]\n"
            for m in reversed(past_messages):
                clean_response = m.ai_response.split("\n\n[Key:")[0]
                history_str += f"User: {m.user_query}\nAgent: {clean_response}\n"
        else:
            if current_category != 'general' and last_msg_category != 'general' and current_category != last_msg_category:
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
            
            # --- THE "SOBER UP" PROMPT (No more hallucinations) ---
            backstory_text = (
                f"Date: {current_date}. Tu ek smart AI assistant hai. "
                "TUJHE IN RULES KO HAR HAAL MEIN MAANNA HAI: "
                "1. THE USER IS NIKHIL: Jo insaan tujhse abhi chat kar raha hai, wahi Nikhil Yadav hai. Tujhe Nikhil se directly baat karni hai ('Aap' ya 'Tu' keh kar). "
                "2. BACKGROUND INFO: Nikhil 'Acharya Narendra Dev College (ANDC)' ka student hai. Arvind Kumar uska dost hai. "
                "3. HINGLISH COMPREHENSION: Dhyan se padh! User short form use karega. Jaise 'kha' ka matlab 'kahan' (where) hota hai. "
                "4. NO AI LECTURES: Agar user 'hi', 'hello' ya 'kya bol rahe ho' kahe, toh 1 line mein direct jawab de. Apne rules mat suna. "
                "5. NO TOOL LEAKS: Kabhi bhi <internet_search> ya -function output mein mat likh. "
                f"\n--- Chat History ---\n{history_str}\n-------------------"
            )
            
            smart_agent = Agent(
                role='AI Assistant',
                goal='Talk directly to Nikhil, understand Hinglish, and give smart, short answers without leaking internal tags.',
                backstory=backstory_text,
                tools=[search_tool],
                llm=current_llm,
                verbose=False
            )
            
            task_desc = f"User is Nikhil. Query: {request.question}. Give a direct, smart Hinglish answer. DO NOT print tags or rules."
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
    return {"message": "Bilingual AI (Local ML Router Edition) is Live!"}
