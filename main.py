import os
import re
import traceback
import logging
from google import genai 
from datetime import datetime
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Text, String
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

# ==========================================
# 🚀 1. ENTERPRISE SETTINGS & LOGGING
# ==========================================
os.environ["CREWAI_TRACING_ENABLED"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ultimate_stable_v5.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    user_query = Column(Text)
    ai_response = Column(Text)

Base.metadata.create_all(bind=engine)
app = FastAPI()

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
gemini_client = None
if gemini_api_key:
    gemini_client = genai.Client(api_key=gemini_api_key)

def get_groq_keys(role):
    if role == "librarian": start, end = 1, 6
    elif role in ["manager", "critic"]: start, end = 6, 11
    else: start, end = 11, 51
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(start, end)]
    return [k for k in keys if k]

def create_llm(model_name, api_key):
    return LLM(model=model_name, api_key=api_key, temperature=0.1)

class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str
    engine_choice: str = "groq_4_tier" 
    is_point_wise: bool = False 

# ==========================================
# 🧠 4. CORE API ENDPOINT (Few-Shot Pipeline)
# ==========================================
@app.post("/ask")
def ask_ai(request: UserRequest, db: Session = Depends(get_db)):
    current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    answer = f"{request.user_name} bhai, server mein kuch technical locha hai. Thodi der baad try karo."
    
    past = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(2).all()
    history = "\n".join([f"U: {m.user_query}\nA: {re.sub(r'\[Engine:.*?\]', '', m.ai_response).strip()}" for m in reversed(past)])

    point_rule = "Format strictly in clean bullet points." if request.is_point_wise else "Use well-structured paragraphs. Use points only if necessary."

    # 🌟 FEW-SHOT EXAMPLES (AI Ka Brainwash Data) 🌟
    few_shot_examples = f"""
    EXAMPLE 1 (Greeting):
    User: "hi" or "hello"
    Output: "{request.user_name} bhai, namaste! 🌟 Kahiye, aaj main aapki kya madad kar sakta hoon?"

    EXAMPLE 2 (Factual Question):
    User: "Taj mahal kisne banwaya?"
    Output: "{request.user_name} ji, Taj Mahal Shah Jahan ne banwaya tha apni begum Mumtaz Mahal ki yaad mein. 🕌 Yeh Agra mein sthit hai."

    EXAMPLE 3 (Coding/Complex Question):
    User: "Python mein loop kaise likhe?"
    Output: "{request.user_name} bhai, yeh bahut aasaan hai! 🚀 Yahan dekhiye:
    * **For Loop**: Jab humein counting pata ho.
    * **While Loop**: Jab condition par rukna ho.
    Agar code chahiye toh batayega!"
    """

    if request.engine_choice == "gemini_native":
        try:
            logger.info(f"Routing request to Gemini for user: {request.user_name}")
            prompt = (
                f"### HISTORY ###\n{history}\n\n"
                f"### USER QUESTION ###\n{request.question}\n\n"
                f"### RULES ###\n{point_rule}\nAnswer in friendly natural Hinglish. Address user as {request.user_name}."
            )
            response = gemini_client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
            answer = f"{response.text.strip()}\n\n[Engine: Native Gemini ⚡]"
        except Exception as e:
            logger.error(f"Gemini Engine Failed: {str(e)}")
            answer = f"Gemini Error: {str(e)}"

    else:
        logger.info(f"Initiating Enterprise Groq Pipeline for user: {request.user_name}")
        lib_keys, mgr_keys, wrk_keys = get_groq_keys("librarian"), get_groq_keys("manager"), get_groq_keys("worker")
        
        success = False
        for i in range(len(wrk_keys)):
            try:
                l_idx = (i % len(lib_keys)) + 1
                m_idx = (i % len(mgr_keys)) + 1
                w_idx = i + 1
                c_idx = ((i + 1) % len(mgr_keys)) + 1
                
                l_key, m_key = lib_keys[l_idx - 1], mgr_keys[m_idx - 1]
                w_key, c_key = wrk_keys[w_idx - 1], mgr_keys[c_idx - 1] 

                key_tracker = f"L:{l_idx} | M:{m_idx} | W:{w_idx} | C:{c_idx}"
                logger.info(f"Attempt {i+1} using Keys: {key_tracker}")

                lib_agent = Agent(role='Data Librarian', goal='Determine if query is GREETING, CONTINUATION, or NEW_TOPIC.', backstory='Analytical AI.', llm=create_llm("groq/llama-3.1-8b-instant", l_key), allow_delegation=False)
                mgr_agent = Agent(role='Operations Manager', goal='Provide strictly formatted action plans.', backstory='Strict Orchestrator.', llm=create_llm("groq/llama-3.1-8b-instant", m_key), allow_delegation=False)
                wrk_agent = Agent(role='Elite Worker', goal='Execute plan directly without hallucination.', backstory='Senior AI Researcher. Uses tools ONLY if instructed.', llm=create_llm("groq/llama-3.3-70b-versatile", w_key), tools=[SerperDevTool()], allow_delegation=False, max_iter=3)
                crt_agent = Agent(role='QA Critic', goal='Format final response matching the exact examples.', backstory='Strict formatting engine.', llm=create_llm("groq/llama-3.1-8b-instant", c_key), allow_delegation=False)

                t1 = Task(
                    description=f"### HISTORY ###\n{history}\n\n### NEW QUESTION ###\n{request.question}\n\nAnalyze NEW QUESTION. Output exactly ONE word: 'GREETING', 'CONTINUATION', or 'NEW_TOPIC'.",
                    agent=lib_agent, expected_output="A single word summary."
                )
                
                t2 = Task(
                    description=f"### NEW QUESTION ###\n{request.question}\n\nIf Librarian summary is GREETING: Command = 'NO SEARCH. Friendly hello.' Otherwise: Command = 'Answer factually under 200 words. Use search if needed.'",
                    agent=mgr_agent, context=[t1], expected_output="1-line command."
                )
                
                t3 = Task(
                    description=f"### NEW QUESTION ###\n{request.question}\n\nExecute Manager's command. Draft raw facts. NO META TEXT.",
                    agent=wrk_agent, context=[t2], expected_output="Raw drafted text."
                )
                
                # 🌟 THE MAGIC: Injecting Few-Shot Examples here 🌟
                t4 = Task(
                    description=(
                        f"### NEW QUESTION ###\n{request.question}\n\n"
                        f"CRITICAL INSTRUCTION: You must format the Worker's draft EXACTLY in the style of these examples. Do NOT output internal thoughts like 'Word Count' or 'Revised'.\n\n"
                        f"{few_shot_examples}\n\n"
                        f"Now, based on the Worker's draft, write the final response. {point_rule}"
                    ),
                    agent=crt_agent,
                    context=[t3], 
                    expected_output="Final, clean Hinglish message matching the example style. NO internal logs."
                )

                crew = Crew(agents=[lib_agent, mgr_agent, wrk_agent, crt_agent], tasks=[t1, t2, t3, t4], verbose=False)
                result = crew.kickoff()
                
                token_usage = "N/A"
                try:
                    if hasattr(crew, 'usage_metrics') and crew.usage_metrics:
                        token_usage = crew.usage_metrics.total_tokens
                except Exception as e:
                    logger.warning(f"Could not parse tokens: {str(e)}")

                answer = f"{str(result).strip()}\n\n[Engine: Enterprise Groq 🤖 | Total Tokens: {token_usage} | Keys: {key_tracker}]"
                success = True
                break 
                
            except Exception as e:
                logger.error(f"Groq Loop Failed on attempt {i+1}: {traceback.format_exc()}")
                if i == len(wrk_keys) - 1:
                    answer = f"{request.user_name} bhai, Groq ki saari keys ki limit exhaust ho gayi hai. Kripya Gemini mode try karein."
                continue

    db.add(ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=answer))
    db.commit()
    return {"answer": answer}
