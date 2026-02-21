import os
import re  # 🚀 Regex cleaning ke liye zaroori hai
from datetime import datetime
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Text, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_history_v3.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, default="default_user")
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

def get_groq_llm(key_index):
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    valid_keys = [k for k in keys if k]
    if not valid_keys: raise ValueError("Render par API Keys missing hain!")
    
    return LLM(
        model="groq/llama-3.1-8b-instant", 
        api_key=valid_keys[key_index % len(valid_keys)],
        base_url="https://api.groq.com/openai/v1",
        temperature=0.3  # 🚀 MAGIC WAND: Isse AI overacting aur fake words banana band kar dega
    )

def get_total_valid_keys():
    keys = [os.getenv(f"GROQ_API_KEY_{i}", "").strip() for i in range(1, 51)]
    return len([k for k in keys if k])

class UserRequest(BaseModel):
    session_id: str
    user_name: str
    question: str

# --- MASSIVE ML ROUTER DATASET (Ultra-High Accuracy) ---
TRAIN_DATA = [
    # 💻 CODING, DEV-OPS, & SOFTWARE ENGINEERING
    ("python c++ java javascript typescript go rust ruby php swift kotlin dart code compile syntax error run build output terminal", "coding"),
    ("dsa data structures algorithms doubly linked list array stack queue tree graph recursion dynamic programming overflow logic music player project node pointer", "coding"),
    ("ubuntu linux operating system terminal bash shell command line sudo apt get install chmod chown grep awk ssh ftp network", "coding"),
    ("django python framework web development virtual environment setup local server runserver wsgi asgi settings urls views models forms templates", "coding"),
    ("gui development dear imgui framework c++ graphical user interface rendering event loop window creation input handling", "coding"),
    ("opengl vulkan graphics api qt framework development rendering pipeline shaders textures buffers vertices matrix transformations", "coding"),
    ("html css tailwind bootstrap frontend web design flexbox grid responsive media queries animation transition dom manipulation", "coding"),
    ("react js next js angular vue frontend component state management props hook redux context api router virtual dom jsx", "coding"),
    ("node js express backend server api endpoint routing middleware rest graphql websocket cors json web token jwt authentication", "coding"),
    ("sql mysql postgresql mongodb nosql database query select join inner left right drop create update delete crud indexing", "coding"),
    ("git github gitlab bitbucket version control push pull commit merge conflict branch rebase checkout stash clone remote origin", "coding"),
    ("docker container kubernetes pod deployment yaml aws azure gcp cloud hosting lambda ec2 s3 bucket serverless architecture", "coding"),
    ("bug debug traceback fix stack overflow copy paste error exception handling try catch finally raise assert testing unittest", "coding"),
    ("object oriented programming oops class object inheritance polymorphism abstraction encapsulation method overriding overloading", "coding"),
    ("segmentation fault null pointer exception memory leak garbage collection memory allocation malloc free pointer reference", "coding"),
    ("visual studio code vscode shortcut extensions pycharm intellij ide setup editor formatter linter debugging tools", "coding"),
    ("api key environment variable dotenv security hide credentials authorization oauth cors helmet rate limiting web scraping", "coding"),
    ("npm yarn pip install package dependency version control requirements txt package json node modules virtualenv venv activate", "coding"),
    ("regex regular expression tags clean re sub html string manipulation find replace split join strip uppercase lowercase", "coding"),
    ("crewai agent task llm backstory prompt engineering ai integration framework memory tools callback execution workflow", "coding"),
    ("sqlite database table schema update sqlalchemy orm session commit rollback query filter base declarative migration alembic", "coding"),
    ("json parsing output dictionary key value error serialization formatting dump load stringify parse object mapping", "coding"),
    ("indentation error unindent does not match block scope loop if else switch case break continue logic flow syntax warning", "coding"),
    ("import os datetime sys module not found external library path error pip wheel build resolution dependencies package missing", "coding"),

    # 🚀 DEPLOYMENT, SERVER INFRASTRUCTURE & ERRORS
    ("deployment fail render platform host server port bind issue 0.0.0.0 web service background worker sleep awake ping", "deployment"),
    ("uvicorn fastapi server start stop restart application startup complete lifecycle event asynchronous event loop uvloop", "deployment"),
    ("litellm crash apscheduler missing dependency module not found error fallback proxy gateway proxy server routing", "deployment"),
    ("groq api key organization restricted error rate limit exceeded 429 status bad request 400 forbidden 403 server error 500", "deployment"),
    ("model decommissioned llama3 8b instant change update model version string endpoint url base path completion chat api", "deployment"),
    ("build cache clear manual deploy redeploy web service webhook trigger git push pipeline ci cd github actions actions yaml", "deployment"),
    ("fastapi sso python multipart library install dependency missing wheel build compiler pydantic validation typing schema", "deployment"),
    ("gunicorn uvicorn worker thread process concurrency web scaling traffic load balancer reverse proxy nginx apache timeout", "deployment"),

    # 📊 DATA SCIENCE, AI, ML & ANALYTICS
    ("pandas dataframe read csv data manipulation analysis data science plot chart series loc iloc groupby merge concat join", "data_science"),
    ("calldata csv dataset cleaning preprocessing null values drop fillna dropna sort values filtering aggregation sum mean", "data_science"),
    ("movies dataset grouping aggregation released per year visualization trend line histogram distribution exploratory eda", "data_science"),
    ("diamonds dataset seaborn bubble plot facet grid color cut clarity price carat scatterplot matrix correlation heatmap", "data_science"),
    ("matplotlib pyplot figure size label x y axis title scatter line bar pie plot subplot grid legend styling colors markers", "data_science"),
    ("machine learning pipeline tfidf vectorizer naive bayes multinomialnb text classification model training evaluation fit predict", "data_science"),
    ("numpy array reshape dimension math operations broadcast matrix dot product vector calculus algebra random seed shape", "data_science"),
    ("scikit learn accuracy score confusion matrix f1 precision recall evaluation metrics cross validation grid search hyperparameter", "data_science"),
    ("deep learning neural network tensorflow keras pytorch layer activation function relu sigmoid softmax loss optimizer adam sgd", "data_science"),
    ("ai llm prompt tokens rate limit context window embedding vector database chroma pinecone faiss text generation rag semantic", "data_science"),
    ("data scaling normalization feature engineering min max standard scaler encoding categorical one hot label encoder text processing", "data_science"),
    ("time series forecasting regression linear logistic classification k means clustering unsupervised supervised learning pca", "data_science"),
    ("jupyter notebook cell run kernel restart google colab environment setup markdown code block output clear magic commands", "data_science"),

    # 🎓 COLLEGE, PRESENTATIONS & STUDENT LIFE
    ("acharya narendra dev college andc assignment submission datesheet syllabus physical science computer science department", "college"),
    ("india's journey on the 17 sustainable development goals presentation ppt pdf slideshow format human touch ai generated detail", "college"),
    ("front page title slide nikhil yadav arvind kumar naam dalo group members college name logo background theme design", "college"),
    ("slide number 9 pe ek missing image photo reference add kardo delete pages bibliography citation source reference formatting", "college"),
    ("har topic ke liye scheme target challenge status add karo bullet points formatting clear concise description points", "college"),
    ("practical file lab manual viva questions external examiner internal marks signature checking deadline extension request", "college"),
    ("professor attendance proxy lagwa de short hai medical certificate leave application principal hod faculty staff room", "college"),
    ("delhi university du admission cut off merit list cuet entrance exam result counseling seat allocation migration certificate", "college"),
    ("hostel pg room rent near college campus accommodation flatmate search broker security deposit mess food laundry rules", "college"),
    ("semester marks grading cgpa sgpa topper padhai tips study material notes pyq previous year question paper solutions books", "college"),
    ("college fest society farewell freshers party cultural event competition dance drama coding hackathon debate quiz participation", "college"),
    ("scholarship form last date fee payment online portal registration receipt download acknowledgement admin office query", "college"),
    ("class project ideas final year submission thesis synopsis report writing internship certificate project guide approval", "college"),

    # 📍 LOCATION, TRAVEL, WEATHER & NAVIGATION
    ("delhi kahan hai location india map gps coordinates state capital geography area population language culture history", "location"),
    ("delhi ka weather mausam kaisa hai sardi garmi pollution aqi winter summer rainfall humidity forecast today tomorrow", "location"),
    ("cp connaught place yahan se kitna door hai distance kilometers miles drive time navigation traffic routing alternate path", "location"),
    ("navigation rasta batao google maps direction left right u turn straight destination origin current location compass", "location"),
    ("delhi ke bare main aur batao famous jagah tourist attractions historical places red fort india gate qutub minar lotus temple", "location"),
    ("nearest metro station hospital petrol pump atm dhundh do local search nearby pharmacy clinic grocery store mall market", "location"),
    ("traffic jam kaisa hai time kitna lagega pahunchne me estimated eta delay road block accident diversion highway toll", "location"),
    ("mumbai pune bangalore chennai kolkata flight ticket train booking irctc pnr status waiting list tatkal seat availability", "location"),
    ("hotel stay oyo cheap accommodation near me resort booking holiday package travel agency room tariff check in out timing", "location"),
    ("ola uber cab fare airport terminal 3 railway station route pricing booking ride share auto rickshaw public transport bus", "location"),

    # 🧮 MATH, LOGIC, FINANCE & CALCULATION
    ("2+2*5-3 kitna hoga solution batana calculate solve equation sum difference arithmetic expression evaluate operand", "math"),
    ("bodmas rule brackets order division multiplication addition subtraction priority precedence calculation steps logic", "math"),
    ("algebra matrix determinant inverse transpose scalar vector mathematics polynomial variable constant coefficient", "math"),
    ("calculus integration differentiation limits derivative formula theorem proof sequence series arithmetic geometric progression", "math"),
    ("trigonometry sin cos tan cot sec cosec angle theta radian degree find right angle triangle hypotenuse base perpendicular", "math"),
    ("geometry square circle triangle rectangle area perimeter volume radius diameter circumference polygon surface area 3d 2d", "math"),
    ("percentage nikal do discount profit loss simple compound interest rate calculation principal amount emi loan mortgage", "math"),
    ("probability statistics mean median mode standard deviation variance distribution combinations permutations factorial", "math"),
    ("fractions decimals ratio proportion calculation mixed number simplify cross multiply numerator denominator math trick", "math"),
    ("speed distance time formula pipe cistern work labor efficiency problem solving trains boats stream reasoning aptitude", "math"),
    ("square root cube root power exponent logarithmic calculation base value scientific calculator memory clear function", "math"),
    ("linear quadratic equation roots math problem factoring polynomial expression prime composite number system lcm hcf", "math"),
    ("gst tax calculation inflation gdp income tax return mutual fund sip roi return on investment stock market chart analysis", "math"),

    # 📰 NEWS, CURRENT AFFAIRS & GLOBAL EVENTS
    ("aaj ki latest news kya hai duniya ki khabar batao breaking headline newspaper digital media live update coverage broadcast", "news"),
    ("current affairs update samachar pm modi politics election parliament bill lok sabha rajya sabha opposition ruling party", "news"),
    ("stock market sensex nifty share price crash boom investment portfolio dividend ipo trading finance economy recession", "news"),
    ("technology tech news latest mobile gadget launch specification review tech updates apple samsung google microsoft ai tools", "news"),
    ("sports news cricket match score virat kohli rohit sharma football fifa olympics tennis badminton athletics gold medal", "news"),
    ("government policy scheme yojana supreme court high court law judgement legal constitution rights amendment act police", "news"),
    ("space isro nasa mission chandrayaan satellite launch global warming climate change environment earthquake disaster tsunami", "news"),
    ("crypto bitcoin ethereum price update cryptocurrency blockchain web3 nft mining exchange wallet security hacking cyber", "news"),
    ("international war conflict treaty summit global affairs un who wto brics g20 foreign policy relations borders military", "news"),

    # 🩺 HEALTH, FITNESS & LIFESTYLE
    ("health diet plan weight loss gain gym workout exercise fitness routine bodybuilding cardio yoga pilates zumba marathon", "health"),
    ("fever headache cold cough sardi khasi medicine symptoms treatment cure doctor clinic physician appointment prescription", "health"),
    ("hospital checkup vitamins nutrition protein carbohydrates fat balanced diet vegan organic keto intermittent fasting calories", "health"),
    ("yoga meditation mental health stress relief tips anxiety depression wellness therapy psychologist counseling mindfulness", "health"),
    ("pet dard stomach ache home remedy ilaaj ayurveda natural healing therapy digestion acidity hydration water intake sleep", "health"),

    # 🎬 ENTERTAINMENT, MOVIES & POP CULTURE
    ("movie review bollywood hollywood new release trailer teaser cast director production box office collection hit flop superstar", "entertainment"),
    ("netflix amazon prime hotstar web series recommendation binge watch episode season finale pilot streaming platform ott", "entertainment"),
    ("music song play gaana spotify playlist trending viral audio track singer mp3 album concert live show band beats bass guitar", "entertainment"),
    ("cinema ticket booking showtime pvr inox multiplex popcorn screen 3d imax theatre drama comedy action horror thriller scifi", "entertainment"),
    ("celebrity gossip pop culture fashion trend award show oscars grammys memes viral video social media influencer reel tiktok", "entertainment"),

    # 🗣️ GENERAL, CHIT-CHAT, EMOTIONS & EDGE CASES
    ("hi hello hey namaste aur batao kya haal hai kaise ho greetings good day morning evening night welcome back dear friend", "general"),
    ("kya bol rhe ho kuch samajh nahi aaya theek se batao clear talk conversation repeat again clarify meaning definition explain", "general"),
    ("tumhara naam kya hai tum kaun ho kisne banaya AI bot assistant identity creator owner maker developer intelligence code", "general"),
    ("ok haan nahi theek hai achha samajh gaya yes no alright fine got it done perfect nice beautiful awesome great job well done", "general"),
    ("joke sunao funny chutkula poetry shayari comedy humor laugh lol rofl satire sarcasm pun dad joke entertainment smile happy", "general"),
    ("mera naam kya hai main kaun hoon meri info identity verification user detail remember me personal talk friend relationship", "general"),
    ("tu pagal hai kya bewaqoof bot gaali mat de abusive language polite respect rules terms conditions feedback complaint angry", "general"),
    ("kuch nahi bas aise hi bore ho raha hoon timepass casual talk chat discussion random topic feeling lonely sad unmotivated", "general"),
    ("tell me a fun fact random trivia universe space knowledge interesting information historical fact weird bizarre amazing", "general"),
    ("zindagi ka matlab kya hai philosophy motivation inspiration quote deep meaning life struggle success hard work dedication", "general"),
    ("mujhe neend aa rahi hai thak gaya hoon rest karna hai sleeping tired fatigue exhausted lazy day night schedule routine", "general"),
    ("khaana khaya kya kar rahe ho aaj kal personal life question basic talk chat routine habits hobbies interests likes dislikes", "general"),
    ("ek kahani sunao short story sad happy feeling narrative bedtime story tale fantasy fiction adventure mystery suspense drama", "general"),
    ("bhai tu bada smart hai yaar meri help kar do intelligence praise compliment gratitude appreciation thanks grateful helping hand", "general"),
    ("test testing 123 mic check bot active hai server ping hello response online offline status health check diagnostic system", "general"),
    ("alien ufo time travel exist karte hain kya mystery conspiracy theory scifi future parallel universe dimensions multiverse", "general")
]

texts, labels = zip(*TRAIN_DATA)
ml_router = make_pipeline(TfidfVectorizer(), MultinomialNB())
ml_router.fit(texts, labels)

def detect_category(text):
    return ml_router.predict([text.lower()])[0]

def is_similar(current_q, past_q):
    words_current = set(current_q.lower().split())
    words_past = set(past_q.lower().split())
    if not words_current or not words_past: return False
    overlap = len(words_current.intersection(words_past))
    return (overlap / len(words_current)) >= 0.50

@app.post("/ask")
def ask_agent(request: UserRequest, db: Session = Depends(get_db)):
    past_messages = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.id.desc()).limit(4).all()
    current_category = detect_category(request.question)
    history_str = ""
    
    if past_messages:
        last_msg_category = detect_category(past_messages[0].user_query)
        
        # 🚀 SMART MEMORY MANAGER: Topic badalte hi Memory Delete!
        if current_category != last_msg_category and current_category != 'general':
            history_str = "[System Alert: User switched the topic. Forget all previous chat memory and focus ONLY on the new query.]\n"
        else:
            for m in reversed(past_messages):
                clean_ai_resp = re.sub(r'<.*?>', '', m.ai_response).split('\n\n[Key:')[0].strip()
                history_str += f"User: {m.user_query}\nAgent: {clean_ai_resp}\n"

    answer = "Bhai, saari keys busy hain. Thoda wait kar le."
    current_date = datetime.now().strftime("%Y-%m-%d")
    total_keys = get_total_valid_keys()

    for i in range(total_keys if total_keys > 0 else 1):
        try:
            current_llm = get_groq_llm(i)
            
            backstory_text = (
                f"You are a smart, highly accurate, and helpful AI assistant talking to your friend {request.user_name}. "
                "CRITICAL RULES FOR YOUR RESPONSE: "
                "1. LANGUAGE: You MUST reply in natural, everyday 'Hinglish' (Hindi spoken in daily life, written in the English alphabet). "
                "2. TONE: Be friendly, clear, and direct. Do NOT use highly formal pure Hindi words (avoid words like 'uplabdh', 'vishal', 'shasit'). "
                "3. NO FAKE SLANG: Do NOT invent weird words (like 'khanos' or 'nayaan'). Speak simply. "
                "4. MATH & LOGIC: If asked a math calculation, solve it step-by-step using standard English math terms (like 'multiply', 'subtract', 'BODMAS rule'). Do NOT translate math terms into weird Hindi. "
                f"5. ADDRESSING: Politely address the user by their name ({request.user_name}) in a natural way. "
                "6. CLEAN OUTPUT: Never output <function> tags, XML, or internal JSON. "
                f"\n--- Chat History ---\n{history_str}\n-------------------"
            )
            
            smart_agent = Agent(
                role='AI Assistant',
                goal=f'Provide highly accurate and natural Hinglish answers to {request.user_name}.',
                backstory=backstory_text,
                tools=[search_tool],
                llm=current_llm,
                verbose=False
            )
            
            task_desc = f"User ({request.user_name}) asks: {request.question}. Provide a smart, logically correct, and natural Hinglish response."
            task = Task(description=task_desc, expected_output="A clean, logical Hinglish response.", agent=smart_agent)
            
            raw_answer = str(Crew(agents=[smart_agent], tasks=[task]).kickoff())
            
            if raw_answer and not raw_answer.startswith("Agent stopped"):
                clean_answer = re.sub(r'<.*?>', '', raw_answer)
                clean_answer = re.sub(r'function=.*?>', '', clean_answer)
                clean_answer = clean_answer.replace('{ "code":', '').replace('}', '').strip()
                
                approx_tokens = int((len(backstory_text) + len(task_desc) + len(clean_answer)) / 4) 
                answer = f"{clean_answer}\n\n[Key: {i+1} | Est. Tokens: {approx_tokens}]"
                break 

        except Exception as e:
            print(f"DEBUG: Error with Key {i+1}: {str(e)}")
            continue

    new_entry = ChatMessage(session_id=request.session_id, user_query=request.question, ai_response=answer)
    db.add(new_entry)
    db.commit()
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "Multi-Tenant AI is Live!"}
