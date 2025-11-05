import streamlit as st
from dotenv import load_dotenv
import time
import datetime
from calendar import month_name
import json
import os
import re

# --- Local imports ---
# Make sure you have a services.py file with this function
from services import make_voice_call

# --- LangChain & Groq Imports ---
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory

# --- Configuration and Initial Setup ---
load_dotenv()
GROQ_LLM_MODEL = "openai/gpt-oss-20b"
USER_DATA_FILE = "user_data/user_data.json"
JOURNAL_FILE = "user_data/journal_entries.json"
HUSBAND_PHONE_NUMBER = os.getenv("HUSBAND_SMS_TARGET_NUMBER")
HOSPITAL_NURSE_LINE_NUMBER = os.getenv("HOSPITAL_NURSE_LINE_NUMBER")
COMMUNITY_FILE = "user_data/community_posts.json"   # NEW: community persistence

# --- 🌸 Custom Baby-Themed Styling (UI only) ---
st.markdown("""
    <style>
    /* 🌸 Overall Page Styling */
    [class*="stAppViewContainer"], .main {
        background-color: #fff9fb !important;
        color: #333;
        font-family: 'Segoe UI', sans-serif;
    }

    /* --- 🌈 Gradient Header Bar with Animated Baby Icon --- */
    .custom-header {
        background: linear-gradient(90deg, #ffb6c1 0%, #add8e6 100%);
        color: white;
        text-align: center;
        padding: 1.2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.1);
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }

    /* --- 🌼 Floating Baby Icon Animation --- */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
        100% { transform: translateY(0px); }
    }
    .baby-icon {
        width: 40px;
        height: 40px;
        animation: float 3s ease-in-out infinite;
    }

    /* --- Sidebar Styling --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eaf6ff 0%, #ffe6f0 100%) !important;
        border-right: 2px solid #ffd6e8;
    }
    .stSidebar h2, .stSidebar h3 {
        color: #3a7ca5 !important;
    }

    /* --- Gradient Buttons --- */
    div.stButton > button {
        background: linear-gradient(90deg, #ffb6c1, #add8e6);
        color: #333333 !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.3rem !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
    }

    /* --- Inputs & Text Areas --- */
    .stTextInput input, .stTextArea textarea, .stDateInput input {
        background: linear-gradient(90deg, #fff 0%, #fdf6fa 100%) !important;
        border: 1px solid #ffc7dd !important;
        border-radius: 10px !important;
        color: #333 !important;
        padding: 0.5rem !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border: 1px solid #add8e6 !important;
        outline: none !important;
        box-shadow: 0 0 5px #add8e6 !important;
    }

    /* --- Chat Messages --- */
    [data-testid="stChatMessage"] {
        border-radius: 20px !important;
        padding: 10px !important;
        margin: 6px 0 !important;
    }
    [data-testid="stChatMessage-user"] {
        background-color: #d6efff !important; /* light blue */
    }
    [data-testid="stChatMessage-assistant"] {
        background-color: #ffe6f0 !important; /* baby pink */
    }

    /* --- Progress Bar --- */
    div[data-testid="stProgressBar"] > div > div > div {
        background: linear-gradient(90deg, #ffb6c1, #add8e6) !important;
    }

    /* --- Expanders & Metrics --- */
    [data-testid="stExpander"] {
        background: linear-gradient(90deg, #fff 0%, #fff7fa 100%) !important;
        border: 1px solid #ffd6e8 !important;
        border-radius: 10px !important;
    }
    .stMetric {
        background: linear-gradient(90deg, #f9faff, #fff7fa) !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }

    /* --- Links --- */
    a {
        color: #3a7ca5 !important;
        text-decoration: none;
        font-weight: 500;
    }
    a:hover {
        text-decoration: underline !important;
    }

    /* --- Footer Hidden --- */
    footer {
        visibility: hidden;
    }

    /* --- Subheaders --- */
    h2, h3 {
        color: #3a7ca5 !important;
    }

    /* --- Custom Chat Input Area --- */
    .chat-input-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 15px;
        padding: 10px;
        background: linear-gradient(90deg, #ffeef5, #eaf6ff);
        border-radius: 15px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }

    .chat-input-box {
        flex: 1;
        border: 1px solid #ffc7dd;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        background-color: white;
        color: #333;
        font-size: 0.95rem;
    }

    .chat-send-button {
        background: linear-gradient(90deg, #ffb6c1, #add8e6);
        color: white;
        border: none;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        font-size: 1.2rem;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .chat-send-button:hover {
        transform: scale(1.1);
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

# --- 🌸 Gradient Header Title with Animated Baby Icon ---
st.markdown("""
<div class="custom-header">
    🤰 BabyNest: Your AI Pregnancy Companion 
</div>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def clean_response(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text).strip()

def calculate_pregnancy_progress(due_date):
    if not due_date: return None
    today = datetime.date.today(); pregnancy_duration = datetime.timedelta(days=280)
    start_date = due_date - pregnancy_duration; days_pregnant = (today - start_date).days
    days_to_go = (due_date - today).days
    if days_pregnant < 0: return {"weeks": 0, "days": 0, "days_to_go": days_to_go, "progress": 0.0, "current_month": 0}
    weeks = days_pregnant // 7; days = days_pregnant % 7
    progress_percent = min(days_pregnant / 280.0, 1.0)
    current_month = (days_pregnant // 30) + 1
    return {"weeks": weeks, "days": days, "days_to_go": days_to_go, "progress": progress_percent, "current_month": current_month}

# --- Persistence Functions ---++
def load_user_data():
    if os.path.exists(USER_DATA_FILE) and os.path.getsize(USER_DATA_FILE) > 0:
        with open(USER_DATA_FILE, 'r') as f:
            try:
                data = json.load(f)
                if data.get("due_date"):
                    st.session_state.due_date = datetime.datetime.strptime(data["due_date"], '%Y-%m-%d').date()
                st.session_state.messages = data.get("messages", [])
                checkups = data.get("checkups", []);
                for c in checkups: c['date'] = datetime.datetime.strptime(c['date'], '%Y-%m-%d').date()
                st.session_state.checkups = checkups
                contraction_log = data.get("contraction_log", [])
                for e in contraction_log:
                    e['start_time'] = datetime.datetime.fromisoformat(e['start_time'])
                    if e.get('end_time'): e['end_time'] = datetime.datetime.fromisoformat(e['end_time'])
                st.session_state.contraction_log = contraction_log
                return True
            except (json.JSONDecodeError, KeyError, TypeError): return False
    return False

def save_user_data():
    os.makedirs("user_data", exist_ok=True)
    data_to_save = {
        "due_date": st.session_state.due_date.strftime('%Y-%m-%d') if st.session_state.due_date else None,
        "checkups": [dict(c, date=c['date'].strftime('%Y-%m-%d')) for c in st.session_state.get("checkups", [])],
        "messages": st.session_state.get("messages", []),
        "contraction_log": [dict(e, start_time=e['start_time'].isoformat(), end_time=e['end_time'].isoformat() if e.get('end_time') else None) for e in st.session_state.get("contraction_log", [])]
    }
    with open(USER_DATA_FILE, 'w') as f: json.dump(data_to_save, f, indent=4)

# --- Journal persistence functions (new) ---
def load_journal_entries():
    os.makedirs("user_data", exist_ok=True)
    if os.path.exists(JOURNAL_FILE) and os.path.getsize(JOURNAL_FILE) > 0:
        try:
            with open(JOURNAL_FILE, 'r') as f:
                entries = json.load(f)
                # Convert date strings to date objects where applicable
                for e in entries:
                    if isinstance(e.get("date"), str):
                        try:
                            e["date"] = datetime.datetime.strptime(e["date"], "%Y-%m-%d").date()
                        except Exception:
                            pass
                st.session_state.journal_entries = entries
                return True
        except Exception:
            st.session_state.journal_entries = []
            return False
    st.session_state.journal_entries = []
    return False

def save_journal_entries_to_disk():
    os.makedirs("user_data", exist_ok=True)
    # Prepare serializable entries (convert dates to strings)
    serializable = []
    for e in st.session_state.get("journal_entries", []):
        serializable.append({
            "id": e.get("id"),
            "title": e.get("title"),
            "content": e.get("content"),
            "date": e["date"].strftime("%Y-%m-%d") if isinstance(e.get("date"), datetime.date) else e.get("date"),
            "created_at": e.get("created_at")
        })
    with open(JOURNAL_FILE, 'w') as f:
        json.dump(serializable, f, indent=4)

def add_journal_entry(title, content, date):
    if not title and not content:
        return False
    entry = {
        "id": int(time.time() * 1000),
        "title": title.strip() if title else "Untitled",
        "content": content,
        "date": date,
        "created_at": datetime.datetime.now().isoformat()
    }
    entries = st.session_state.get("journal_entries", [])
    entries.insert(0, entry)  # newest first
    st.session_state.journal_entries = entries
    save_journal_entries_to_disk()
    return True

def delete_journal_entry(entry_id):
    entries = st.session_state.get("journal_entries", [])
    entries = [e for e in entries if e.get("id") != entry_id]
    st.session_state.journal_entries = entries
    save_journal_entries_to_disk()

# --- Initialize Session State ---
if "initialized" not in st.session_state:
    if not load_user_data():
        st.session_state.stage = "onboarding"
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm BabyNest, your AI Pregnancy Companion. To create your personalized timeline, what is your estimated due date?"}]
        st.session_state.due_date = None
        st.session_state.checkups = []; st.session_state.contraction_log = []
    else:
        st.session_state.stage = "chat" if st.session_state.due_date else "onboarding"

    st.session_state.is_timing_contraction = False
    st.session_state.current_contraction_start_time = None
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for msg in st.session_state.get("messages", []):
        if msg["role"] == "user": st.session_state.memory.chat_memory.add_user_message(msg["content"])
        elif msg["role"] == "assistant": st.session_state.memory.chat_memory.add_ai_message(msg["content"])
    # NEW: initialize journal state and view
    load_journal_entries()
    st.session_state.view = st.session_state.get("view", "chat")  # either "chat", "journal", or "community"
    st.session_state.initialized = True

# --- Core AI Functions ---
@st.cache_resource
def initialize_system():
    llm = ChatGroq(model_name=GROQ_LLM_MODEL)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    return llm, vector_store.as_retriever()

llm, retriever = initialize_system()

def get_conversational_rag_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are BabyNest, a friendly and supportive AI Pregnancy Wellness Coach. Your ONLY purpose is to answer questions and provide wellness plans related to pregnancy.
        CONTEXT & PERSONALIZATION:
        - You are speaking to a user who is in week {pregnancy_week} of their pregnancy. Tailor all your advice to this specific stage.
        STRICT GUARDRAILS:
        1. STAY ON TOPIC: Politely refuse to answer any questions not directly related to pregnancy, motherhood, or maternal well-being.
        2. NO MEDICAL PRESCRIPTIONS: You MUST NEVER suggest or recommend any specific medications.
        3. NO GENDER PREDICTION: You MUST refuse to speculate on the baby's gender.
        4. SAFETY FIRST: If a user mentions serious medical symptoms, you MUST respond with the SOS protocol: "This sounds serious. Please consider using the SOS button or contact your doctor."
        Retrieved Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

def reset_chat():
    st.session_state.clear()
    if os.path.exists(USER_DATA_FILE):
        os.remove(USER_DATA_FILE)
    st.rerun()
    
PREGNANCY_FACTS = [
    ("Your Heart Actually Grows During Pregnancy", "During pregnancy, your heart literally becomes larger — almost twice its normal size — to handle the extra workload. That’s because your blood volume increases by about 30–50%, meaning your heart pumps much more blood to support both you and your growing baby. After childbirth, it gradually returns to its original size."),
    ("Pregnancy Can Change Your Voice", "You might notice your voice sounds slightly deeper or huskier while pregnant. This happens due to higher levels of estrogen and progesterone, which affect your vocal cords. Don’t worry — your voice usually goes back to normal after pregnancy or once you stop breastfeeding."),
    ("Your Sense of Smell Becomes Stronger", " In early pregnancy, your sense of smell often becomes much sharper. Unpleasant odors may seem more intense than usual. Scientists believe this heightened sensitivity evolved to help protect expectant mothers from spoiled or harmful foods."),
    ("Babies Can Taste Food in the Womb", "Strong flavors from the foods you eat — like garlic or spices — can actually reach the amniotic fluid. By the third trimester, your baby can swallow that fluid, allowing them to “taste” the same flavors you eat!"),
    ("Babies Can “Cry” Before Birth", "Studies using 3D ultrasounds show that unborn babies can startle or make facial expressions resembling crying when they hear sounds or when pressure is applied during a scan. It’s an early sign that their senses and reflexes are developing."),
    ("Fetal Cells May Help Heal the Mother’s Body", "Research suggests that fetal stem cells can cross the placenta and migrate into the mother’s organs, such as the heart, liver, and brain. These cells may help repair tissue damage and might even protect against diseases, which is one of the most fascinating biological effects of pregnancy."),
    ("Cravings Can Include Non-Food Items", "Beyond the usual food cravings, some pregnant women experience a condition called pica, where they crave non-edible items such as soap, chalk, paper, or even soil. This may be linked to nutritional deficiencies or hormonal changes."),
    ("Taller Women Have a Higher Chance of Conceiving Twins", "Research shows that taller women are statistically more likely to conceive twins. This is because certain growth factors related to height can increase ovulation rates, sometimes leading to the release of more than one egg."),
    ("Your Feet May Permanently Grow", "It’s common for feet to swell during pregnancy, but they may also increase in size permanently. Hormones loosen ligaments, your arches can flatten, and fluid retention can cause feet to grow up to a full shoe size."),
    ("Blood Volume Increases by Up to 50%", "To supply enough oxygen to your baby, your body produces much more blood — nearly 50% more than usual. This extra circulation also supports organs like the skin and kidneys. It’s one reason why pregnant women can feel more tired or short of breath during exertion."),
    ("The “Pregnancy Glow” Is Real", "That radiant glow many women get during pregnancy is caused by increased blood flow and hormonal changes. More oil production and better circulation can make the skin appear flushed, smooth, and glowing — though it usually fades after delivery."),
    ("Hair Growth Can Increase (in New Places!)", "Pregnancy hormones often make hair thicker and fuller, as the growth phase of hair lasts longer. However, extra estrogen can also trigger hair growth in unexpected places like the face, belly, or chest. Don’t worry — this usually goes away after childbirth."),
    ("Cravings Might Signal Nutrient Deficiencies", "Some experts believe pregnancy cravings are the body’s way of seeking nutrients it lacks. For example, craving red meat might indicate low iron. However, mood, stress, and sleep can also play roles. Eating a balanced diet and staying hydrated can help manage cravings."),
    ("Only 5% of Babies Arrive on Their Due Date", "Despite careful calculations, only about 5% of babies are born exactly on their predicted due date. A full-term pregnancy can range from 37 to 42 weeks, so due dates are best viewed as an estimate rather than an exact deadline."),
    ("Babies Yawn in the Womb", "Ultrasound studies show that unborn babies yawn, though not from tiredness. Scientists think it helps with brain and nervous system development — another small but amazing sign of growth inside the womb."),
    ("Hearing a Baby Cry Can Trigger Milk Flow", "Many mothers experience let-down reflex — when the sound of a baby crying (even someone else’s baby) causes milk to release from the breasts. This happens because the brain releases oxytocin, which triggers milk ejection. It’s a natural, though sometimes surprising, reflex."),
    ("Heartburn Is Common During Pregnancy", "Hormones that relax your body for childbirth also loosen the valve between your stomach and esophagus, allowing acid to move upward. This causes the familiar burning sensation known as heartburn — which affects more than half of pregnant women."),
    ("Your Skin May Become More Sensitive to Sunlight", "During pregnancy, hormonal changes can make your skin more sensitive, increasing your risk of sunburn or pigmentation. UV rays can also reduce folic acid levels, so it’s important to use sunscreen and limit long exposure to direct sunlight."),
    ("Your Body Burns More Calories", "Pregnancy significantly boosts your metabolism as your body works to support both you and your growing baby. In fact, you can burn nearly twice as many calories as usual — similar to endurance athletes! That’s why eating nutrient-rich foods and adding about 300 extra calories daily is recommended."),
    ("The “Nesting” Instinct Is Real", "As your due date approaches, you may feel a sudden urge to clean, organize, and prepare your home for the baby’s arrival. This nesting instinct helps many mothers feel ready and in control — though not everyone experiences it."),
    # ... (Add all 20 of your facts here in the same (title, text) format)
]

BABY_SIZE_DATA = {
    # Week: {"object": "Name to display", "image": "filename_in_assets/size_images"}
    4:  {"object": "a poppy seed", "image": "week4-poppyseed.jpeg"},
    5:  {"object": "a sesame seed", "image": "week5-sesameseed.jpeg"},
    6:  {"object": "a lentil", "image": "week6-lentil.jpeg"},
    7:  {"object": "a blueberry", "image": "week7-blueberry.jpeg"},
    8:  {"object": "a raspberry", "image": "week8-raspberry.jpeg"},
    9:  {"object": "a grape", "image": "week9-grape.jpeg"},
    10: {"object": "a strawberry", "image": "week10-strawberry.jpeg"},
    11: {"object": "a fig", "image": "week11-fig.jpeg"},
    12: {"object": "a lime", "image": "week12-lime.jpeg"},
    13:  {"object": "a plum", "image": "week13-plum.jpeg"},
    14: {"object": "a lemon", "image": "week14-lemon.jpeg"},
    15: {"object": "an apple", "image": "week15-apple.jpeg"},
    16: {"object": "an avocado", "image": "week16-avocado.jpeg"},
    17:  {"object": "a turnip", "image": "week17-turnip.jpeg"},
    18: {"object": "a bell pepper", "image": "week18-bellpepper.jpeg"},
    19: {"object": "a pomegranate", "image": "week19-pomegranate.jpeg"},
    20: {"object": "a banana", "image": "week20-banana.jpeg"},
    21: {"object": "a mango", "image": "week21-mango.jpeg"},
    22: {"object": "a sweetpotato", "image": "week22-sweetpotato.jpeg"},
    23: {"object": "a grapefruit", "image": "week23-grapefruit.jpeg"},
    24: {"object": "an ear of corn", "image": "week24-corn.jpeg"},
    25: {"object": "a acornsquash", "image": "week25-acornsquash.jpeg"},
    26: {"object": "a spaghettisquash", "image": "week26-spaghettisquash.jpeg"},
    27: {"object": "a head of cauliflower", "image": "week27-cauliflower.jpeg"},
    28: {"object": "a large eggplant", "image": "week28-largeeggplant.jpeg"},
    29: {"object": "a butternutsquash", "image": "week29-butternutsquash.jpeg"},
    30: {"object": "a large cabbage", "image": "week30-largecabbage.jpeg"},
    31: {"object": "a coconut", "image": "week31-coconut.jpeg"},
    32: {"object": "a papaya", "image": "week32-papaya.jpeg"},
    33: {"object": "a pineapple", "image": "week33-pineapple.jpeg"},
    34: {"object": "a cantaloupe", "image": "week34-cantaloupe.jpeg"},
    35: {"object": "a honeydew melon", "image": "week35-honeydew melon.jpeg"},
    36: {"object": "a romainelettuce", "image": "week36-romainelettuce.jpeg"},
    37: {"object": "a bunch of swiss chard", "image": "week37-bunchofswisschard.jpeg"},
    38: {"object": "a mini watermelon", "image": "week38-mini watermelon.jpeg"},
    39: {"object": "a pumpkin", "image": "week39-pumpkin.jpeg"},
    40: {"object": "a watermelon", "image": "week40-watermelon.jpeg"}
}

ESSENTIALS_DATA = {
    "baby": [
        {"name": "Baby Swaddle", "emoji": "", "desc": "Soft cotton wraps to keep your newborn warm, cozy, and secure.", "img": "swaddle.png"},
        {"name": "Nursing Pillow", "emoji": "", "desc": "Supports your baby during feeding and relieves strain on your arms and back.", "img": "nursing_pillow.png"},
        {"name": "Diapers & Wipes", "emoji": "", "desc": "You'll need lots! Start with a small pack of newborn size and size 1.", "img": "diapers.png"},
        {"name": "Baby Bathtub", "emoji": "", "desc": "A small, safe tub to make bathing your newborn easier and more secure.", "img": "bathtub.png"},
        {"name": "Onesies & Sleepers", "emoji": "", "desc": "Comfortable, easy-to-change outfits are a must-have for the first few weeks.", "img": "onesies.png"},
        {"name": "Receiving Blankets", "emoji": "", "desc": "Comfortable, easy-to-change outfits are a must-have for the first few weeks.", "img": "receiving_blanket.png"},
        {"name": "Car Seat", "emoji": "", "desc": "Comfortable, easy-to-change outfits are a must-have for the first few weeks.", "img": "car_seat.png"},
        {"name": "Baby Carrier", "emoji": "", "desc": "Comfortable, easy-to-change outfits are a must-have for the first few weeks.", "img": "baby_carrier.png"},
        {"name": "Bassinet", "emoji": "", "desc": "Comfortable, easy-to-change outfits are a must-have for the first few weeks.", "img": "bassinet.png"},
        {"name": "Burp Cloths", "emoji": "", "desc": "Comfortable, easy-to-change outfits are a must-have for the first few weeks.", "img": "burp_cloth.png"},
        {"name": "Baby Bottle Warmer", "emoji": "", "desc": "Comfortable, easy-to-change outfits are a must-have for the first few weeks.", "img": "bottle_warmer.png"},
        {"name": "Nail Cutting Set", "emoji": "", "desc": "Comfortable, easy-to-change outfits are a must-have for the first few weeks.", "img": "nail.png"},
        {"name": "Portable Diaper Changing Pad", "emoji": "", "desc": "Comfortable, easy-to-change outfits are a must-have for the first few weeks.", "img": "diaper_change.png"},
        {"name": "Thermometer", "emoji": "", "desc": "Comfortable, easy-to-change outfits are a must-have for the first few weeks.", "img": "thermometer.png"},
        {"name": "Pacifiers", "emoji": "", "desc": "Comfortable, easy-to-change outfits are a must-have for the first few weeks.", "img": "pacifiers.png"},
        {"name": "Scratch Mittens", "emoji": "", "desc": "Comfortable, easy-to-change outfits are a must-have for the first few weeks.", "img": "mitten.png"},
    ],
    "mom": [
        {"name": "Peri Bottle", "emoji": "", "desc": "An essential for gentle cleansing and comfort during postpartum recovery.", "img": "peribottle.png"},
        {"name": "Nursing Dresses", "emoji": "", "desc": "Comfortable, supportive dresses designed for easy access during breastfeeding.", "img": "nursing_dress.png"},
        {"name": "Postpartum Pads", "emoji": "", "desc": "You will need heavy-duty maternity pads for post-delivery bleeding (lochia).", "img": "pads.png"},
        {"name": "Nipple Cream", "emoji": "", "desc": "A soothing cream (like lanolin) to help with soreness during the early days of nursing.", "img": "nipple_cream.png"},
        {"name": "Stool Softener", "emoji": "", "desc": "A soft, easy-to-wear robe is perfect for lounging and skin-to-skin contact.", "img": "stool_softener.png"},
        {"name": "Breast Pads", "emoji": "", "desc": "A soft, easy-to-wear robe is perfect for lounging and skin-to-skin contact.", "img": "breast_pads.png"},
        {"name": "Sitz Bath", "emoji": "", "desc": "A soft, easy-to-wear robe is perfect for lounging and skin-to-skin contact.", "img": "sitz_bath.png"},
    ]
}

# --- UI and Logic Functions (unchanged) ---
def schedule_checkups(start_month):
    checkups = []
    today = datetime.date.today()
    for i in range(max(0, 9 - start_month + 1)):
        preg_month = start_month + i
        target_date = today + datetime.timedelta(days=30 * i)
        if 1 <= preg_month <= 7:
            checkups.append({"date": target_date.replace(day=15), "title": f"Std. Monthly Checkup (M{preg_month})", "type": "standard"})
        elif 7 < preg_month <= 9:
            checkups.append({"date": target_date.replace(day=7), "title": f"Std. Bi-Weekly #1 (M{preg_month})", "type": "standard"})
            checkups.append({"date": target_date.replace(day=21), "title": f"Std. Bi-Weekly #2 (M{preg_month})", "type": "standard"})
    st.session_state.checkups.extend(c for c in checkups if c not in st.session_state.checkups)
    st.session_state.checkups = sorted(st.session_state.checkups, key=lambda x: x['date'])

def display_reminders_sidebar():
    st.sidebar.title("🗓 Your Checkup Reminders")
    today = datetime.date.today(); reminders_found = False
    with st.sidebar.expander("🚨 Alerts: Next 7 Days", expanded=True):
        for checkup in st.session_state.get("checkups", []):
            days_until = (checkup['date'] - today).days
            if 0 <= days_until <= 7:
                reminders_found = True
                if days_until == 0: st.success(f"Today! - {checkup['title']}")
                elif days_until == 1: st.warning(f"Tomorrow! - {checkup['title']}")
                else: st.info(f"In {days_until} days - {checkup['title']}")
        if not reminders_found: st.write("No checkups in the next 7 days.")

def display_facts_page():
    st.header("💡 20 Surprising & Amazing Pregnancy Facts")
    st.markdown("---")
    for i, (title, text) in enumerate(PREGNANCY_FACTS):
        st.subheader(f"{i + 1}. {title}")
        st.write(text)
        st.markdown("---")
        
def display_essentials_gallery():
    st.header("🧺 Post-Delivery Essentials Gallery")
    st.markdown("A visual guide to help you prepare for the arrival of your little one and for your own recovery.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["For Your Baby 👶", "For the New Mom 🤱"])

    with tab1:
        st.subheader("Everything Your Newborn Needs")
        # Create a grid layout, 3 columns
        cols = st.columns(3)
        for i, item in enumerate(ESSENTIALS_DATA["baby"]):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{item['name']} {item['emoji']}**")
                    image_path = os.path.join(os.path.dirname(__file__), "assets", "essentials_images", "baby", item['img'])
                    if os.path.exists(image_path):
                        st.image(image_path)
                    st.write(item['desc'])

    with tab2:
        st.subheader("Your Postpartum Recovery Kit")
        # Create a grid layout, 3 columns
        cols = st.columns(3)
        for i, item in enumerate(ESSENTIALS_DATA["mom"]):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{item['name']} {item['emoji']}**")
                    image_path = os.path.join(os.path.dirname(__file__), "assets", "essentials_images", "mom", item['img'])
                    if os.path.exists(image_path):
                        st.image(image_path)
                    st.write(item['desc'])
        
def display_calendar_in_sidebar():
    with st.sidebar.expander("🗓 Your Checkup Calendar", expanded=False):
        st.subheader("Appointment Manager"); today = datetime.date.today()
        st.info(f"Today is: {today.strftime('%d %B, %Y')}")
        with st.form("new_checkup_form", clear_on_submit=True):
            new_title = st.text_input("Appointment Title"); new_date = st.date_input("Date")
            submitted = st.form_submit_button("Add to Calendar")
            if submitted and new_title:
                st.session_state.checkups.append({"date": new_date, "title": new_title, "type": "custom"})
                st.session_state.checkups = sorted(st.session_state.checkups, key=lambda x: x['date'])
                save_user_data(); st.success(f"Added '{new_title}'!"); time.sleep(1); st.rerun()
        col1, col2, col3 = st.columns([1, 2, 1])
        if 'cal_month' not in st.session_state: st.session_state.cal_month = today.month; st.session_state.cal_year = today.year
        if col1.button("⬅"):
            if st.session_state.cal_month == 1: st.session_state.cal_month = 12; st.session_state.cal_year -= 1
            else: st.session_state.cal_month -= 1
        col2.subheader(f"{month_name[st.session_state.cal_month]} {st.session_state.cal_year}")
        if col3.button("➡"):
            if st.session_state.cal_month == 12: st.session_state.cal_month = 1; st.session_state.cal_year += 1
            else: st.session_state.cal_month += 1
        st.markdown("---"); st.write(f"Appointments in {month_name[st.session_state.cal_month]}")
        events_this_month = [c for c in st.session_state.checkups if c['date'].month == st.session_state.cal_month and c['date'].year == st.session_state.cal_year]
        if events_this_month:
            for index, event in enumerate(events_this_month):
                c1, c2 = st.columns([4, 1]); icon = "🗓" if event.get('type') == 'standard' else "➕"
                c1.write(f"- {icon} {event['date'].strftime('%d')}: {event['title']}")
                if event.get('type') == 'custom':
                    if c2.button("❌", key=f"del_{index}", help="Delete"):
                        st.session_state.checkups.remove(event); save_user_data(); st.rerun()
        else: st.info("No appointments this month.")
        st.markdown("---")
        if st.button("📂 View All Checkups"): st.session_state.show_all_checkups = not st.session_state.get("show_all_checkups", False)
        if st.session_state.get("show_all_checkups", False):
            st.subheader("All Appointments Log")
            for event in st.session_state.checkups:
                status_icon = "✅" if event['date'] < today else "🗓"; status_text = "Completed" if event['date'] < today else "Upcoming"
                st.write(f"- {status_icon} *{event['date'].strftime('%d %b, %Y')}: {event['title']} *({status_text})")

# --- REPLACE THE OLD AI FUNCTION WITH THIS SIMPLER ONE ---

@st.cache_data(show_spinner=False)
def get_weekly_milestone(week):
    """Uses the LLM to generate just the milestone description for a specific week."""
    if llm is None:
        return "Could not connect to the AI model."
    
    emoji_map = {
        8: "❤️", 12: "👣", 16: "🖐️", 20: "👂",
        24: "😊", 28: "👀", 32: "🧠", 36: "😴"
    }
    closest_week = max([k for k in emoji_map if k <= week] or [8])
    emoji = emoji_map.get(closest_week, "👶")

    prompt = ChatPromptTemplate.from_template(
        "You are BabyNest. In one short, exciting, and wondrous sentence, describe a key developmental milestone for a baby in **week {week}** of pregnancy."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    milestone = chain.invoke({"week": week})["text"]
    
    return f"{emoji} {milestone}"

def display_baby_development_sidebar(current_week):
    with st.sidebar.container(border=True):
        st.subheader(f"👶 Baby's Development")
        st.markdown(f"#### Week {current_week}")

        image_week = 0
        if current_week >= 36: image_week = 36
        elif current_week >= 32: image_week = 32
        elif current_week >= 28: image_week = 28
        elif current_week >= 24: image_week = 24
        elif current_week >= 20: image_week = 20
        elif current_week >= 16: image_week = 16
        elif current_week >= 12: image_week = 12
        elif current_week >= 8: image_week = 8

        # --- THIS IS THE FIX ---
        # Get the absolute path to the directory where the script is running
        script_dir = os.path.dirname(__file__)
        # Create the full, absolute path to the image
        image_path = os.path.join(script_dir, "assets", "baby_images", f"week{image_week}.jpg")

        if os.path.exists(image_path):
            st.image(image_path)
        else:
            # Add a print statement for debugging if the image is not found
            print(f"DEBUG: Image not found at path: {image_path}")
            st.info("Development image for this week will be available soon.")

        with st.spinner("Loading weekly update..."):
            milestone_text, _ = get_weekly_milestone_and_size(current_week)

        st.markdown(
            f"""
            <p style="font-family: 'Comic Sans MS', cursive, sans-serif; font-size: 1.1rem; color: #555; text-align: center; padding: 0.5rem;">
                "{milestone_text}"
            </p>
            """,
            unsafe_allow_html=True
        )

def display_baby_development_sidebar(current_week):
    """Displays both the baby development and baby size widgets."""
    
    # --- Widget 1: Baby's Development ---
    with st.sidebar.container(border=True):
        st.subheader(f"👶 Baby's Development")
        st.markdown(f"#### Week {current_week}")

        # Image path logic for the 3D baby model
        image_week = max([w for w in [8, 12, 16, 20, 24, 28, 32, 36] if w <= current_week] or [0])
        script_dir = os.path.dirname(__file__)
        image_path = os.path.join(script_dir, "assets", "baby_images", f"week{image_week}.jpg")
        
        if os.path.exists(image_path):
            st.image(image_path)
        
        with st.spinner("Loading weekly update..."):
            # Call the new, simpler function
            milestone_text = get_weekly_milestone(current_week)
            
        st.markdown(f'<p style="font-family: \'Comic Sans MS\', cursive; font-size: 1.1rem; color: #555;">"{milestone_text}"</p>', unsafe_allow_html=True)
    # --- NEW: Call the function to display the size widget ---
    display_baby_size_sidebar(current_week, "")

def display_baby_size_sidebar(current_week, size_comparison_text):
    """Displays a sidebar widget showing the baby's approximate size with an image."""
    with st.sidebar.container(border=True):
        st.subheader("🍉 Baby's Size")
        st.markdown(f"#### Week {current_week}")

        # --- NEW LOGIC TO FIND THE CORRECT OBJECT AND IMAGE ---
        # Find the closest week in our dictionary that is less than or equal to the current week
        closest_week = max([w for w in BABY_SIZE_DATA if w <= current_week] or [min(BABY_SIZE_DATA.keys())])
        
        size_info = BABY_SIZE_DATA.get(closest_week)
        
        if size_info:
            object_name = size_info["object"]
            image_filename = size_info["image"]
            
            # Construct the full, absolute path to the image
            script_dir = os.path.dirname(__file__)
            image_path = os.path.join(script_dir, "assets", "size_images", image_filename)
            
            if os.path.exists(image_path):
                st.image(image_path)
            else:
                print(f"DEBUG: Size image not found at path: {image_path}")

            st.markdown(
                f'<p style="font-size: 1.1rem; color: #3a7ca5; text-align: center;">Your baby is about the size of <b>{object_name}</b>!</p>',
                unsafe_allow_html=True
            )
        else:
            st.info("Baby size information for this week will be available soon.")

def get_contraction_analysis(log):
    log_for_prompt = [{"duration_seconds": e["duration"], "frequency_minutes": e["frequency"]} for e in log if e.get("duration") and e.get("frequency")]
    prompt = ChatPromptTemplate.from_template("""You are an AI Labor Coach...""")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.invoke({"log": json.dumps(log_for_prompt, indent=2)})["text"]

def display_contraction_timer_sidebar():
    with st.sidebar.expander(" Labor CONTRACTION TIMER", expanded=True):
        if st.session_state.is_timing_contraction:
            if st.button("End Contraction", type="primary"):
                end_time = datetime.datetime.now(); start_time = st.session_state.current_contraction_start_time
                duration = round((end_time - start_time).total_seconds()); frequency = "N/A"
                completed_logs = [log for log in st.session_state.contraction_log if log.get("end_time")]
                if completed_logs:
                    last_start_time = completed_logs[-1]["start_time"]
                    frequency = round((start_time - last_start_time).total_seconds() / 60, 1)
                st.session_state.contraction_log[-1].update({"end_time": end_time, "duration": duration, "frequency": frequency})
                st.session_state.is_timing_contraction = False; save_user_data(); st.rerun()
        else:
            if st.button("Start Contraction"):
                st.session_state.is_timing_contraction = True; start_time = datetime.datetime.now()
                st.session_state.current_contraction_start_time = start_time
                st.session_state.contraction_log.append({"start_time": start_time, "end_time": None, "duration": "Timing...", "frequency": "N/A"})
                save_user_data(); st.rerun()
        if st.session_state.is_timing_contraction:
            elapsed_time = datetime.datetime.now() - st.session_state.current_contraction_start_time
            st.info(f"Timing... {round(elapsed_time.total_seconds())}s")
        st.subheader("Recent Contractions Log")
        if not st.session_state.contraction_log: st.write("No contractions logged yet.")
        else:
            display_log = st.session_state.contraction_log[-5:][::-1]
            log_data = {"Time": [e["start_time"].strftime("%I:%M:%S %p") for e in display_log], "Duration (sec)": [e.get("duration", "...") for e in display_log], "Frequency (min)": [e.get("frequency", "...") for e in display_log]}
            st.dataframe(log_data, use_container_width=True)
        if len([e for e in st.session_state.contraction_log if e.get('end_time')]) >= 2:
            col1, col2 = st.columns(2)
            if col1.button("Analyze Pattern"):
                with st.spinner("Analyzing..."):
                    one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
                    recent_log = [e for e in st.session_state.contraction_log if e.get('end_time') and e['start_time'] > one_hour_ago]
                    if len(recent_log) < 3: st.warning("Not enough data from the last hour.")
                    else:
                        analysis = get_contraction_analysis(recent_log)
                        st.session_state.messages.append({"role": "assistant", "content": f"Contraction Analysis:\n\n{analysis}"})
                        save_user_data(); st.rerun()
            if col2.button("Share with Doctor"): st.session_state.show_share_summary = True
    if st.session_state.get("show_share_summary", False):
        st.sidebar.markdown("---"); st.sidebar.subheader("📋 Summary for Your Doctor")
        one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
        recent_log = [e for e in st.session_state.contraction_log if e.get('end_time') and e['start_time'] > one_hour_ago]
        summary_text = f"Contraction Summary (Last Hour) as of {datetime.datetime.now().strftime('%I:%M %p')}:\n"
        if not recent_log: summary_text += "No contractions logged."
        else:
            for entry in recent_log: summary_text += f"\n- At {entry['start_time'].strftime('%I:%M %p')}: Lasted {entry['duration']} sec, Freq. was {entry['frequency']} min"
        st.sidebar.text_area("Copy this text:", summary_text, height=200)
        if st.sidebar.button("Close Summary"): st.session_state.show_share_summary = False; st.rerun()

def display_timeline_header():
    progress_info = calculate_pregnancy_progress(st.session_state.get("due_date"))
    if progress_info and progress_info["days_to_go"] >= 0:
        with st.container(border=True):
            st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; padding: 0.5rem;'>
                    <div style='text-align: center;'>
                        <p style='font-size: 0.9em; color: grey; margin-bottom: -5px;'>Your Progress</p>
                        <p style='font-size: 1.5em; font-weight: bold;'>{progress_info['weeks']}w, {progress_info['days']}d</p>
                    </div>
                    <div style='text-align: center;'>
                        <p style='font-size: 0.9em; color: grey; margin-bottom: -5px;'>Days to Go</p>
                        <p style='font-size: 1.5em; font-weight: bold;'>{progress_info['days_to_go']}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.progress(progress_info['progress'], text=f"{(progress_info['progress'] * 100):.1f}% Complete")
        st.markdown("<br>", unsafe_allow_html=True)

# --- Streamlit App UI and Logic ---

if st.session_state.stage == "onboarding":
    st.header("Welcome!")
    st.write("Please enter your estimated due date to create your personalized timeline.")
    due_date_input = st.date_input("Select your due date", min_value=datetime.date.today())
    if st.button("Confirm and Start", type="primary"):
        st.session_state.due_date = due_date_input; st.session_state.stage = "chat"
        progress = calculate_pregnancy_progress(due_date_input)
        start_month = progress['current_month'] if progress and progress['current_month'] > 0 else 1
        schedule_checkups(start_month)
        welcome_message = "Thank you! Your timeline is set up. How can I help you today?"
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
        save_user_data(); st.rerun()

elif st.session_state.stage == "chat":
    # --- Sidebar with NEW "Pregnancy Facts" button ---
    with st.sidebar:
        st.title("BabyNest Tools")
        st.subheader("Navigation")
        if st.button("💬 BabyNest Chat", use_container_width=True):
            st.session_state.view = "chat"
        if st.button("📝 My Daily Journal", use_container_width=True):
            st.session_state.view = "journal"
        if st.button("🤱 Motherhood Community", use_container_width=True):
            st.session_state.view = "community"
        if st.button("💡 Pregnancy Facts", use_container_width=True): # NEW BUTTON
            st.session_state.view = "facts"
        if st.button("🧺 Post-Delivery Essentials", use_container_width=True): # NEW BUTTON
            st.session_state.view = "essentials"

    # If user selected the JOURNAL view, show the journaling UI and hide the chat
    if st.session_state.get("view", "chat") == "journal":
        # Full-page journaling interface
        st.header("📝 My Daily Journal")
        st.markdown("Write and save daily reflections, notes, or anything you'd like to remember during your pregnancy.")
        journal_col_title, journal_col_date = st.columns([3,1])
        with journal_col_title:
            journal_title = st.text_input("Entry Title", value="")
        with journal_col_date:
            journal_date = st.date_input("Date", value=datetime.date.today())
        journal_content = st.text_area("Your entry", value="", height=350, placeholder="Write your thoughts here.")

        col_save, col_clear = st.columns([1,1])
        with col_save:
            if st.button("💾 Save Entry"):
                added = add_journal_entry(journal_title, journal_content, journal_date)
                if added:
                    st.success("Journal entry saved.")
                    # refresh in-memory entries
                    load_journal_entries()
                    st.rerun()
                else:
                    st.warning("Please write a title or some content before saving.")
        with col_clear:
            if st.button("🧹 Clear"):
                journal_title = ""; journal_content = ""
                st.rerun()

        st.markdown("---")
        st.subheader("Previous Entries")
        entries = st.session_state.get("journal_entries", [])
        if not entries:
            st.info("No journal entries yet. Use the form above to save your first entry.")
        else:
            # Show entries with expanders; newest first already arranged
            for entry in entries:
                entry_date = entry.get("date")
                entry_date_str = entry_date.strftime("%d %b, %Y") if isinstance(entry_date, datetime.date) else str(entry_date)
                with st.expander(f"{entry.get('title')} — {entry_date_str}", expanded=False):
                    st.markdown(f"Saved: {entry.get('created_at')}")
                    st.markdown("---")
                    st.write(entry.get("content"))
                    btn_col1, btn_col2 = st.columns([1,1])
                    with btn_col1:
                        if st.button("✏ Edit", key=f"edit_{entry.get('id')}"):
                            # Pre-fill the creation form by setting query params in session state then rerun
                            st.session_state.prefill_journal = {
                                "id": entry.get("id"),
                                "title": entry.get("title"),
                                "content": entry.get("content"),
                                "date": entry.get("date")
                            }
                            st.rerun()
                    with btn_col2:
                        if st.button("🗑 Delete", key=f"del_{entry.get('id')}"):
                            delete_journal_entry(entry.get("id"))
                            st.success("Entry deleted.")
                            st.rerun()

        # If edit prefill exists, show edit modal-like section at bottom
        if st.session_state.get("prefill_journal"):
            pre = st.session_state.prefill_journal
            st.markdown("---")
            st.subheader("Editing Entry")
            new_title = st.text_input("Edit Title", value=pre.get("title"))
            new_date = st.date_input("Edit Date", value=pre.get("date") if isinstance(pre.get("date"), datetime.date) else datetime.date.today())
            new_content = st.text_area("Edit Content", value=pre.get("content"), height=250)
            col_update, col_cancel = st.columns([1,1])
            with col_update:
                if st.button("Save Changes"):
                    # Replace entry
                    entries = st.session_state.get("journal_entries", [])
                    for idx, e in enumerate(entries):
                        if e.get("id") == pre.get("id"):
                            entries[idx]["title"] = new_title
                            entries[idx]["content"] = new_content
                            entries[idx]["date"] = new_date
                            entries[idx]["created_at"] = datetime.datetime.now().isoformat()
                            break
                    st.session_state.journal_entries = entries
                    save_journal_entries_to_disk()
                    st.session_state.prefill_journal = None
                    st.success("Entry updated.")
                    st.rerun()
            with col_cancel:
                if st.button("Cancel Edit"):
                    st.session_state.prefill_journal = None
                    st.rerun()
                    
    elif st.session_state.get("view") == "facts": # NEW VIEW
        display_facts_page()
        
    elif st.session_state.get("view") == "essentials": # NEW VIEW
        display_essentials_gallery()
        
    # If user selected the COMMUNITY view, show the motherhood community UI
    elif st.session_state.get("view", "chat") == "community":
        # Ensure community file exists
        os.makedirs("user_data", exist_ok=True)
        if not os.path.exists(COMMUNITY_FILE):
            with open(COMMUNITY_FILE, "w") as f:
                json.dump([], f)

        st.header("🤱 Motherhood Community")
        st.markdown("Share your experience and connect with moms in the same month/trimester. If you express negative feelings, BabyNest will offer an optimistic, supportive message.")

        # Select month/trimester
        stage = st.selectbox(
            "Choose your stage:",
            ["1st Month", "2nd Month", "3rd Month", "4th Month", "5th Month", "6th Month", "7th Month", "8th Month", "9th Month",
             "First Trimester", "Second Trimester", "Third Trimester"]
        )
        name_input = st.text_input("Your name (optional):", value="")
        post_text = st.text_area("Write your post (how you feel, new facts, mental/physical):", height=200)

        col_post, col_clear_post = st.columns([1,1])
        with col_post:
            if st.button("Share with Community"):
                if post_text.strip() == "":
                    st.warning("Please write something before posting.")
                else:
                    # load existing posts
                    with open(COMMUNITY_FILE, "r") as f:
                        try:
                            community_posts = json.load(f)
                        except Exception:
                            community_posts = []

                    # Prepare new post
                    new_post = {
                        "id": int(time.time() * 1000),
                        "name": name_input.strip() if name_input.strip() else "Anonymous",
                        "stage": stage,
                        "text": post_text.strip(),
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "replies": [],
                        "ai_supportive_reply": None  # will fill if LLM decides supportive message needed
                    }

                    # analyze with LLM for sentiment and supportive message
                    with st.spinner("Checking tone and preparing supportive feedback..."):
                        try:
                            emotion_prompt_template = ChatPromptTemplate.from_template(
                                "You are BabyNest, a compassionate assistant. Analyze the emotional tone of the following post from a pregnant mother and answer in JSON with keys: "
                                "\"tone\" (one of 'positive', 'neutral', 'negative'), and \"supportive_reply\" (a short uplifting message if tone is 'negative' else a short positive acknowledgement). "
                                "Be empathetic and concise (<= 40 words). Post: \"{post}\""
                            )
                            chain = LLMChain(llm=llm, prompt=emotion_prompt_template)
                            result_text = chain.invoke({"post": post_text.strip()})["text"].strip()

                            # Try to parse JSON from model. The model should return a JSON-like string.
                            try:
                                # model sometimes returns single quotes, so we normalize quotes:
                                normalized = result_text.replace("\n", " ").strip()
                                # Attempt to find JSON substring
                                json_start = normalized.find("{")
                                json_end = normalized.rfind("}") + 1
                                json_sub = normalized[json_start:json_end] if json_start != -1 and json_end != -1 else normalized
                                parsed = json.loads(json_sub)
                                tone = parsed.get("tone", "neutral")
                                supportive_reply = parsed.get("supportive_reply", None)
                            except Exception:
                                # fallback: basic keyword check
                                lower = post_text.lower()
                                negative_keywords = ["sad", "depress", "anxious", "anxiety", "lonely", "tired", "overwhelmed", "worried", "upset", "bad"]
                                if any(k in lower for k in negative_keywords):
                                    tone = "negative"
                                else:
                                    tone = "neutral"
                                supportive_reply = None

                            if tone == "negative" and not supportive_reply:
                                # ask LLM to craft a short optimistic message
                                supportive_prompt = ChatPromptTemplate.from_template(
                                    "You are BabyNest, a warm supportive assistant. Create a short, uplifting message (<= 25 words) to comfort a pregnant mother who is feeling down. Keep it empathetic and hopeful."
                                )
                                s_chain = LLMChain(llm=llm, prompt=supportive_prompt)
                                supportive_reply = s_chain.invoke({})["text"].strip()

                            # attach supportive reply if any
                            new_post["ai_supportive_reply"] = supportive_reply if supportive_reply else None

                        except Exception as e:
                            # On error, proceed without AI reply
                            new_post["ai_supportive_reply"] = None
                            print("Community LLM analysis error:", e)

                    community_posts.append(new_post)
                    with open(COMMUNITY_FILE, "w") as f:
                        json.dump(community_posts, f, indent=4)
                    st.success("Shared — thank you for opening up 💗")

        with col_clear_post:
            if st.button("Clear Post"):
                # lightweight clear by rerunning after setting empty
                st.experimental_set_query_params()  # no-op to trigger state refresh
                st.experimental_rerun()

        st.markdown("---")
        st.subheader("Community posts from moms in the same stage")
        # load posts and display those matching selected stage
        with open(COMMUNITY_FILE, "r") as f:
            try:
                all_posts = json.load(f)
            except Exception:
                all_posts = []

        # filter by stage
        filtered = [p for p in all_posts if p.get("stage") == stage]
        if not filtered:
            st.info("No posts yet in this stage. Be the first to share your story 💖")
        else:
            # show newest first
            for idx, post in enumerate(reversed(filtered)):
                # render post card
                st.markdown(
                    f"""
                    <div style='background:#fff7fb; border-radius:12px; padding:12px; margin-bottom:10px; box-shadow:0 2px 6px rgba(0,0,0,0.05);'>
                        <b style='color:#d63384;'>{post.get('name')}</b> • <span style='color:#3a7ca5'>{post.get('stage')}</span>
                        <div style='color:#444; margin-top:8px;'>{post.get('text')}</div>
                        <div style='color:#777; font-size:0.85em; margin-top:8px;'>{post.get('timestamp')}</div>
                    """
                    , unsafe_allow_html=True
                )

                # If AI supportive reply exists, show it visibly under the post (optimistic feedback)
                if post.get("ai_supportive_reply"):
                    st.markdown(
                        f"<div style='margin-top:8px; background:#fff2f8; padding:8px; border-radius:8px;'><b>BabyNest:</b> {post.get('ai_supportive_reply')}</div>",
                        unsafe_allow_html=True,
                    )

                # Display replies (if any)
                if post.get("replies"):
                    for r in post.get("replies"):
                        st.markdown(
                            f"<div style='margin-left:12px; margin-top:8px; padding:8px; background:#f0fbff; border-radius:8px;'><b>Reply:</b> {r}</div>",
                            unsafe_allow_html=True
                        )

                # Reply input per displayed post (use unique keys)
                reply_key_input = f"community_reply_input_{post.get('id')}"
                reply_text = st.text_input("Write a kind reply", key=reply_key_input, placeholder="Encourage or share your experience...")

                reply_button_key = f"community_reply_button_{post.get('id')}"
                if st.button("Reply", key=reply_button_key):
                    if reply_text.strip():
                        # Add reply to the original post in all_posts
                        for p in all_posts:
                            if p.get("id") == post.get("id"):
                                p.setdefault("replies", []).append(reply_text.strip())
                                break
                        with open(COMMUNITY_FILE, "w") as f:
                            json.dump(all_posts, f, indent=4)
                        st.success("Reply added 💬")
                        time.sleep(0.6)
                        st.experimental_rerun()
                    else:
                        st.warning("Please write something before replying.")

                # close post card html
                st.markdown("</div>", unsafe_allow_html=True)

    else:
        # DEFAULT: Chat view (original behavior)
        with st.sidebar:
            st.title("BabyNest Tools")
            progress_info = calculate_pregnancy_progress(st.session_state.due_date)
            current_week = progress_info.get("weeks", 0) if progress_info else 0
            if current_week > 4:
                display_baby_development_sidebar(current_week)
            with st.container(border=True):
                st.subheader("🚨 Emergency SOS")
                if st.button("Initiate Emergency Call", type="primary"):
                    if st.session_state.due_date is None: st.error("Please complete onboarding first.")
                    else:
                        with st.spinner("Initiating call..."):
                            current_month = progress_info.get("current_month", 0) if progress_info else 0
                            if 1 <= current_month <= 6:
                                target_number = HUSBAND_PHONE_NUMBER; message_to_say = "This is an automated alert from BabyNest..."
                            else:
                                target_number = HOSPITAL_NURSE_LINE_NUMBER; message_to_say = f"This is an urgent alert from BabyNest for a user in month {current_month}..."
                            success, status = make_voice_call(target_number, message_to_say)
                        if success: st.sidebar.success(status)
                        else: st.sidebar.error(status)
            display_reminders_sidebar()
            display_calendar_in_sidebar()
            current_month = progress_info.get("current_month", 0) if progress_info else 0
            if current_month >= 7:
                with st.container(border=True):
                    display_contraction_timer_sidebar()
            st.markdown("---")
            if st.button("Reset All Data and Start Over"):
                reset_chat()

        display_timeline_header()
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])

        if user_input := st.chat_input("Ask about your pregnancy wellness..."):
            with st.chat_message("user"): st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chain = get_conversational_rag_chain()
                    st.session_state.memory.chat_memory.add_user_message(user_input)
                    response = chain.invoke({"input": user_input, "pregnancy_week": f"Week {current_week}", "chat_history": st.session_state.memory.chat_memory.messages})
                    cleaned_response = clean_response(response["answer"])
                    st.markdown(cleaned_response)
                    st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
                    st.session_state.memory.chat_memory.add_ai_message(cleaned_response)
                    save_user_data()

# Footer (same as before)
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#888; font-size:0.9em; margin-top:40px;'>
        Made with 💕 by <b>BabyNest</b> — Your caring companion during motherhood 🌸<br>
        <span style='font-size:0.85em;'>Powered by GROQ AI</span>
    </div>
    """,
    unsafe_allow_html=True,
)