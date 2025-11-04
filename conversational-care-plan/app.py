import streamlit as st
import json
import re
from dotenv import load_dotenv
import logging

# --- LangChain & Groq Imports ---
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- Configuration and Initial Setup ---
load_dotenv()
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# --- The list of initial questions the AI will ask ---
INITIAL_QUESTIONS = [
    ("main_goal", "To start, what is your main, specific wellness goal? (e.g., clear up my pimples, lose 10 pounds, fade dark spots, manage PMS)"),
    ("physical_feeling", "How are you feeling physically right now? (e.g., tired, energetic, specific pains)"),
    ("age_gender_height_weight", "For a more accurate plan, could you share your age, gender, height, and weight?"),
    ("exercise_habits", "Describe your current weekly exercise routine, if any."),
    ("eating_habits", "Describe your typical daily diet. What do you usually eat for breakfast, lunch, and dinner?"),
    ("self_care_obstacles", "What is the biggest challenge preventing you from achieving this goal right now?"),
    ("timeline_goal", "How many months would you like this plan to be? (Please enter a number from 1 to 6)")
]

# --- Initialize session state variables ---
if "stage" not in st.session_state:
    st.session_state.stage = "interview"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am an AI Wellness Protocol Designer..."}] # Abridged for brevity
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "question_index" not in st.session_state:
    st.session_state.question_index = 0

# --- Core AI Functions ---
@st.cache_resource
def initialize_system():
    llm = ChatGroq(model_name=GROQ_LLM_MODEL)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    return llm, vector_store

llm, vector_store = initialize_system()

def generate_plan_with_rag(profile, num_weeks):
    # --- THIS IS THE NEW "SPECIALIST PROGRAM DESIGNER" PROMPT ---
    prompt_template = """
    You are a Specialist Health Protocol Designer. Your task is to use your expert knowledge to synthesize a user's profile with evidence-based wellness guides to create a hyper-specific, actionable, and productive plan.

    **SAFETY PRECAUTION:** You are NOT a medical doctor. For severe medical issues, you MUST refuse and advise seeing a professional.

    **USER PROFILE:**
    {profile}

    **CONTEXT from Relevant Wellness Program Modules:**
    {context}

    **TASK:**
    Design a systematic and progressive **{num_weeks}-week protocol** that is **laser-focused** on the user's primary goal. DO NOT give generic advice. Be creative, detailed, and create a plan designed to produce real results.

    **STRUCTURE:**
    ### Your Hyper-Specific {num_weeks}-Week Protocol for: [State The User's Goal Here]

    **1. The Scientific Strategy:**
    - Explain the "why" behind the plan. What is the core scientific principle we are using to achieve the user's goal? (e.g., "To clear acne, our strategy is to reduce inflammation and regulate sebum production...").

    **2. Your Nutritional Protocol:**
    - **Macro/Micronutrient Focus:** What key nutrients are most important for this goal? (e.g., "For muscle gain, we will focus on protein intake...").
    - **Specific Meal Improvements:** Provide 3-5 concrete "swap" suggestions. (e.g., "Swap your morning sugary cereal for oatmeal with berries and nuts.").
    - **Sample Shopping List:** Generate a bulleted list of 5-10 key items from the context that the user should buy.

    **3. Your Lifestyle & Exercise Protocol:**
    - Detail any specific lifestyle changes required (e.g., sleep hygiene, skincare steps).
    - Provide a clear, actionable exercise plan if relevant to the goal.

    **4. Your Week-by-Week Progressive Plan:**
    - For EACH week from Week 1 to Week {num_weeks}:
        - **`### Week [Number]: [Theme of the Week]`**
        - **`* Focus:`** The main objective for that week.
        - **`* Detailed Action Items:`** A specific, bulleted list of tasks for diet, lifestyle, and exercise. Build upon the previous week's tasks. This must be detailed and not generic.

    **5. A Note of Encouragement:**
    - A final positive message.

    **Disclaimer:**
    - Always end with the required disclaimer about not being medical advice.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(), llm=llm
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever_from_llm, document_chain)
    
    main_goal = profile.get("main_goal", "general wellness")
    
    response = retrieval_chain.invoke({
        "profile": json.dumps(profile, indent=2),
        "input": main_goal,
        "num_weeks": num_weeks
    })
    return response["answer"]

# --- Streamlit App UI and Logic (Same as before) ---
st.title("🤖 AI Wellness Protocol Designer")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.stage == "interview":
    if len(st.session_state.messages) == 1:
        _, question = INITIAL_QUESTIONS[0]
        st.session_state.messages.append({"role": "assistant", "content": question})
        with st.chat_message("assistant"):
            st.markdown(question)

    elif st.session_state.messages[-1]["role"] == "user":
        if st.session_state.question_index < len(INITIAL_QUESTIONS):
            _, question = INITIAL_QUESTIONS[st.session_state.question_index]
            with st.chat_message("assistant"):
                st.markdown(question)
            st.session_state.messages.append({"role": "assistant", "content": question})
        else:
            st.session_state.stage = "generating"
            st.rerun()

if user_input := st.chat_input("Your response..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.stage == "interview":
        key, _ = INITIAL_QUESTIONS[st.session_state.question_index]
        if key == "timeline_goal":
            try:
                months = int(re.search(r'\d+', user_input).group())
                if 1 <= months <= 6:
                    st.session_state.user_profile[key] = user_input
                    st.session_state.question_index += 1
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "Please enter a number between 1 and 6 months."})
            except (ValueError, AttributeError):
                 st.session_state.messages.append({"role": "assistant", "content": "That doesn't look like a valid number. Please enter a number between 1 and 6."})
        else:
            st.session_state.user_profile[key] = user_input
            st.session_state.question_index += 1
        st.rerun()

if st.session_state.stage == "generating":
    with st.chat_message("assistant"):
        with st.spinner("Designing your hyper-specific protocol... This may take a moment."):
            timeline_str = st.session_state.user_profile.get("timeline_goal", "1 month")
            try:
                num_months = int(re.search(r'\d+', timeline_str).group())
                num_weeks = num_months * 4
            except:
                num_weeks = 4
            final_plan = generate_plan_with_rag(st.session_state.user_profile, num_weeks)
            st.markdown(final_plan)
            st.session_state.messages.append({"role": "assistant", "content": final_plan})
    st.session_state.stage = "done"
    st.rerun()

if st.session_state.stage == "done":
    st.success("Your personalized protocol has been generated!")
    if st.button("Start Over"):
        st.session_state.clear()
        st.rerun()