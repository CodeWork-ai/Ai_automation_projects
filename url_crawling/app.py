import streamlit as st
import re
from main_app import Assistant

# Page configuration
st.set_page_config(page_title="AI Advisor", layout="wide")

st.title("🤖 AI Shopping & Research Advisor")

# --- Helper Functions & State Management ---

@st.cache_resource
def get_assistant():
    """Caches the Assistant object to avoid re-initializing it on every interaction."""
    return Assistant()

# Initialize session state variables
if "mode" not in st.session_state:
    st.session_state.mode = "Shopping Advisor"

# Use a dictionary to store separate chat histories for each mode
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {
        "Shopping Advisor": [],
        "Crunchbase Q&A": []
    }

if "crunchbase_url" not in st.session_state:
    st.session_state.crunchbase_url = None
if "crunchbase_history" not in st.session_state:
    st.session_state.crunchbase_history = []

assistant = get_assistant()

# --- Sidebar for Mode Selection and Controls ---

with st.sidebar:
    st.header("Controls")
    
    # The main switch for the app's mode
    mode = st.radio(
        "Choose your Assistant Mode:",
        ("Shopping Advisor", "Crunchbase Q&A"),
        key="mode_selection"
    )
    st.session_state.mode = mode

    if st.session_state.mode == "Shopping Advisor":
        st.markdown("---")
        st.subheader("Amazon Settings")
        domain = st.selectbox(
            "Select Amazon Region:",
            ("in", "com", "co.uk", "de"),
            index=0 # Default to 'in'
        )
        assistant.amazon_domain_tld = domain
        st.info("You are in Shopping Advisor mode. Ask for product recommendations.")

    elif st.session_state.mode == "Crunchbase Q&A":
        st.markdown("---")
        st.subheader("Crunchbase Settings")
        if st.button("Clear Crunchbase Context"):
            st.session_state.crunchbase_url = None
            st.session_state.crunchbase_history = []
            st.session_state.chat_histories["Crunchbase Q&A"] = [] # Clear only the Crunchbase chat
            st.success("Context cleared.")

        
        if st.session_state.crunchbase_url:
            st.success(f"Context Loaded: {st.session_state.crunchbase_url}")
        else:
            st.info("First, paste a Crunchbase URL in the chat to load its data.")


# --- Main Chat Interface ---

# Display existing chat messages for the current mode
for message in st.session_state.chat_histories[st.session_state.mode]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Get user input
if prompt := st.chat_input("What are you looking for?"):
    # Add user message to chat history
# For the user message
    st.session_state.chat_histories[st.session_state.mode].append({"role": "user", "content": prompt})

    # --- Backend Logic Execution ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            
            response = ""
            # SHOPPING ADVISOR MODE
            if st.session_state.mode == "Shopping Advisor":
                response = assistant.find_and_recommend_product(prompt)

            # CRUNCHBASE Q&A MODE
            elif st.session_state.mode == "Crunchbase Q&A":
                # Check if the user provided a URL to load context
                if re.search(r'crunchbase\.com/organization/', prompt):
                    response, success = assistant.load_url_for_qa(prompt)
                    if success:
                        st.session_state.crunchbase_url = prompt
                        st.session_state.crunchbase_history = [] # Reset history for new URL
                
                # If context is already loaded, treat input as a question
                elif st.session_state.crunchbase_url:
                    response = assistant.answer_crunchbase_question(
                        prompt,
                        st.session_state.crunchbase_url,
                        st.session_state.crunchbase_history
                    )
                    # Add to conversation history for follow-ups
                    st.session_state.crunchbase_history.append((prompt, response))
                
                # If no context is loaded, ask for a URL
                else:
                    response = "Please provide a Crunchbase organization URL first."

            message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.chat_histories[st.session_state.mode].append({"role": "assistant", "content": response})
