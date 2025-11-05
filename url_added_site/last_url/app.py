import streamlit as st
import re
from main_app import Assistant
from fpdf import FPDF

# --- Helper function for PDF Generation ---
def create_pdf_from_text(text: str, title: str = "Report"):
    """Creates a PDF file in memory from a string of text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add a title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(10) # Add a little space
    
    # Add the body text
    pdf.set_font("Arial", size=12)
    # Use multi_cell to handle line breaks automatically
    # We need to encode the text to latin-1 for FPDF
    pdf.multi_cell(0, 10, txt=text.encode('latin-1', 'replace').decode('latin-1'))
    
    # Return the PDF as bytes
    return pdf.output(dest='S').encode('latin-1')

# --- Page Configuration ---
st.set_page_config(page_title="🤖 AI Advisor", layout="wide")
st.title("🤖 AI Shopping & Research Advisor")

# --- Helper Functions & State Management ---
@st.cache_resource
def get_assistant():
    """Caches the Assistant object to avoid re-initializing it on every interaction."""
    return Assistant()

# --- Initialize all session state variables at the top ---
if "mode" not in st.session_state:
    st.session_state.mode = "Shopping Advisor"

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {
        "Shopping Advisor": [],
        "Crunchbase Q&A": [],
        "Yahoo Finance Q&A": [] # Added new mode history
    }

if "crunchbase_url" not in st.session_state:
    st.session_state.crunchbase_url = None

if "crunchbase_history" not in st.session_state:
    st.session_state.crunchbase_history = []

if "crunchbase_raw_data" not in st.session_state:
    st.session_state.crunchbase_raw_data = None

# New state for Yahoo Finance
if "yahoo_finance_url" not in st.session_state:
    st.session_state.yahoo_finance_url = None

if "yahoo_finance_history" not in st.session_state:
    st.session_state.yahoo_finance_history = []

if "yahoo_finance_raw_data" not in st.session_state:
    st.session_state.yahoo_finance_raw_data = None


assistant = get_assistant()

# --- Sidebar for Controls and History ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    mode = st.radio(
        "Choose your Assistant Mode:",
        ("Shopping Advisor", "Crunchbase Q&A", "Yahoo Finance Q&A"), # Added new mode
        key="mode_selection"
    )
    # Update the mode in session state when the radio button changes
    if st.session_state.mode != mode:
        st.session_state.mode = mode
        st.rerun()

    st.markdown("---")
    st.header("📜 Search History")
    
    with st.expander("View History", expanded=False):
        history_for_mode = st.session_state.chat_histories[st.session_state.mode]
        if not history_for_mode:
            st.write("No history yet.")
        else:
            for message in reversed(history_for_mode):
                if message["role"] == "user":
                    st.info(message["content"])

    if st.session_state.mode == "Shopping Advisor":
        st.markdown("---")
        st.subheader("Amazon Settings")
        domain = st.selectbox("Select Amazon Region:", ("in", "com", "co.uk", "de"), index=0)
        assistant.amazon_domain_tld = domain
        st.info("You are in Shopping Advisor mode. Ask for product recommendations.")

    elif st.session_state.mode == "Crunchbase Q&A":
        st.markdown("---")
        st.subheader("Crunchbase Settings")
        if st.button("Clear Crunchbase Context"):
            st.session_state.crunchbase_url = None
            st.session_state.crunchbase_history = []
            st.session_state.chat_histories["Crunchbase Q&A"] = []
            st.session_state.crunchbase_raw_data = None
            st.success("Context cleared.")
            st.rerun()
        
        if st.session_state.crunchbase_url:
            st.success(f"Context Loaded: {st.session_state.crunchbase_url}")
        else:
            st.info("First, paste a Crunchbase URL in the chat to load its data.")

    # New UI for Yahoo Finance mode
    elif st.session_state.mode == "Yahoo Finance Q&A":
        st.markdown("---")
        st.subheader("Yahoo Finance Settings")
        if st.button("Clear Yahoo Finance Context"):
            st.session_state.yahoo_finance_url = None
            st.session_state.yahoo_finance_history = []
            st.session_state.chat_histories["Yahoo Finance Q&A"] = []
            st.session_state.yahoo_finance_raw_data = None
            st.success("Context cleared.")
            st.rerun()
        
        if st.session_state.yahoo_finance_url:
            st.success(f"Context Loaded: {st.session_state.yahoo_finance_url}")
        else:
            st.info("First, paste a Yahoo Finance URL in the chat to load its data (e.g., for Apple: https://finance.yahoo.com/quote/AAPL).")


# --- Main Chat Interface ---
for message in st.session_state.chat_histories[st.session_state.mode]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Backend Logic ---
if prompt := st.chat_input("What are you looking for?"):
    st.session_state.chat_histories[st.session_state.mode].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        response = ""
        # SHOPPING ADVISOR MODE
        if st.session_state.mode == "Shopping Advisor":
            with st.spinner("Thinking..."):
                response = assistant.find_and_recommend_product(prompt)
            message_placeholder.markdown(response)

        # CRUNCHBASE Q&A MODE
        elif st.session_state.mode == "Crunchbase Q&A":
            if re.search(r'crunchbase\.com/organization', prompt):
                with st.spinner("Loading comprehensive data from Crunchbase..."):
                    response, success = assistant.load_url_for_qa(prompt)
                if success:
                    st.session_state.crunchbase_url = prompt
                    st.session_state.crunchbase_history = []
                    st.session_state.crunchbase_raw_data = assistant.crunchbase_agent.get_cached_data(prompt)
                message_placeholder.markdown(response)
            
            elif st.session_state.crunchbase_url:
                with st.spinner("Thinking..."):
                    response = assistant.answer_crunchbase_question(prompt, st.session_state.crunchbase_url, st.session_state.crunchbase_history)
                    st.session_state.crunchbase_history.append((prompt, response))
                message_placeholder.markdown(response)

                if "crunchbase_raw_data" in st.session_state and st.session_state.crunchbase_raw_data:
                    org_name = st.session_state.crunchbase_url.split("organization/")[-1]
                    pdf_bytes = create_pdf_from_text(st.session_state.crunchbase_raw_data, title=f"Crunchbase Report: {org_name}")
                    st.download_button(label="⬇️ Download Full Report as PDF", data=pdf_bytes, file_name=f"{org_name}_report.pdf", mime="application/pdf")
            else:
                response = "Please provide a Crunchbase organization URL first."
                message_placeholder.markdown(response)
        
        # YAHOO FINANCE Q&A MODE
        elif st.session_state.mode == "Yahoo Finance Q&A":
            if re.search(r'finance\.yahoo\.com/quote/', prompt):
                with st.spinner("Loading comprehensive data from Yahoo Finance..."):
                    response, success = assistant.load_url_for_qa(prompt)
                if success:
                    st.session_state.yahoo_finance_url = prompt
                    st.session_state.yahoo_finance_history = []
                    # Get and store the raw data for PDF download
                    st.session_state.yahoo_finance_raw_data = assistant.yahoo_finance_agent.get_cached_data(prompt)
                message_placeholder.markdown(response)
            
            elif st.session_state.yahoo_finance_url:
                with st.spinner("Thinking..."):
                    response = assistant.answer_yahoo_finance_question(prompt, st.session_state.yahoo_finance_url, st.session_state.yahoo_finance_history)
                    st.session_state.yahoo_finance_history.append((prompt, response))
                message_placeholder.markdown(response)

                # Add PDF download button for Yahoo Finance data
                if "yahoo_finance_raw_data" in st.session_state and st.session_state.yahoo_finance_raw_data:
                    ticker = st.session_state.yahoo_finance_url.split("quote/")[-1].split('/')[0]
                    pdf_bytes = create_pdf_from_text(st.session_state.yahoo_finance_raw_data, title=f"Yahoo Finance Report: {ticker}")
                    st.download_button(label="⬇️ Download Full Report as PDF", data=pdf_bytes, file_name=f"{ticker}_report.pdf", mime="application/pdf")
            else:
                response = "Please provide a Yahoo Finance quote URL first (e.g., https://finance.yahoo.com/quote/AAPL)."
                message_placeholder.markdown(response)

        st.session_state.chat_histories[st.session_state.mode].append({"role": "assistant", "content": response})