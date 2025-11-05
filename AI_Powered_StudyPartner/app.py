import streamlit as st
from ui import setup_ui

def initialize_session_state():
    """Initializes all the necessary session state variables."""
    # App mode state for sidebar navigation
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Elley AI Chat"
        
    # File and text processing state
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None
    if 'pdf_text' not in st.session_state: # This will hold text of ALL docs
        st.session_state.pdf_text = ""
    if 'document_topics' not in st.session_state: # This will hold topics of ALL docs
        st.session_state.document_topics = None
        
    # --- NEW: State for the currently selected file and its specific content ---
    if 'selected_file_name' not in st.session_state:
        st.session_state.selected_file_name = None
    if 'active_pdf_text' not in st.session_state: # Text of the selected file
        st.session_state.active_pdf_text = ""
    if 'active_document_topics' not in st.session_state: # Topics of the selected file
        st.session_state.active_document_topics = None
    if 'active_extracted_images' not in st.session_state: # Images of the selected file
        st.session_state.active_extracted_images = []
    # --- END NEW ---
        
    if 'selected_topic' not in st.session_state:
        st.session_state.selected_topic = "Entire Document"
        
    # Elley AI Chatbot state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Learn Deck Quiz state
    if 'quiz_generated' not in st.session_state:
        st.session_state.quiz_generated = False
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = []
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = [None] * 100
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'weak_area_analysis' not in st.session_state:
        st.session_state.weak_area_analysis = ""
    if 'quiz_context' not in st.session_state:
        st.session_state.quiz_context = ""
    if 'card_flipped' not in st.session_state:
        st.session_state.card_flipped = False
    if 'show_detailed_explanation' not in st.session_state:
        st.session_state.show_detailed_explanation = [False] * 100
    if 'quiz_source_was_entire_document' not in st.session_state:
        st.session_state.quiz_source_was_entire_document = False
    if 'uncovered_topics' not in st.session_state:
        st.session_state.uncovered_topics = None

    # New states for cumulative quiz scoring
    if 'total_score' not in st.session_state:
        st.session_state.total_score = 0
    if 'total_questions_mcq' not in st.session_state:
        st.session_state.total_questions_mcq = 0
    if 'all_wrong_questions' not in st.session_state:
        st.session_state.all_wrong_questions = []


def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title="CogniScan", layout="wide", initial_sidebar_state="expanded")

    initialize_session_state()

    with st.sidebar:
        st.header("CogniScan Dashboard")
        
        st.subheader("Features")
        
        # Use buttons for a fixed tab-like navigation
        if st.button("Elley AI Chat", use_container_width=True, type="primary" if st.session_state.app_mode == "Elley AI Chat" else "secondary"):
            st.session_state.app_mode = "Elley AI Chat"
            st.rerun()

        if st.button("Learn Deck", use_container_width=True, type="primary" if st.session_state.app_mode == "Learn Deck" else "secondary"):
            st.session_state.app_mode = "Learn Deck"
            st.rerun()

    # Display the title and setup the main UI
    st.title("📚 CogniScan: Your AI-Powered Study Partner")
    setup_ui(st.session_state.app_mode)


if __name__ == "__main__":
    main()