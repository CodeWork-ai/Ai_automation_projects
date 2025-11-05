import streamlit as st
import json
import os
from pdf_processor import process_pdf_text, get_text_from_page, extract_images_from_pdf
from groq_logic import (
    chat_with_elley,
    simplify_and_structure_text,
    answer_question_from_text,
    generate_summary,
    generate_quiz_from_text,
    get_brief_explanation,
    get_concept_explanation,
    analyze_weak_areas,
    extract_topics_from_text,
    find_uncovered_topics
)

# --- HELPER FUNCTIONS ---

def load_css(file_name):
    """Loads a CSS file and injects it into the Streamlit app."""
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_styled_option(status, text):
    """Returns an HTML string for a styled MCQ option based on its status."""
    icon_map = {
        "correct": "✔️",
        "incorrect": "❌",
    }
    if status in icon_map:
        return f'<div class="option-box option-{status}">{icon_map[status]} {text}</div>'
    else: # Neutral
        return f'<div class="option-box option-neutral">{text}</div>'

def start_quiz(context, num_questions, quiz_type, is_follow_up=False):
    """Helper function to start the quiz generation process."""
    if not is_follow_up:
        # Reset cumulative scores only for a brand new quiz session
        st.session_state.total_score = 0
        st.session_state.total_questions_mcq = 0
        st.session_state.all_wrong_questions = []

    with st.spinner("AI is crafting your quiz..."):
        quiz_json_str = generate_quiz_from_text(context, num_questions, quiz_type)
        try:
            quiz_data = json.loads(quiz_json_str)
            st.session_state.quiz_questions = quiz_data.get("questions", [])
            st.session_state.quiz_generated = True
            st.session_state.current_question_index = 0
            st.session_state.user_answers = [None] * len(st.session_state.quiz_questions)
            st.session_state.show_detailed_explanation = [False] * len(st.session_state.quiz_questions)
            st.session_state.card_flipped = False
            st.session_state.show_results = False
            st.session_state.weak_area_analysis = ""
            st.session_state.uncovered_topics = None
            st.rerun()
        except json.JSONDecodeError:
            st.error("Failed to generate the quiz. The AI's response was not in the correct format. Please try again.")
            st.code(quiz_json_str)

def process_selected_file(selected_file_name):
    """Processes the selected file for text and topics if it hasn't been processed already."""
    # Find the file object from the uploaded files list
    selected_file = next((f for f in st.session_state.uploaded_files if f.name == selected_file_name), None)
    if not selected_file:
        st.error("Selected file not found. Please try uploading again.")
        return

    # Trigger processing for the selected file
    with st.spinner(f"Analyzing '{selected_file.name}'... This may take a moment."):
        # Process for text
        st.session_state.active_pdf_text = process_pdf_text(selected_file)
        # Process for images
        st.session_state.active_extracted_images = extract_images_from_pdf(selected_file)
        
        # Process for topics
        topics_json_str = extract_topics_from_text(st.session_state.active_pdf_text)
        try:
            st.session_state.active_document_topics = json.loads(topics_json_str)
        except json.JSONDecodeError:
            st.session_state.active_document_topics = {}
            st.error("Could not extract distinct topics from the document. You can still interact with the entire text.")
    st.success(f"Processed '{selected_file.name}' successfully!")
    st.rerun()
    
# --- FEATURE RENDERING FUNCTIONS ---
def render_simplify_note_content(context):
    """Renders the content for the Simplify Note tab."""
    if not context:
        st.info("Select a file and topic to simplify.")
        return
    st.markdown("This tool will break down the selected text into a more digestible, structured format.")
    if st.button("✨ Simplify Your Note", use_container_width=True, type="primary"):
        with st.spinner("Elley is creating a simplified summary..."):
            simplified_text = simplify_and_structure_text(context)
            st.markdown(simplified_text)

def render_summary_generator_content(context):
    """Renders the content for the Summary Generation tab."""
    if not context:
        st.info("Select a file and topic to generate a summary.")
        return
    st.markdown("Select the type of summary you want to create from the selected topic.")
    summary_type = st.selectbox(
        "Choose summary length and style:",
        ("Quick Overview", "Detailed Summary", "Key Points Breakdown"),
        key="summary_type"
    )
    if st.button("Generate Summary", use_container_width=True, type="primary"):
        with st.spinner(f"Generating a {summary_type}..."):
            summary = generate_summary(context, summary_type)
            st.success("Summary generated successfully!")
            st.markdown(summary)

def render_note_lens_content():
    """Renders the content for the Note Lens tab."""
    if not st.session_state.get('selected_file_name'):
        st.info("Please select a document to activate the Note Lens.")
        return
        
    selected_file = next((f for f in st.session_state.uploaded_files if f.name == st.session_state.selected_file_name), None)

    st.markdown(f"Use this feature to ask a question about a single page of **{st.session_state.selected_file_name}**.")
    col1, col2 = st.columns([1, 3])
    with col1:
        page_num = st.number_input("Enter Page Number:", min_value=1, step=1, key="note_lens_page")
    with col2:
        question = st.text_input("What is your question about this page?", key="note_lens_question", placeholder="e.g., 'What does the first paragraph mean?'")
    
    if st.button("Analyze Page with Note Lens", use_container_width=True, type="primary"):
        if not question:
            st.warning("Please enter a question about the page.")
        else:
            with st.spinner(f"Note Lens is scanning page {page_num} of {selected_file.name}..."):
                page_text = get_text_from_page(selected_file, page_num)
                if "Error:" in page_text:
                    st.error(page_text)
                else:
                    answer = answer_question_from_text(page_text, question)
                    st.info(f"**Answer from {selected_file.name}, Page {page_num}:**")
                    st.markdown(answer)

def render_elley_ai_chat():
    """Renders the entire UI for the Elley AI Chat feature."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("Elley AI Assistant")
        st.markdown("Upload your study materials, select a file, and interact with Elley to simplify, summarize, and query your document.")
    with col2:
        uploaded_files = st.file_uploader(
            "Upload your study material(s) here", 
            type="pdf", 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
    
    if not st.session_state.get('uploaded_files'):
        st.info("Upload one or more PDFs to begin.")
        return
        
    st.markdown("---")
    
    # --- UI FOR FILE AND TOPIC SELECTION ---
    selection_tab, tools_tab = st.tabs(["📁 Select File & Topic", "🛠️ Document Tools & Chat"])
    
    with selection_tab:
        st.subheader("Step 1: Select a File")
        file_names = [f.name for f in st.session_state.uploaded_files]
        
        # Use on_change to trigger processing when a new file is selected
        selected_file_name = st.selectbox(
            "Choose a document to focus on:", 
            file_names, 
            index=file_names.index(st.session_state.selected_file_name) if st.session_state.selected_file_name in file_names else 0,
            key="file_selector"
        )
        
        # If selection changes, update state and trigger processing
        if selected_file_name != st.session_state.selected_file_name:
            st.session_state.selected_file_name = selected_file_name
            st.session_state.messages = [] # Clear chat history for new file
            process_selected_file(selected_file_name)

        # Initialize the first file if none is selected
        if not st.session_state.selected_file_name and file_names:
            st.session_state.selected_file_name = file_names[0]
            process_selected_file(file_names[0])
            
        context_text = ""
        selected_topic_summary = "Entire Document"
        
        if st.session_state.get('active_document_topics') is not None:
            st.subheader("Step 2: Select Topic(s) to Focus On")
            topic_options = ["Entire Document"] + list(st.session_state.active_document_topics.keys())
            
            selected_topics = st.multiselect(
                "Choose topics (select 'Entire Document' to use all content):", 
                topic_options, 
                key="topic_multiselector"
            )
            
            if "Entire Document" in selected_topics or not selected_topics:
                context_text = st.session_state.active_pdf_text
                selected_topic_summary = f"Entire Document ({st.session_state.selected_file_name})"
            else:
                combined_text = [st.session_state.active_document_topics.get(topic, "") for topic in selected_topics]
                context_text = "\n\n---\n\n".join(combined_text)
                selected_topic_summary = f"{', '.join(selected_topics)} ({st.session_state.selected_file_name})"
    
    with tools_tab:
        if not st.session_state.selected_file_name:
            st.info("Please select a file from the 'Select File & Topic' tab first.")
            return

        st.subheader(f"Tools for: **{selected_topic_summary}**")
        tab1, tab2, tab3, tab4 = st.tabs(["✨ Simplify Note", "📝 Summary Generation", "🔬 Note Lens", "🖼️ View Extracted Images"])
        with tab1:
            render_simplify_note_content(context_text)
        with tab2:
            render_summary_generator_content(context_text)
        with tab3:
            render_note_lens_content()
        with tab4:
            if st.session_state.active_extracted_images:
                st.success(f"Found {len(st.session_state.active_extracted_images)} images in '{st.session_state.selected_file_name}'.")
                for i, img_bytes in enumerate(st.session_state.active_extracted_images):
                    st.image(img_bytes, caption=f"Image {i+1}", use_column_width=True)
            else:
                st.info("No images were found in the selected PDF.")

        st.markdown("---")
        st.subheader("Query Dropper")
        st.markdown(f"Ask Elley a question about: **{selected_topic_summary}**.")
        
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Your question here..."):
            if not context_text:
                st.warning("Please select a file and topic(s) before asking questions.")
                return
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Elley is thinking..."):
                    response = chat_with_elley(prompt, context_text)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

def render_learn_deck():
    """Renders the quiz generation and interaction UI with immediate feedback."""
    st.header("🧠 Learn Deck Quiz Generator")

    if not st.session_state.quiz_generated:
        st.subheader("Step 1: Choose Your Quiz Material from an Uploaded Document")
        
        context_text = ""
        st.session_state.quiz_source_was_entire_document = False
        
        if not st.session_state.get('uploaded_files'):
            st.warning("Please go to 'Elley AI Chat' to upload PDF document(s) first.")
            return

        # --- NEW FILE & TOPIC SELECTION UI ---
        selection_tab, config_tab = st.tabs(["📁 Select Material", "⚙️ Configure Quiz"])
        
        with selection_tab:
            st.markdown("#### First, select a file")
            file_names = [f.name for f in st.session_state.uploaded_files]
            selected_file_name = st.selectbox(
                "Choose a document for the quiz:",
                file_names,
                index=file_names.index(st.session_state.selected_file_name) if st.session_state.selected_file_name in file_names else 0,
                key="quiz_file_selector"
            )

            if selected_file_name != st.session_state.selected_file_name:
                st.session_state.selected_file_name = selected_file_name
                process_selected_file(selected_file_name)
            
            if st.session_state.get('active_document_topics'):
                st.markdown("#### Next, select topics")
                topic_options = ["Entire Document"] + list(st.session_state.active_document_topics.keys())
                selected_topics = st.multiselect("Choose topics for the quiz:", topic_options, key="quiz_topic_multiselect")

                if "Entire Document" in selected_topics or not selected_topics:
                    context_text = st.session_state.active_pdf_text
                    st.session_state.quiz_source_was_entire_document = True
                else:
                    context_text = "\n\n---\n\n".join([st.session_state.active_document_topics.get(t, "") for t in selected_topics])
            else:
                st.info("No specific topics were extracted. The quiz will be based on the entire document.")
                context_text = st.session_state.active_pdf_text
                st.session_state.quiz_source_was_entire_document = True
            
            st.session_state.quiz_context = context_text

        with config_tab:
            st.subheader("Step 2: Set Quiz Options")
            if not st.session_state.quiz_context:
                st.warning("Please select a file and topics in the 'Select Material' tab.")
                return

            col1, col2 = st.columns(2)
            with col1: num_questions_combined = st.slider("Number of Questions:", min_value=3, max_value=15, value=5, key="combined_quiz_num")
            with col2: quiz_type_combined = st.selectbox("Type of Questions:", ("Multiple Choice", "Flashcards"), key="combined_quiz_type")

            if st.button("🚀 Generate Quiz", disabled=(not context_text), type="primary", use_container_width=True):
                start_quiz(context_text, num_questions_combined, quiz_type_combined)

    elif not st.session_state.show_results:
        q_index = st.session_state.current_question_index
        st.subheader(f"Question {q_index + 1}/{len(st.session_state.quiz_questions)}")
        if st.button("⬅️ Back to Configuration"):
            st.session_state.quiz_generated = False
            st.rerun()
        st.markdown("---")

        question_data = st.session_state.quiz_questions[q_index]
        
        if question_data['type'] == 'flashcard':
            question = question_data.get('question', 'No question found')
            answer = question_data.get('answer', 'No answer found')
            card_content = f'<div class="flashcard-container"><div class="flashcard"><p class="question-text">{question}</p>'
            if st.session_state.card_flipped: card_content += f'<p class="answer-text">{answer}</p>'
            card_content += "</div></div>"
            st.markdown(card_content, unsafe_allow_html=True)
            _, col2, _ = st.columns(3)
            with col2:
                if st.button("Flip Card" if not st.session_state.card_flipped else "Hide Answer", use_container_width=True):
                    st.session_state.card_flipped = not st.session_state.card_flipped
                    st.rerun()
        else: # Multiple Choice
            st.markdown(f"**{question_data['question']}**")
            st.markdown("---")
            answer_submitted = st.session_state.user_answers[q_index] is not None

            if not answer_submitted:
                for option in question_data['options']:
                    if st.button(option, key=f"{option}_{q_index}"):
                        st.session_state.user_answers[q_index] = option
                        st.rerun()
            else:
                user_choice = st.session_state.user_answers[q_index]
                correct_answer = question_data['answer']
                for option in question_data['options']:
                    if option == correct_answer: st.markdown(get_styled_option("correct", option), unsafe_allow_html=True)
                    elif option == user_choice: st.markdown(get_styled_option("incorrect", option), unsafe_allow_html=True)
                    else: st.markdown(get_styled_option("neutral", option), unsafe_allow_html=True)
                
                st.markdown("---")
                with st.container(border=True):
                    with st.spinner("Generating explanation..."):
                        brief_explanation = get_brief_explanation(question=question_data['question'], correct_answer=correct_answer, context=st.session_state.quiz_context)
                        st.markdown(brief_explanation)
                    
                    st.markdown('<div class="explain-further-button">', unsafe_allow_html=True)
                    if st.button("✨ Explain this further", key=f"explain_{q_index}"):
                        st.session_state.show_detailed_explanation[q_index] = not st.session_state.show_detailed_explanation[q_index]
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

                    if st.session_state.show_detailed_explanation[q_index]:
                        with st.spinner("Generating detailed explanation..."):
                            detailed_explanation = get_concept_explanation(question=question_data['question'], user_answer=user_choice, correct_answer=correct_answer, context=st.session_state.quiz_context)
                            st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
                            st.markdown(detailed_explanation)
                            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        is_last_question = q_index == len(st.session_state.quiz_questions) - 1
        _, col2, _ = st.columns(3)
        with col2:
            if not is_last_question:
                if st.button("Next Question ➡️", use_container_width=True):
                    st.session_state.current_question_index += 1
                    st.session_state.card_flipped = False
                    st.rerun()
            elif question_data['type'] != 'flashcard' or st.session_state.card_flipped:
                if st.button("✅ Finish & See Full Report", type="primary", use_container_width=True):
                    st.session_state.show_results = True
                    st.rerun()
    else: # Show results page
        st.subheader("📊 Final Quiz Report")
        
        current_score = 0
        current_wrong_questions = []
        current_mcq_count = 0
        for i, q in enumerate(st.session_state.quiz_questions):
            if q['type'] == 'multiple_choice':
                current_mcq_count += 1
                user_ans = st.session_state.user_answers[i]
                if user_ans and q.get('answer') and user_ans.strip() == q['answer'].strip():
                    current_score += 1
                else:
                    current_wrong_questions.append(q)
        
        st.session_state.total_score += current_score
        st.session_state.total_questions_mcq += current_mcq_count
        st.session_state.all_wrong_questions.extend(current_wrong_questions)

        if st.session_state.total_questions_mcq > 0:
            percentage = (st.session_state.total_score / st.session_state.total_questions_mcq) * 100
            st.header(f"Total Score: {st.session_state.total_score}/{st.session_state.total_questions_mcq} ({percentage:.2f}%)")
        else:
            st.header("Quiz complete! No multiple-choice questions were included in this round.")

        if st.session_state.all_wrong_questions:
            st.header("📉 Weak Area Analysis")
            if not st.session_state.weak_area_analysis:
                with st.spinner("AI is analyzing your weak areas..."):
                    full_context = st.session_state.get('active_pdf_text', st.session_state.quiz_context)
                    analysis = analyze_weak_areas(st.session_state.all_wrong_questions, full_context)
                    st.session_state.weak_area_analysis = analysis
            st.markdown(st.session_state.weak_area_analysis)
        elif st.session_state.total_questions_mcq > 0:
            st.success("🎉 Great job! You answered all questions correctly.")
        
        if st.session_state.quiz_source_was_entire_document and st.session_state.get('active_document_topics'):
            if st.session_state.uncovered_topics is None:
                with st.spinner("Analyzing quiz coverage..."):
                    all_topic_titles = list(st.session_state.active_document_topics.keys())
                    st.session_state.uncovered_topics = find_uncovered_topics(all_topic_titles, st.session_state.quiz_questions, st.session_state.active_pdf_text)
            
            if st.session_state.uncovered_topics:
                st.markdown("---")
                st.warning("Your quiz may not have covered all topics from the document.")
                with st.expander("View Uncovered Topics and Generate a Follow-up Quiz"):
                    st.markdown("The following topics might have been missed:")
                    for topic in st.session_state.uncovered_topics: st.write(f"- {topic}")
                    
                    st.markdown("**Generate a new quiz focusing on these topics?**")
                    num_q_uncovered = st.slider("Number of Questions:", min_value=1, max_value=10, value=3, key="uncovered_q_num")

                    if st.button("Generate Follow-up Quiz", type="primary"):
                        uncovered_context = "\n\n---\n\n".join([st.session_state.active_document_topics.get(t, "") for t in st.session_state.uncovered_topics])
                        st.session_state.quiz_context = uncovered_context
                        start_quiz(uncovered_context, num_q_uncovered, "Multiple Choice", is_follow_up=True)

        _, col2, _ = st.columns(3)
        with col2:
            if st.button("🔁 Take a New Quiz", type="primary", use_container_width=True):
                st.session_state.quiz_generated = False
                st.session_state.show_results = False
                st.session_state.weak_area_analysis = ""
                st.session_state.uncovered_topics = None
                st.session_state.total_score = 0
                st.session_state.total_questions_mcq = 0
                st.session_state.all_wrong_questions = []
                st.rerun()

# --- MAIN UI ROUTER ---
def setup_ui(app_mode):
    """Renders the main UI based on the selected app mode."""
    load_css("style.css")
    if app_mode == "Elley AI Chat":
        render_elley_ai_chat()
    elif app_mode == "Learn Deck":
        render_learn_deck()