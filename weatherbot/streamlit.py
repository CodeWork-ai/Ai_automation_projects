import streamlit as st
from workflow import WeatherAgent

# Use caching to load the agent only once
@st.cache_resource
def load_weather_agent():
    print("Initializing Weather Agent...")
    return WeatherAgent()

# --- App Setup ---
st.set_page_config(page_title="Weather Chatbot", layout="centered")
st.title("🌦️ Weather Assistant")

# Load the agent
agent = load_weather_agent()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me about the weather in any city."}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is the weather in...?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get the response from your weather agent
            response, _, _ = agent.run(prompt)
            # Streamlit automatically renders Markdown!
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})