import os
import base64
import pickle
import webbrowser
from email.mime.text import MIMEText

from groq import Groq
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import streamlit as st

# ----------------- CONFIG -----------------
GROQ_API_KEY = ""  # Replace with your Groq API key
GROQ_MODEL = "llama-3.3-70b-versatile"    # Updated Groq model
SCOPES = ['https://www.googleapis.com/auth/gmail.compose']

# ----------------- GROQ LLM -----------------
client = Groq(api_key=GROQ_API_KEY)

def analyze_sentiment(text):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            {"role": "user", "content": f"Analyze the sentiment of this text and say if it's Positive, Negative, or Neutral: {text}"}
        ],
        model=GROQ_MODEL
    )
    return response.choices[0].message.content.strip()

def draft_email(text, sentiment):
    prompt = f"""
    The client said: "{text}"
    The sentiment is: {sentiment}


    Write a professional, polite follow-up sales email (only the email body).
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL
    )
    return response.choices[0].message.content.strip()

# ----------------- GMAIL -----------------
def gmail_authenticate():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def create_draft(service, to, subject, message_text):
    message = MIMEText(message_text)
    message['to'] = to
    message['subject'] = subject
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    draft = {'message': {'raw': raw_message}}
    draft = service.users().drafts().create(userId='me', body=draft).execute()
    print(f"✅ Draft created with ID: {draft['id']}")
    return draft

# ----------------- STREAMLIT UI -----------------
st.title('Email Drafting with Sentiment Analysis')
client_text = st.text_area('Enter client text:')
recipient = st.text_input('Enter recipient email:')
subject = st.text_input('Enter email subject:')

if st.button('Generate and Save Draft'):
    with st.spinner('Analyzing sentiment and drafting email...'):
        sentiment = analyze_sentiment(client_text)
        st.write(f"Sentiment: {sentiment}")
        email_body = draft_email(client_text, sentiment)
        st.write('Drafted Email:')
        st.write(email_body)

        service = gmail_authenticate()
        draft = create_draft(service, recipient, subject, email_body)
        st.success('Draft created successfully! Redirecting to Gmail drafts...')
        # Open Gmail drafts in the browser
        webbrowser.open('https://mail.google.com/mail/u/prisoriaintoworld@gmail.com/#drafts')

