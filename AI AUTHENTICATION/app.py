# app.py

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import speech_recognition as sr
import json
import os
from signature_auth import extract_signature_features, compare_signature_features

USER_FILE = "users_db.json"

# ---------- DATABASE FUNCTIONS (No Changes) ----------
def load_db():
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, "r") as f: return json.load(f)
        except Exception:
            st.warning("DB corrupted. New DB created.")
            with open(USER_FILE, "w") as f: f.write("{}")
            return {}
    return {}

def save_db(db):
    with open(USER_FILE, "w") as f: json.dump(db, f)

def register_user(username, name, email, phrase, voice_features, signature_image):
    db = load_db()
    db[username] = {"name": name, "email": email, "registered_phrase": phrase,
                    "voice_features": voice_features.tolist(), "signature_image": signature_image.tolist()}
    save_db(db)

def get_user(username):
    db = load_db()
    user = db.get(username)
    if user:
        user["voice_features"] = np.array(user["voice_features"])
        user["signature_image"] = np.array(user["signature_image"])
    return user

# ---------- PROCESSING FUNCTIONS (No Changes) ----------
def transcribe_audio(audio_bytes):
    with open("temp.wav", "wb") as f: f.write(audio_bytes)
    r = sr.Recognizer()
    with sr.AudioFile("temp.wav") as source: audio = r.record(source)
    try: return r.recognize_google(audio)
    except Exception: return ""

def extract_voice_features(audio_bytes): return np.random.rand(512)

def compare_voice_features(f1, f2):
    if f1 is None or f2 is None: return 0.0
    return float(np.dot(f1 / np.linalg.norm(f1), f2 / np.linalg.norm(f2)))

def get_signature_ndarray(image_data):
    if image_data is None: return None
    return np.array(image_data).astype(np.uint8)

# ---------- UI LAYOUT ----------
st.title("AI-Powered Voice + Signature Authentication")

# +++ THIS IS THE CHANGE +++
# The previous threshold of 0.20 was too low.
# 0.50 is a much stricter and safer starting point.
AUTHENTICATION_THRESHOLD = 0.50 

mode = st.radio("Mode", ["Register User", "Authenticate User", "Tune Threshold"])

if mode == "Register User":
    # --- REGISTRATION UI (No Changes) ---
    st.subheader("User Registration")
    username = st.text_input("Username")
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    reg_phrase = st.text_input("Choose a secret phrase")
    st.subheader("Voice Registration")
    audio_bytes = audio_recorder()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        st.write(f"You said: {transcribe_audio(audio_bytes)}")
    st.subheader("Signature Registration")
    canvas_result = st_canvas(fill_color="rgba(0,0,0,0)", stroke_width=3, height=150, width=400, drawing_mode="freedraw", key="reg_canvas")
    if st.button("Register Now"):
        if not all([username.strip(), name.strip(), email.strip(), reg_phrase.strip()]):
            st.error("All fields are required.")
        elif not audio_bytes:
            st.error("Please record your voice.")
        elif canvas_result.image_data is None:
            st.error("Please draw your signature.")
        else:
            if transcribe_audio(audio_bytes).strip().lower() != reg_phrase.strip().lower():
                st.error("Spoken phrase doesn’t match your typed phrase.")
            else:
                register_user(username, name, email, reg_phrase.strip().lower(), 
                              extract_voice_features(audio_bytes), 
                              get_signature_ndarray(canvas_result.image_data))
                st.success("Registration successful!")

elif mode == "Authenticate User":
    # --- AUTHENTICATION UI ---
    st.subheader("User Authentication")
    username = st.text_input("Username")
    user = get_user(username)
    if username and user:
        st.info(f"Welcome back {user['name']}! Please verify.")
        st.subheader("Step 1: Voice Verification")
        audio_bytes = audio_recorder()
        if audio_bytes: st.audio(audio_bytes, format="audio/wav")
        if st.button("Verify Voice"):
            if not audio_bytes: st.error("Please record your voice.")
            else:
                spoken = transcribe_audio(audio_bytes)
                st.write(f"You said: {spoken}")
                if spoken.strip().lower() == user['registered_phrase'] and compare_voice_features(extract_voice_features(audio_bytes), user["voice_features"]) >= 0.75:
                    st.success("Voice verified!")
                    st.session_state.voice_passed = True
                else:
                    st.error("Voice verification failed.")
                    st.session_state.voice_passed = False
        
        if st.session_state.get("voice_passed", False):
            st.subheader("Step 2: Signature Verification")
            canvas_result = st_canvas(fill_color="rgba(0,0,0,0)", stroke_width=3, height=150, width=400, drawing_mode="freedraw", key="auth_canvas")
            if st.button("Verify Signature"):
                attempt_sig = get_signature_ndarray(canvas_result.image_data)
                if attempt_sig is None or np.sum(attempt_sig[:, :, 3]) == 0:
                    st.error("No signature provided.")
                else:
                    stored_features = extract_signature_features(user['signature_image'])
                    attempt_features = extract_signature_features(attempt_sig)
                    similarity = compare_signature_features(stored_features, attempt_features)
                    st.write(f"Signature similarity score: {similarity:.2f}")
                    if similarity > AUTHENTICATION_THRESHOLD:
                        st.success("Signature verified successfully! User authenticated.")
                    else:
                        st.error("Signature verification failed. Try again.")

elif mode == "Tune Threshold":
    # --- TUNING MODE (No Changes) ---
    st.subheader("Find the Perfect Signature Threshold")
    username = st.text_input("Enter username to load registered signature")
    user = get_user(username)
    if username and user:
        st.success(f"Loaded signature for {user['name']}.")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Draw a **GOOD** signature")
            canvas_good = st_canvas(fill_color="rgba(0,0,0,0)", stroke_width=3, height=150, drawing_mode="freedraw", key="tune_good")
        with col2:
            st.write("Draw a **BAD** signature (a fake)")
            canvas_bad = st_canvas(fill_color="rgba(0,0,0,0)", stroke_width=3, height=150, drawing_mode="freedraw", key="tune_bad")
        if st.button("Run Tuning Test"):
            good_img = get_signature_ndarray(canvas_good.image_data)
            bad_img = get_signature_ndarray(canvas_bad.image_data)
            if good_img is None or bad_img is None or np.sum(good_img[:, :, 3]) == 0 or np.sum(bad_img[:, :, 3]) == 0:
                st.error("Please draw in both boxes.")
            else:
                stored_features = extract_signature_features(user['signature_image'])
                good_features = extract_signature_features(good_img)
                bad_features = extract_signature_features(bad_img)
                score_good = compare_signature_features(stored_features, good_features)
                score_bad = compare_signature_features(stored_features, bad_features)
                st.subheader("Results:")
                st.metric("Similarity for GOOD signature", f"{score_good:.3f}")
                st.metric("Similarity for BAD signature", f"{score_bad:.3f}")
                st.info(f"Recommendation: Your threshold should be between the bad score and the good score. A good starting point would be around **{ (score_bad + (score_good - score_bad) / 2) :.2f}**.")
                st.warning("After you find a value, change the `AUTHENTICATION_THRESHOLD` variable at the top of `app.py` and restart.")