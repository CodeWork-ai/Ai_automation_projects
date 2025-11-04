# voice_auth.py

from audio_recorder_streamlit import audio_recorder
import streamlit as st
import numpy as np

def record_and_verify_voice(new_user=True, stored_voice=None):
    st.write("Click to record, then speak clearly into your microphone.")
    audio_bytes = audio_recorder()
    if st.button("Verify Voice"):
        if audio_bytes:
            # Placeholder: extract voice features (embedding) using pretrained AI model
            voice_features = extract_voice_features(audio_bytes)
            if new_user:
                return True, voice_features
            else:
                similarity = compare_voice_features(voice_features, stored_voice)
                return similarity > 0.85, voice_features
        else:
            st.error("No audio detected. Please click the microphone icon and record your voice.")
            return False, None
    return False, None

def extract_voice_features(audio_bytes):
    # Implement your pretrained AI model for extracting an embedding here
    # This is a placeholder stub returning dummy features
    return np.random.rand(512)

def compare_voice_features(features1, features2):
    # Cosine similarity between embeddings
    if features1 is None or features2 is None:
        return 0.0
    features1 = features1 / np.linalg.norm(features1)
    features2 = features2 / np.linalg.norm(features2)
    return float(np.dot(features1, features2))
