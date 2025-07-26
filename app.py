import os
import streamlit as st
import requests
import tempfile
import base64

st.set_page_config(page_title="Voice to Text with Whisper")
st.title("🎤 Voice to Text - Whisper API")

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face token not found. Please set HF_TOKEN as an environment variable.")
    st.stop()

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def transcribe_audio(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

uploaded_file = st.file_uploader("Upload an audio file (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    file_bytes = uploaded_file.read()

    with st.spinner("Transcribing audio..."):
        result = transcribe_audio(file_bytes)

    if "text" in result:
        st.success("Transcription complete!")
        st.markdown(f"**Transcript:** {result['text']}")
    else:
        st.error("Failed to transcribe. Response:")
        st.json(result)

