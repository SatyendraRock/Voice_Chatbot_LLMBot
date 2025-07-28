import streamlit as st
import requests
import os
import tempfile

# Set Hugging Face token securely via environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("Hugging Face token not found. Please set HF_TOKEN as an environment variable.")
    st.stop()

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Function to transcribe audio
def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        data = f.read()
    
    response = requests.post(API_URL, headers=headers, data=data)

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return {"error": "Invalid response from Hugging Face API."}

# Streamlit UI
st.title("🎙️ Voice to Text - Whisper API")
st.markdown("Upload an audio file (wav/mp3/m4a)")

uploaded_file = st.file_uploader("Drag and drop file here", type=["wav", "mp3", "m4a"])

if uploaded_file:
    file_bytes = uploaded_file.read()

    with st.spinner("Transcribing audio..."):
        result = transcribe_audio(file_bytes)

    if "text" in result:
        st.success("Transcription complete!")
        st.write(result["text"])
    elif "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.error("An unknown error occurred.")

