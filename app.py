import streamlit as st
import whisper
from transformers import pipeline
import torch
from TTS.api import TTS
import soundfile as sf
import os
import tempfile

st.set_page_config(page_title="Voice Chatbot", page_icon="🎤")
st.title("🎤 Voice Chatbot (Speech → Text → LLM → TTS)")

# Load ASR
@st.cache_resource
def load_asr_model():
    return whisper.load_model("base")

# Load LLM
@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Load TTS
@st.cache_resource
def load_tts():
    return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())

asr_model = load_asr_model()
chatbot = load_llm()
tts = load_tts()

# Rule-based fallback
def rule_based_logic(text):
    text = text.lower()
    if "name" in text:
        return "Can you please tell me your full name?"
    elif "age" in text:
        return "Got it. How old are you currently?"
    elif "done" in text:
        return "Thanks for your time! We'll process your responses."
    return None

# Upload audio or record
st.markdown("### 📥 Upload Your Voice (WAV format)")
audio_file = st.file_uploader("Upload a `.wav` file", type=["wav"])

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_input:
        tmp_input.write(audio_file.read())
        tmp_input_path = tmp_input.name

    st.audio(tmp_input_path, format="audio/wav")

    with st.spinner("🔍 Transcribing..."):
        result = asr_model.transcribe(tmp_input_path)
        user_text = result["text"]

    st.markdown(f"**📝 Transcription:** `{user_text}`")

    # Get response
    response_text = rule_based_logic(user_text)
    if not response_text:
        with st.spinner("💬 Generating Response..."):
            result = chatbot(user_text, max_length=100, num_return_sequences=1)
            response_text = result[0]["generated_text"]

    st.markdown(f"**🤖 Chatbot:** {response_text}")

    # Text-to-Speech
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
        tts.tts_to_file(text=response_text, file_path=tmp_out.name)
        st.audio(tmp_out.name, format="audio/wav")

    # Clean up
    os.remove(tmp_input_path)
else:
    st.info("Please upload a WAV file recorded from your voice.")
