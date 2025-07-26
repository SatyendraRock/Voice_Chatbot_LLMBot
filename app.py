import streamlit as st
import requests
import os

# Hugging Face Whisper API URL
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large"
HF_TOKEN = st.secrets["HF_TOKEN"]  # Securely fetch token from secrets

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def transcribe_audio(audio_bytes):
    response = requests.post(API_URL, headers=headers, data=audio_bytes)

    if response.status_code != 200:
        return {"error": f"API Error: {response.status_code} - {response.text}"}

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return {"error": "Failed to decode JSON. Response: " + response.text}


def main():
    st.set_page_config(page_title="Audio Transcriber", page_icon="🎙️")
    st.title("🎙️ Audio Transcriber using Whisper API")
    st.write("Upload an audio file and get the transcription using OpenAI's Whisper model hosted on Hugging Face.")

    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')

        with st.spinner("Transcribing..."):
            audio_bytes = audio_file.read()
            result = transcribe_audio(audio_bytes)

            if "error" in result:
                st.error(result["error"])
            else:
                st.success("✅ Transcription complete!")
                st.markdown("**Transcription Output:**")
                st.write(result.get("text", "No text returned."))


if __name__ == "__main__":
    main()

