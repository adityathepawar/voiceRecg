import streamlit as st
import numpy as np
import wave
import speech_recognition as sr
import time
import os
import requests
import re
import contextlib
import editdistance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
FILENAME = "audio.wav"
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
MODEL = "deepseek/deepseek-chat:free"

# Session State Initialization
for key in ["transcription", "corrected_text", "similarity_score"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# ---------------------- Utility Functions ----------------------

def preprocess(text):
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def compute_confidence_cosine(transcribed, expected):
    transcribed = clean_text(transcribed)
    expected = clean_text(expected)
    vectorizer = TfidfVectorizer().fit_transform([transcribed, expected])
    vectors = vectorizer.toarray()
    if len(vectors) < 2:
        return 0.0
    return round(cosine_similarity([vectors[0]], [vectors[1]])[0][0] * 100, 2)

def compute_confidence_edit_distance(transcribed, expected):
    t_words = clean_text(transcribed).split()
    e_words = clean_text(expected).split()
    dist = editdistance.eval(t_words, e_words)
    max_len = max(len(t_words), len(e_words))
    if max_len == 0:
        return 100.0
    return round((1 - dist / max_len) * 100, 2)

def heuristic_confidence(text):
    words = clean_text(text).split()
    score = 100
    if len(words) < 5:
        score -= 30
    fillers = ["um", "uh", "like", "you know"]
    filler_count = sum(text.lower().count(f) for f in fillers)
    score -= filler_count * 5
    return max(0, min(100, round(score, 2)))

def speech_rate_penalty(transcribed, duration_seconds):
    word_count = len(transcribed.split())
    if duration_seconds == 0:
        return 0
    words_per_minute = (word_count / duration_seconds) * 60
    if words_per_minute < 80 or words_per_minute > 180:
        return 70
    return 100

def punctuation_score(corrected):
    punctuations = ['.', ',', '?', '!']
    count = sum(corrected.count(p) for p in punctuations)
    word_count = len(corrected.split())
    if word_count == 0:
        return 0
    punct_density = count / word_count
    return round(min(100, punct_density * 300), 2)

def filler_word_score(text):
    fillers = ["um", "uh", "like", "you know", "basically", "so"]
    words = text.lower().split()
    count = sum(words.count(f) for f in fillers)
    density = count / len(words) if words else 0
    return max(0, 100 - density * 100)

def get_audio_duration(path=FILENAME):
    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def final_confidence(transcribed, expected, corrected, duration_sec):
    c = compute_confidence_cosine(transcribed, expected)
    e = compute_confidence_edit_distance(transcribed, expected)
    h = heuristic_confidence(transcribed)
    f = filler_word_score(transcribed)
    p = punctuation_score(corrected)
    s = speech_rate_penalty(transcribed, duration_sec)
    return round((c + e + h + f + p + s) / 6, 2)

# ---------------------- Streamlit UI ----------------------

st.title("🎤 Voice Monitor: Upload → Transcribe → Correct → Analyze")

uploaded_file = st.file_uploader("Upload a .wav audio file", type="wav")

if uploaded_file:
    with open(FILENAME, "wb") as f:
        f.write(uploaded_file.read())
    st.success("Audio uploaded and saved as 'audio.wav'")

    if st.button("📝 Transcribe Audio"):
        recognizer = sr.Recognizer()
        with sr.AudioFile(FILENAME) as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            st.session_state.transcription = text
            st.success("📝 Transcription:")
            st.write(text)
        except sr.UnknownValueError:
            st.error("Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Speech Recognition error: {e}")

if st.session_state.transcription:
    st.subheader("🌠 Original Transcribed Text")
    st.text_area("Transcription", st.session_state.transcription, height=150)

    if st.button("🤖 Correct Grammar (via DeepSeek)"):
        with st.spinner("Sending transcription to DeepSeek LLM..."):
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Correct the grammar in the following text:\n\n\"{st.session_state.transcription}\"\n\nReturn only the corrected version."
                    }
                ]
            }
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, verify=False)
            if response.status_code == 200:
                corrected = response.json()['choices'][0]['message']['content']
                st.session_state.corrected_text = corrected
                st.success("✅ Corrected Grammar Text")
                st.text_area("Corrected Output", corrected, height=150)

                similarity_score = compute_confidence_cosine(st.session_state.transcription, corrected)
                duration_sec = get_audio_duration()
                final_score = final_confidence(st.session_state.transcription, st.session_state.transcription, corrected, duration_sec)

                st.subheader("📊 Transcription Confidence Metrics")
                st.metric(label="Cosine Similarity Score", value=f"{similarity_score}%")
                st.metric(label="Overall Confidence Score", value=f"{final_score}%")
            else:
                st.error(f"LLM API Error {response.status_code}: {response.text}")

st.markdown("---")
st.subheader("📈 Chunked Monitoring of Saved File (audio.wav)")

if st.button("📂 Process Saved Audio in Chunks"):
    if not os.path.exists(FILENAME):
        st.error("❌ audio.wav not found!")
    else:
        recognizer = sr.Recognizer()
        audio_file = sr.AudioFile(FILENAME)
        confidence_data = []
        with audio_file as source:
            duration = source.DURATION
            st.info(f"🔎 Processing saved audio ({duration:.2f}s)...")
            offset = 0.0
            while offset < duration:
                try:
                    audio_chunk = recognizer.record(source, duration=10)
                    transcribed = recognizer.recognize_google(audio_chunk)
                    corrected = corrected = corrected = response.json()['choices'][0]['message']['content']
                    confidence = final_confidence(transcribed, corrected, corrected, 10)
                    confidence_data.append((round(offset, 2), confidence))
                    st.write(f"🕐 [{offset:.2f}s - {offset + 10:.2f}s]")
                    st.markdown(f"**🎤 Transcribed:** {transcribed}")
                    st.markdown(f"**🪄 Corrected:** {corrected}")
                    st.markdown(f"**📊 Confidence:** `{confidence}%`")
                    st.markdown("---")
                except Exception as e:
                    st.warning(f"Error at {offset:.2f}s: {e}")
                    confidence_data.append((round(offset, 2), 0.0))
                offset += 10
        if confidence_data:
            _, scores = zip(*confidence_data)
            st.line_chart({"Confidence Score (%)": scores})
