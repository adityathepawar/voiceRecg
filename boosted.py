import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import speech_recognition as sr
import threading
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
FS = 44100
CHANNELS = 1
OPENROUTER_API_KEY = "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"
MODEL = "deepseek/deepseek-chat:free"

# Session State Initialization
for key in ["is_recording", "record_thread", "audio_buffer", "transcription", "corrected_text", "similarity_score"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "is_recording" else []

if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()

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

# ---------------------- Audio Functions ----------------------

def record_audio(stop_event):
    st.session_state.audio_buffer = []

    def callback(indata, frames, time_info, status):
        if stop_event.is_set():
            raise sd.CallbackStop()
        st.session_state.audio_buffer.append(indata.copy())

    with sd.InputStream(samplerate=FS, channels=CHANNELS, callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)

    audio_np = np.concatenate(st.session_state.audio_buffer, axis=0)
    audio_int16 = np.int16(audio_np * 32767)

    with wave.open(FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(FS)
        wf.writeframes(audio_int16.tobytes())

    st.success(f"✅ Audio saved as `{FILENAME}`")

def transcribe_audio(path=FILENAME):
    if not os.path.exists(path):
        st.error("No audio.wav file found!")
        return

    recognizer = sr.Recognizer()
    with sr.AudioFile(path) as source:
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

def correct_grammar_with_llm(transcribed_text):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"Correct the grammar in the following text:\n\n\"{transcribed_text}\"\n\nReturn only the corrected version."
            }
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, verify=False)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        st.error(f"❌ LLM API Error {response.status_code}: {response.text}")
        return None

def calculate_similarity_score(original, corrected):
    original_clean = preprocess(original)
    corrected_clean = preprocess(corrected)
    vectorizer = TfidfVectorizer().fit_transform([original_clean, corrected_clean])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return round(similarity * 100, 2)

def process_audio_file_in_chunks(path="audio.wav", chunk_duration=10):
    if not os.path.exists(path):
        st.error("❌ audio.wav not found!")
        return []

    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(path)
    confidence_data = []

    with audio_file as source:
        duration = source.DURATION
        st.info(f"🔎 Processing saved audio ({duration:.2f}s)...")
        offset = 0.0

        while offset < duration:
            try:
                audio_chunk = recognizer.record(source, duration=chunk_duration)
                transcribed = recognizer.recognize_google(audio_chunk)
                corrected = correct_grammar_with_llm(transcribed)
                confidence = final_confidence(transcribed, corrected, corrected, chunk_duration)

                confidence_data.append((round(offset, 2), confidence))

                st.write(f"🕐 [{offset:.2f}s - {offset + chunk_duration:.2f}s]")
                st.markdown(f"**🎙️ Transcribed:** {transcribed}")
                st.markdown(f"**🪄 Corrected:** {corrected}")
                st.markdown(f"**📊 Confidence:** `{confidence}%`")
                st.markdown("---")

            except Exception as e:
                st.warning(f"Error at {offset:.2f}s: {e}")
                confidence_data.append((round(offset, 2), 0.0))

            offset += chunk_duration

    return confidence_data

# ---------------------- Streamlit UI ----------------------

st.title("🎤 Voice Monitor: Record → Transcribe → Correct → Analyze")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start Recording", disabled=st.session_state.is_recording):
        st.session_state.stop_event.clear()
        st.session_state.is_recording = True
        st.session_state.record_thread = threading.Thread(target=record_audio, args=(st.session_state.stop_event,))
        st.session_state.record_thread.start()
        st.info("🎙️ Recording... Speak now!")

with col2:
    if st.button("⏹️ Stop Recording", disabled=not st.session_state.is_recording):
        st.session_state.stop_event.set()
        st.session_state.record_thread.join()
        st.session_state.is_recording = False
        st.success("Recording stopped and saved.")

st.markdown("---")

if st.button("📝 Transcribe Audio"):
    transcribe_audio()

if st.session_state.transcription:
    st.subheader("🔤 Original Transcribed Text")
    st.text_area("Transcription", st.session_state.transcription, height=150)

    if st.button("🤖 Correct Grammar (via DeepSeek)"):
        with st.spinner("Sending transcription to DeepSeek LLM..."):
            corrected = correct_grammar_with_llm(st.session_state.transcription)

        if corrected:
            st.session_state.corrected_text = corrected
            st.success("✅ Corrected Grammar Text")
            st.text_area("Corrected Output", corrected, height=150)

            similarity_score = calculate_similarity_score(
                st.session_state.transcription,
                st.session_state.corrected_text
            )
            st.session_state.similarity_score = similarity_score

            duration_sec = get_audio_duration()
            final_score = final_confidence(
                st.session_state.transcription,
                st.session_state.transcription,
                corrected,
                duration_sec
            )

            st.subheader("📊 Transcription Confidence Metrics")
            st.metric(label="Cosine Similarity Score", value=f"{similarity_score}%")
            st.metric(label="Overall Confidence Score", value=f"{final_score}%")

st.markdown("---")
st.subheader("📈 Chunked Monitoring of Saved File (audio.wav)")

if st.button("📂 Process Saved Audio in Chunks"):
    with st.spinner("Analyzing chunks..."):
        results = process_audio_file_in_chunks(path=FILENAME, chunk_duration=10)

    if results:
        timestamps, scores = zip(*results)
        st.line_chart({"Confidence Score (%)": scores})
