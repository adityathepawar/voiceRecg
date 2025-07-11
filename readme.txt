VOICE RECOGNITION AND GRAMMAR ANALYSIS SYSTEM

ABOUT THE PROJECT:
VoiceRecg is a real-time speech monitoring and grammar analysis system developed using Python and Streamlit. The application records user audio, transcribes it using Google's SpeechRecognition API, analyzes grammatical correctness using an OpenRouter-hosted LLM (e.g., DeepSeek Chat), and evaluates speaking accuracy via cosine similarity. It also plots confidence scores over time to help users track improvement.

SETUP INSTRUCTIONS:
1. Ensure Python 3.8 or above is installed on your system.
2. Install required packages using `pip install -r requirements.txt`. This includes `streamlit`, `speechrecognition`, `openai`, `matplotlib`, `scikit-learn`, and other dependencies.
3. Obtain API access from OpenRouter and set your API key in a `.env` file or directly inside the code.
4. Run the app with `streamlit run app.py`.

FEATURES:
- Record and timestamp live audio from microphone
- Transcribe audio to text
- Analyze and correct grammar using AI
- Compute accuracy using cosine similarity
- Plot confidence graph over time
- Export results and logs if needed

USAGE:
Simply launch the app, speak into the microphone, and view real-time feedback. Ideal for improving spoken English and grammar accuracy in voice.

