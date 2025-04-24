SATYA - Local AI Voice Assistant: Offline Conversational Agent with Transcription, LLM, TTS, and Live Interrupt
Overview
This project is a fully-local, privacy-first AI assistant that allows seamless spoken interaction with a voice agent. The system transcribes microphone input, processes it with an LLM to generate context-aware responses (using memory and intent classification), and speaks them out using a text-to-speech (TTS) engine. It also supports live user interruption detection and logs all conversations for memory-based interaction.
Key Features
🎙️ Speech-to-Text: Real-time transcription using OpenAI Whisper (small model, locally run).


🧠 Intent Classification: Dynamically classifies user input (Accuracy, Logic, Conversation, Creativity) to choose appropriate LLM temperature.


📜 Memory-Prompting: Injects last 4 dialogue turns into prompt context to maintain conversational history.


💬 LLM Integration: Sends prompt to a locally or remotely hosted LLM (Gemma or other open models).


📢 TTS Playback: Converts assistant responses to speech with TTS models.


🎧 Live Playback Interruption: Detects if the user starts speaking during TTS playback and halts response.


🗃️ Logging and Replay: Saves timestamped transcriptions and responses in a CSV for auditing, improvement, or long-term memory.



Installation
1. Clone the Repo
git clone https://github.com/Pratyaksha35/Satya.git
cd Satya

2. Install Python Dependencies
pip install -r requirements.txt

Dependencies include:
openai-whisper


torch


sounddevice


numpy


webrtcvad


pandas


requests


soundfile


simpleaudio


TTS


Ensure you have FFmpeg installed as well.

Usage
Run the assistant:
python assistant.py

The assistant runs a continuous loop:
Waits for user voice input


Transcribes and logs it


Classifies intent


Builds a memory-augmented prompt


Sends it to the LLM


Converts response to speech


Plays the speech back


Monitors mic for interruptions



Components Breakdown
📍 record_audio()
Records user voice using sounddevice.


Uses WebRTC VAD to detect active speech.


Stops after 2s of silence.


🧹 filter_audio_by_rms()
Removes excessive silent frames based on RMS values.


✍️ transcribe_audio()
Chunks recordings and transcribes via Whisper.


Collects segments until two consecutive blanks or timeout.


🧠 generate_assistant_response_with_memory()
Builds a prompt including the last 4 turns from CSV.


Sends it to the LLM endpoint.


Temperature dynamically chosen based on classified intent.


🔊 run_tts_playback()
Converts assistant response to .wav and plays it.


Runs a mic monitoring thread to allow interruption.



CSV Schema
The assistant saves interactions to:
~/Desktop/transcriptions.csv

With columns:
Timestamp
Transcription
Played
Intent
AssistantResponse


Customization
LLM Backend
Update the endpoint in generate_assistant_response_with_memory() to point to your local or remote LLM:
url = "http://localhost:1234/v1/chat/completions"

TTS Model
Swap TTS model in initialization:
TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

Microphone Device
Change the mic input device if needed:
sd.default.device = 0  # Or use `sounddevice.query_devices()` to pick


Roadmap
Add Vector Store Retrieval (make it a true RAG pipeline)


GUI with Streamlit or Tkinter


Multi-user speaker diarization


Multi-language transcription + response


File-based input/output



License
MIT License. Use and modify freely with attribution.

Author
Built by Pratyaksha. Contributions and feedback welcome!

