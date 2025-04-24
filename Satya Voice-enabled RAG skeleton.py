import os                                  # For file path operations
import csv                                 # For CSV reading/writing
import io                                  # For in-memory binary streams
import whisper                             # For transcription using the Whisper model
import sounddevice as sd                   # For audio recording from the microphone
import numpy as np                         # For numerical operations (arrays)
import time                                # For timekeeping and delays
import webrtcvad                           # For Voice Activity Detection (VAD)
import re                                  # For regular expression text cleaning
import torch
import pandas as pd                        # For DataFrame operations (CSV processing)
import requests                            # For HTTP requests (LLM API calls)
from concurrent.futures import ThreadPoolExecutor, as_completed  # For concurrent processing
import simpleaudio as sa                   # For audio playback
import soundfile as sf                     # For reading audio files
import threading                           # For multithreading (monitoring mic)
from TTS.api import TTS                    # For text-to-speech conversion
import queue                               # For thread-safe task queue

########################################
# Global Settings and Utility Functions
########################################

# Define the CSV file path on the Desktop
csv_path = os.path.join(os.path.expanduser("~"), "Desktop", "transcriptions.csv")

# Create a lock for thread-safe CSV writes
csv_lock = threading.Lock()

# (The transcription_queue and background worker below remain for legacy or future use.)
transcription_queue = queue.Queue()

# Initialize the Whisper model for transcription
whisper_model = whisper.load_model("small")
sd.default.device = 0  # Use the default microphone for recording

# Setup WebRTC VAD for transcription (and reuse for playback later)
vad = webrtcvad.Vad()
vad.set_mode(2)  # Set VAD sensitivity (0 = least sensitive, 3 = most sensitive)

# Audio recording settings
SAMPLE_RATE = 16000                      # Audio sample rate in Hz
FRAME_DURATION = 30                      # Frame duration in milliseconds
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # Number of samples per frame

def is_speech(frame):
    """
    Check if the given audio frame contains speech using VAD.
    """
    return vad.is_speech(frame.tobytes(), SAMPLE_RATE)

def record_audio(timeout=None):
    """
    Record audio from the microphone until a period of silence is detected.
    If a timeout is provided and no speech is detected within that time (before recording starts),
    returns None.
    """
    print("🎤 Transcription Phase: Listening...")
    buffer = []              # To store recorded frames
    silence_count = 0        # Counter for consecutive silent frames
    recording = False        # Flag to indicate if recording has started
    start_time = time.time() if timeout is not None else None
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        while True:
            if timeout is not None and not recording and (time.time() - start_time > timeout):
                print("⏰ Timeout reached without speech detection.")
                return None
            frame, _ = stream.read(FRAME_SIZE)
            frame = frame[:, 0]  # Convert stereo to mono if needed
            if is_speech(frame):
                buffer.append(frame)
                silence_count = 0
                if not recording:
                    print("🎙️ Speech detected, recording...")
                    recording = True
            elif recording:
                silence_count += 1
                # For primary segment, adjust this threshold to ~67 frames for 2 seconds of silence
                if silence_count > 67:
                    print("🔇 Speech ended, processing segment...")
                    break
    if buffer:
        return np.concatenate(buffer).astype(np.float32) / 32768.0
    return None

def clean_text(text):
    """
    Clean input text by removing unsupported characters and ensuring sufficient content.
    """
    cleaned = re.sub(r'[^\w\s,.!?-]', '', text)
    if len(cleaned.split()) < 3:
        return "Please provide more details."
    return cleaned

########################################
# Utility Function for Dynamic RMS Filtering
########################################

def filter_audio_by_rms(audio, frame_size=FRAME_SIZE, ema_alpha=0.1, multiplier=0.5, allowed_silence_frames=3):
    """
    Processes the audio frame-by-frame and removes long periods of silence.
    For each frame, an exponential moving average (EMA) of the RMS values is computed.
    The dynamic threshold for each frame is set to (EMA * multiplier).

    Frames with RMS above the dynamic threshold are kept.
    For frames below the threshold, only up to allowed_silence_frames consecutive silent frames are preserved.
    """
    processed_frames = []
    silence_buffer = []
    ema = None  # Initialize EMA
    
    for i in range(0, len(audio), frame_size):
        frame = audio[i:i+frame_size]
        rms = np.sqrt(np.mean(frame**2))
        if ema is None:
            ema = rms
        else:
            ema = ema_alpha * rms + (1 - ema_alpha) * ema
        
        dynamic_threshold = ema * multiplier
        
        if rms >= dynamic_threshold:
            if silence_buffer:
                processed_frames.extend(silence_buffer[-allowed_silence_frames:])
                silence_buffer = []
            processed_frames.append(frame)
        else:
            silence_buffer.append(frame)
    
    if silence_buffer:
        processed_frames.extend(silence_buffer[-allowed_silence_frames:])
    if processed_frames:
        return np.concatenate(processed_frames)
    return np.array([], dtype=np.float32)

########################################
# (Legacy) Background Transcription Worker
########################################

def transcription_worker():
    """
    Background worker that continuously processes audio segments from the queue.
    For each segment, it performs transcription and writes the result to the CSV.
    """
    while True:
        segment = transcription_queue.get()
        if segment is None:
            break  # Sentinel to shut down
        result = whisper_model.transcribe(segment, fp16=False)
        transcription = result.get("text", "").strip()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with csv_lock:
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([timestamp, transcription, False, ""])  # New Intent column added
        print("📝 Transcription:", transcription, "\n")
        transcription_queue.task_done()

worker_thread = threading.Thread(target=transcription_worker, daemon=True)
worker_thread.start()

########################################
# Phase 1: Transcription (Modified for Concurrent Processing)
########################################

def transcribe_segment(segment):
    """Helper function to transcribe a given audio segment using the Whisper model."""
    result = whisper_model.transcribe(segment, fp16=False)
    return result.get("text", "").strip()

def transcribe_audio(max_duration=60):
    """
    Run the transcription phase with the following modifications:
      - Primary segment waits indefinitely until speech is detected and ends after 2 seconds (≈67 frames) of silence.
      - Subsequent segments wait up to 2 seconds for speech; if no speech is detected, segment collection ends.
      - If two consecutive segments yield blank transcriptions, stop collection immediately.
      - All segments are processed concurrently using a ThreadPoolExecutor.
    """
    start_time = time.time()

    # Create CSV file with header if it doesn't exist.
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestamp", "Transcription", "Played", "Intent"])

    transcription_futures = []  # List to store futures for transcription jobs.
    segment_transcriptions = [] # To maintain ordered results.
    blank_count = 0             # Counter for consecutive blank transcriptions.

    with ThreadPoolExecutor(max_workers=6) as executor:
        # Primary Segment
        print("Recording primary segment (waits indefinitely until speech)...")
        primary_audio = record_audio(timeout=None)
        if primary_audio is not None:
            filtered_primary = filter_audio_by_rms(primary_audio)
            if filtered_primary.size != 0:
                future = executor.submit(transcribe_segment, filtered_primary)
                transcription_futures.append(future)
            else:
                print("No frames passed the RMS filter for the primary segment; skipping.")
        else:
            print("No primary audio captured.")

        if transcription_futures:
            transcription = transcription_futures[-1].result(timeout=30)
            segment_transcriptions.append(transcription)
            blank_count = blank_count + 1 if transcription.strip() == "" else 0

        # Subsequent Segments
        segment_index = 1
        while True:
            print(f"Recording subsequent segment {segment_index} (timeout = 2 seconds)...")
            additional_audio = record_audio(timeout=2)
            if additional_audio is None:
                print("No additional speech detected within 2 seconds. Ending segment collection.")
                break

            filtered_additional = filter_audio_by_rms(additional_audio)
            if filtered_additional.size == 0:
                print("No frames passed the RMS filter for this segment; ending session.")
                break

            future = executor.submit(transcribe_segment, filtered_additional)
            transcription = future.result(timeout=30)
            segment_transcriptions.append(transcription)
            if transcription.strip() == "":
                blank_count += 1
                print(f"Segment {segment_index} produced a blank transcription (consecutive blank count = {blank_count}).")
            else:
                blank_count = 0

            if blank_count >= 2:
                print("Two consecutive blank segments detected; stopping further segment collection.")
                break

            segment_index += 1

    aggregated_text = " ".join(seg for seg in segment_transcriptions if seg.strip())
    if aggregated_text:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with csv_lock:
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([timestamp, aggregated_text, False, ""])
        print("📝 Aggregated Transcription:", aggregated_text, "\n")
    else:
        print("No valid transcriptions captured in this session.")

########################################
# Phase 2: LLM Response Generation (with Intent Classification and Memory)
########################################

def build_prompt_with_memory(current_transcription):
    """
    Build an LLM prompt that includes the past 4 conversation pairs
    (each consisting of the user prompt and the assistant reply)
    as context, and then appends the current user prompt.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV for memory prompt: {e}")
        df = pd.DataFrame()

    memory_records = []
    # Only consider rows where both Transcription and AssistantResponse are present.
    if not df.empty and "AssistantResponse" in df.columns:
        valid_rows = df[(df["Transcription"].notna()) & (df["AssistantResponse"].notna()) &
                        (df["Transcription"] != "") & (df["AssistantResponse"] != "")]
        last_four = valid_rows.tail(4)
        for idx, row in last_four.iterrows():
            conversation_pair = f"User: {row['Transcription']}\nAssistant: {row['AssistantResponse']}"
            memory_records.append(conversation_pair)

    prompt_parts = ["The following are the past 4 conversation pairs provided for context:"]
    for i, record in enumerate(memory_records, start=1):
        prompt_parts.append(f"Conversation {i}:\n{record}\n")
    
    prompt_parts.append("The following is the current user prompt:")
    prompt_parts.append(current_transcription)
    prompt_parts.append("\nPlease generate an assistant response that considers the above conversation history.")
    
    full_prompt = "\n".join(prompt_parts)
    
    # Debug print
    print("Built Prompt with Memory:")
    print(full_prompt)
    
    return full_prompt

def generate_assistant_response_with_memory(current_transcription, temp):
    """
    Generate an assistant response using a prompt that includes both
    the past conversation pairs (memory) and the current transcription.
    """
    combined_prompt = build_prompt_with_memory(current_transcription)
    
    messages = [
        {"role": "system", "content": (
            "You are Satya, a personal assistant. Provide a concise, helpful, and friendly response. "
            "Do not include any emojis or special symbols in your output."
        )},
        {"role": "user", "content": combined_prompt}
    ]
    payload = {
        "model": "gemma-3-1b-it@q3_k_l",
        "messages": messages,
        "temperature": temp,
        "max_tokens": 500,
        "stream": False
    }
    try:
        response = requests.post("https://d2f5-103-228-40-107.ngrok-free.app/v1/chat/completions",
                                 headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error generating assistant response with memory: {e}")
        return "Error generating response."

def classify_intent(transcription):
    """
    Classify the intent of the current transcription using only that transcription.
    """
    messages = [
        {"role": "system", "content": (
            "You are an expert in categorizing text. Based on the following transcription, "
            "classify it into one of these categories:\n"
            " - Accuracy (for content requiring precise details, typically generated at 0.1–0.3 temperature),\n"
            " - Logic (for analytical reasoning, 0.3–0.6),\n"
            " - Conversation (for informal dialogue, 0.5–0.9), or\n"
            " - Creativity (for imaginative, free-form text, 0.7–1.2).\n"
            "Respond with only the category name."
        )},
        {"role": "user", "content": transcription}
    ]
    payload = {
        "model": "gemma-3-1b-it@q3_k_l",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 20,
        "stream": False
    }
    try:
        response = requests.post("https://d2f5-103-228-40-107.ngrok-free.app/v1/chat/completions",
                                 headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        data = response.json()
        classification = data["choices"][0]["message"]["content"].strip()
        if classification not in ["Accuracy", "Logic", "Conversation", "Creativity"]:
            print(f"Unexpected classification '{classification}', defaulting to 'Conversation'.")
            return "Conversation"
        return classification
    except Exception as e:
        print(f"Error classifying transcription: {e}. Defaulting to 'Conversation'.")
        return "Conversation"

def process_transcription(transcription):
    """
    Process the current transcription by:
      1. Classifying its intent (using only the current transcription).
      2. Generating an assistant response using the combined prompt (memory + current transcription).
    """
    intent = classify_intent(transcription)
    if intent == "Accuracy":
        dynamic_temp = 0.2
    elif intent == "Logic":
        dynamic_temp = 0.45
    elif intent == "Conversation":
        dynamic_temp = 0.7
    elif intent == "Creativity":
        dynamic_temp = 0.95
    else:
        dynamic_temp = 0.7
    
    assistant_response = generate_assistant_response_with_memory(transcription, dynamic_temp)
    return intent, assistant_response

def run_llm_response_generation():
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Transcription"])
    if "AssistantResponse" in df.columns:
        df_to_process = df[df["AssistantResponse"].isnull() | (df["AssistantResponse"] == "")]
    else:
        df_to_process = df

    if "Intent" not in df.columns:
        df["Intent"] = ""
    else:
        df["Intent"] = df["Intent"].astype(str)

    def process_row(transcription):
        return process_transcription(transcription)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_row, df_to_process["Transcription"]))

    for idx, (intent, assistant_response) in zip(df_to_process.index, results):
        df.at[idx, "Intent"] = str(intent)
        df.at[idx, "AssistantResponse"] = assistant_response

    df.to_csv(csv_path, index=False)
    print("LLM Response Generation: Completed, CSV updated with AssistantResponse and Intent.")

########################################
# Phase 3: TTS Playback with Concurrent Processing
########################################

# Initialize the TTS model and related settings.
device = "mps" if torch.backends.mps.is_available() else "cpu"
tts_playback = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to(device)
print(f"TTS is using: {device}")
vad_playback = webrtcvad.Vad()
vad_playback.set_mode(2)
global_stop = False
monitor_stop_event = threading.Event()

# A thread-safe queue to store pre-processed audio buffers.
audio_queue = queue.Queue()

def process_tts(text):
    """
    Convert text to an audio buffer using TTS.
    Returns a tuple (text, audio_buffer).
    """
    audio_buffer = io.BytesIO()
    tts_playback.tts_to_file(text=text, file_path=audio_buffer)
    audio_buffer.seek(0)
    return text, audio_buffer

def run_tts_playback():
    global global_stop
    monitor_stop_event.clear()
    
    # Start continuous monitoring for interruption.
    continuous_monitor_thread = threading.Thread(target=monitor_continuous)
    continuous_monitor_thread.start()
    
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["AssistantResponse"])
    if "Played" not in df.columns:
        df["Played"] = False

    # Use a thread pool to process TTS concurrently.
    with ThreadPoolExecutor(max_workers=2) as tts_executor:
        futures = {}
        for idx, row in df.iterrows():
            if not row["Played"]:
                text = clean_text(str(row["AssistantResponse"]).strip())
                if text:
                    future = tts_executor.submit(process_tts, text)
                    futures[future] = idx
        
        # As futures complete (with a timeout), place their results in the queue.
        try:
            for future in as_completed(futures, timeout=30):
                idx = futures[future]
                try:
                    text, audio_buffer = future.result(timeout=30)
                    audio_queue.put((idx, text, audio_buffer))
                except Exception as e:
                    print(f"Error processing TTS for row {idx}: {e}")
        except Exception as e:
            print(f"Timeout waiting for TTS futures: {e}")
    
    # Now, play back audio buffers from the queue.
    while not audio_queue.empty() and not global_stop:
        idx, text, audio_buffer = audio_queue.get()
        print(f"Speaking for row {idx}: {text}")
        try:
            data, samplerate = sf.read(audio_buffer, dtype='int16')
            audio_array = np.array(data, dtype=np.int16).tobytes()
            wave_obj = sa.WaveObject(audio_array, num_channels=1, bytes_per_sample=2, sample_rate=samplerate)
            play_obj = wave_obj.play()
            
            # Start a monitoring thread for the current playback.
            monitor_thread = threading.Thread(target=monitor_mic, args=(play_obj,))
            monitor_thread.start()
            play_obj.wait_done()
            monitor_thread.join()
            
            if global_stop:
                print("User speech detected during playback; marking remaining rows as played.")
                df.loc[idx:, "Played"] = True
                df.to_csv(csv_path, index=False)
                break
            
            df.at[idx, "Played"] = True
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error during playback for row {idx}: {e}")
    
    monitor_stop_event.set()
    continuous_monitor_thread.join()

def monitor_continuous(amp_threshold=2400, required_consecutive=5):
    """
    Continuously monitor the microphone for user speech.
    If detected, set the global_stop flag to interrupt playback.
    """
    global global_stop
    speech_counter = 0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        while not monitor_stop_event.is_set():
            frame, _ = stream.read(FRAME_SIZE)
            mic_frame = frame[:, 0]
            amplitude = np.abs(mic_frame).mean()
            if amplitude > amp_threshold and vad_playback.is_speech(mic_frame.tobytes(), SAMPLE_RATE):
                speech_counter += 1
                if speech_counter >= required_consecutive:
                    print("Continuous user speech detected. Interrupting TTS phase.")
                    global_stop = True
                    monitor_stop_event.set()
                    break
            else:
                speech_counter = 0
            time.sleep(0.01)

def monitor_mic(play_obj, amp_threshold=2400, required_consecutive=5):
    """
    Monitor the microphone during audio playback.
    If user speech is detected, stop the playback.
    """
    global global_stop
    speech_counter = 0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        while play_obj.is_playing():
            frame, _ = stream.read(FRAME_SIZE)
            mic_frame = frame[:, 0]
            amplitude = np.abs(mic_frame).mean()
            if amplitude > amp_threshold and vad_playback.is_speech(mic_frame.tobytes(), SAMPLE_RATE):
                speech_counter += 1
                if speech_counter >= required_consecutive:
                    print("User speech detected during clip playback. Stopping playback.")
                    play_obj.stop()
                    global_stop = True
                    break
            else:
                speech_counter = 0
            time.sleep(0.01)

########################################
# Main Execution Pipeline (Continuous Loop)
########################################

if __name__ == '__main__':
    try:
        while True:
            print("\n--- Starting Transcription Phase ---")
            transcribe_audio(max_duration=60)
            
            print("\n--- Running LLM Response Generation Phase ---")
            run_llm_response_generation()
            
            global_stop = False
            print("\n--- Running TTS Playback Phase ---")
            run_tts_playback()
            
            print("\n--- Cycle complete. Restarting ---\n")
    except KeyboardInterrupt:
        print("\n🛑 Transcription interrupted by user.")
        transcription_queue.put(None)
        worker_thread.join()
