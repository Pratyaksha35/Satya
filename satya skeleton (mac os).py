# Import standard and third-party modules
import os                                  # For file path operations
import csv                                 # For CSV reading/writing
import io                                  # For in-memory binary streams
import whisper                             # For transcription using the Whisper model
import sounddevice as sd                   # For audio recording from the microphone
import numpy as np                         # For numerical operations (arrays)
import time                                # For timekeeping and delays
import webrtcvad                         # For Voice Activity Detection (VAD)
import re                                  # For regular expression text cleaning
import pandas as pd                        # For DataFrame operations (CSV processing)
import requests                            # For HTTP requests (LLM API calls)
from concurrent.futures import ThreadPoolExecutor  # For concurrent processing
import simpleaudio as sa                   # For audio playback
import soundfile as sf                     # For reading audio files
import threading                           # For multithreading (monitoring mic)
from TTS.api import TTS                    # For text-to-speech conversion

########################################
# Global Settings and Utility Functions
########################################

# Define the CSV file path on the Desktop
csv_path = os.path.join(os.path.expanduser("~"), "Desktop", "transcriptions.csv")

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
    If a timeout (in seconds) is provided and no speech is detected within that time
    (before recording starts), returns None.
    
    Parameters:
        timeout (float): Maximum time to wait for speech (in seconds) if not already recording.
    
    Returns:
        np.array or None: Normalized audio data if recorded, otherwise None.
    """
    print("🎤 Transcription Phase: Listening...")
    buffer = []              # To store recorded frames
    silence_count = 0        # Counter for consecutive silent frames
    recording = False        # Flag to indicate if recording has started
    start_time = time.time() if timeout is not None else None
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        while True:
            # If a timeout is set and we're not yet recording, exit if time is exceeded
            if timeout is not None and not recording and (time.time() - start_time > timeout):
                print("⏰ Timeout reached without speech detection.")
                return None
            frame, _ = stream.read(FRAME_SIZE)  # Read a frame of audio
            frame = frame[:, 0]                 # Convert stereo to mono (if applicable)
            if is_speech(frame):
                buffer.append(frame)            # Append frame if speech detected
                silence_count = 0               # Reset silence counter
                if not recording:
                    print("🎙️ Speech detected, recording...")
                    recording = True
            elif recording:
                silence_count += 1              # Increase counter if silent
                if silence_count > 70:          # If ~2.10 sec of silence detected, stop recording
                    print("🔇 Speech ended, transcribing...")
                    break
    if buffer:
        # Concatenate all frames, convert to float32, and normalize the amplitude
        return np.concatenate(buffer).astype(np.float32) / 32768.0
    return None

def clean_text(text):
    """
    Clean input text by removing unsupported characters (e.g., emojis)
    and ensuring that there are enough words.
    """
    cleaned = re.sub(r'[^\w\s,.!?-]', '', text)  # Remove characters not in allowed set
    if len(cleaned.split()) < 3:                  # Check if there are at least 3 words
        return "Please provide more details."
    return cleaned

########################################
# Phase 1: Transcription (Modified)
########################################

def transcribe_audio(max_duration=60):
    """
    Run the transcription phase. First, wait indefinitely for a speech segment and transcribe it.
    Then, repeatedly wait for additional speech for up to 2 seconds.
    If no additional speech is detected within 2 seconds, end the transcription phase.
    
    Parameters:
        max_duration (int): Maximum overall duration (in seconds) for transcription.
    """
    start_time = time.time()  # Start time of transcription phase
    file_exists = os.path.exists(csv_path)
    header = ["Timestamp", "Transcription", "Played"]
    # Create the CSV file with header if it does not exist
    if not file_exists:
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
    
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Loop for the first speech segment (wait indefinitely)
        while True:
            # End transcription if maximum duration is exceeded
            if time.time() - start_time > max_duration:
                print("⏰ Max transcription duration reached. Ending transcription phase.")
                return
            # Wait indefinitely until speech is detected
            audio = record_audio(timeout=None)
            if audio is None:
                continue
            print("⏳ Transcription Phase: Processing audio segment...")
            result = whisper_model.transcribe(audio, fp16=False)
            transcription = result.get("text", "").strip()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, transcription, False])
            csvfile.flush()  # Update CSV live
            print("📝 Transcription:", transcription, "\n")
            
            # After a valid transcription, wait for additional speech for up to 3 seconds
            while True:
                print("⏱ Waiting for additional speech for 2 seconds...")
                additional_audio = record_audio(timeout=2)
                if additional_audio is None:
                    print("🛑 No additional speech detected within 2 seconds. Ending transcription phase.")
                    return  # Exit the transcription phase
                # If additional speech is detected, transcribe it and loop again
                print("⏳ Transcription Phase: Processing additional audio segment...")
                result_add = whisper_model.transcribe(additional_audio, fp16=False)
                transcription_add = result_add.get("text", "").strip()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, transcription_add, False])
                csvfile.flush()
                print("📝 Transcription:", transcription_add, "\n")
                # Continue waiting for further additional speech
                # (This inner loop will exit if no speech is detected within 2 seconds.)


########################################
# Phase 2: LLM Response Generation
########################################

def run_llm_response_generation():
    """
    Run the LLM response generation phase. This function reads the CSV,
    generates assistant responses for rows missing them using a local API, and updates the CSV.
    """
    gemma2b_local_api = "http://127.0.0.1:1234/v1/chat/completions"
    def generate_assistant_response(transcription):
        # Define messages for the LLM conversation
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Satya, a personal assistant. Provide a concise, helpful, and friendly response. "
                    "Do not include any emojis in the output. There should be no symbols or special characters in your output."
                )
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
        # Prepare payload for the API request
        payload = {
            "model": "gemma-2-2b-it",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": False
        }
        try:
            response = requests.post(
                gemma2b_local_api, 
                headers={"Content-Type": "application/json"}, 
                json=payload
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error generating response for transcription: '{transcription}': {e}")
            return "Error generating response."
        data = response.json()
        try:
            generated_text = data["choices"][0]["message"]["content"].strip()
            return generated_text
        except (KeyError, IndexError):
            return "No response generated."
    
    # Read CSV data into a DataFrame
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Transcription"])  # Ensure transcriptions exist
    # Filter rows that lack an assistant response
    if "AssistantResponse" in df.columns:
        df_to_process = df[df["AssistantResponse"].isnull() | (df["AssistantResponse"] == "")]
    else:
        df_to_process = df
    # Generate responses concurrently for rows needing them
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(generate_assistant_response, df_to_process["Transcription"]))
    # Update the DataFrame with generated responses and save the CSV
    df.loc[df_to_process.index, "AssistantResponse"] = responses
    df.to_csv(csv_path, index=False)
    print("LLM Response Generation: Completed and CSV updated.")

########################################
# Phase 3: TTS Playback
########################################

# Initialize TTS model for playback
tts_playback = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
# Setup a separate VAD for playback monitoring
vad_playback = webrtcvad.Vad()
vad_playback.set_mode(2)
global_stop = False  # Global flag to indicate if user speech interrupts playback

def monitor_mic(play_obj, amp_threshold=2100, required_consecutive=5):
    """
    Monitor the microphone during TTS playback. Only if the audio input's amplitude exceeds
    the specified threshold and VAD confirms speech for a given number of consecutive frames
    will the playback be stopped.
    
    Parameters:
        play_obj: The playback object returned by simpleaudio.
        amp_threshold (int): Increased amplitude threshold to reduce false positives.
        required_consecutive (int): Increased number of consecutive frames required.
    """
    global global_stop
    speech_counter = 0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        while play_obj.is_playing():
            frame, _ = stream.read(FRAME_SIZE)
            mic_frame = frame[:, 0]  # Get mono channel data
            amplitude = np.abs(mic_frame).mean()  # Compute average amplitude
            
            # Check if amplitude is above the new, higher threshold and VAD confirms speech
            if amplitude > amp_threshold and vad_playback.is_speech(mic_frame.tobytes(), SAMPLE_RATE):
                speech_counter += 1
                if speech_counter >= required_consecutive:
                    print("User speech detected ({} consecutive frames, amplitude {:.0f}). Stopping playback.".format(speech_counter, amplitude))
                    play_obj.stop()  # Stop TTS playback
                    global_stop = True
                    break
            else:
                speech_counter = 0
            time.sleep(0.01)


def run_tts_playback():
    """
    Run the TTS playback phase. For each row in the CSV with an assistant response,
    convert the text to speech, play the audio, and monitor the microphone.
    If user speech is detected, stop playback and mark the remaining rows as played.
    The CSV is updated live after each row is processed.
    """
    global global_stop
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["AssistantResponse"])
    if "Played" not in df.columns:
        df["Played"] = False
    for idx, row in df.iterrows():
        if global_stop:
            print("User speech detected; marking remaining rows as played.")
            df.loc[idx:, "Played"] = True
            df.to_csv(csv_path, index=False)  # Save update to CSV
            break
        if row["Played"]:
            print(f"Skipping row {idx} as it has already been played.")
            continue
        text = str(row["AssistantResponse"]).strip()
        text = clean_text(text)
        if not text:
            print(f"Skipping row {idx} due to empty response after cleaning")
            continue
        print(f"Speaking for row {idx}: {text}")
        try:
            # Create an in-memory buffer for the generated audio
            audio_buffer = io.BytesIO()
            # Generate TTS audio from the assistant response
            tts_playback.tts_to_file(text=text, file_path=audio_buffer)
            audio_buffer.seek(0)
            # Read the generated audio data from the buffer
            data, samplerate = sf.read(audio_buffer, dtype='int16')
            audio_array = np.array(data, dtype=np.int16).tobytes()
            # Create a WaveObject for playback using simpleaudio
            wave_obj = sa.WaveObject(audio_array, num_channels=1, bytes_per_sample=2, sample_rate=samplerate)
            # Start playback
            play_obj = wave_obj.play()
            # Start a thread to monitor the microphone during playback
            monitor_thread = threading.Thread(target=monitor_mic, args=(play_obj,))
            monitor_thread.start()
            # Wait for playback to complete or be interrupted
            play_obj.wait_done()
            monitor_thread.join()
            if global_stop:
                print("User speech detected; marking remaining rows as played.")
                df.loc[idx:, "Played"] = True
                df.to_csv(csv_path, index=False)
                break
            # Mark this row as played and update the CSV live
            df.at[idx, "Played"] = True
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    df.to_csv(csv_path, index=False)

########################################
# Main Execution Pipeline (Continuous Loop)
########################################

if __name__ == '__main__':
    while True:
        # Phase 1: Transcription Phase
        try:
            print("\n--- Starting Transcription Phase ---")
            transcribe_audio(max_duration=60)  # Run transcription for up to 60 seconds
        except KeyboardInterrupt:
            print("\n🛑 Transcription interrupted by user.")
            break

        # Phase 2: LLM Response Generation Phase
        print("\n--- Running LLM Response Generation Phase ---")
        run_llm_response_generation()

        # Reset global_stop before starting TTS Playback Phase
        global_stop = False
        # Phase 3: TTS Playback Phase
        print("\n--- Running TTS Playback Phase ---")
        run_tts_playback()

        print("\n--- Cycle complete. Restarting in 1 second... ---\n")
        time.sleep(0.1)
