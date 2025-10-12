import time
import threading
import re
import numpy as np
import whisper

from audio.mic_stream import MicStream
from config.settings import (
    SAMPLE_RATE, CHUNK_SECONDS, STEP_SECONDS,
    MODEL_SIZE, DEVICE, LANGUAGE,
    WAKE_WORDS, SLEEP_WORDS, WAKE_COOLDOWN_SEC, SLEEP_COOLDOWN_SEC
)

# Compile whole-word regex like r'\b(hi|hello)\b'
WAKE_RE  = re.compile(r"\b(" + "|".join(map(re.escape, WAKE_WORDS)) + r")\b", re.IGNORECASE)
SLEEP_RE = re.compile(r"\b(" + "|".join(map(re.escape, SLEEP_WORDS)) + r")\b", re.IGNORECASE)

class STTService:
    """
    Constantly listens to mic.
    - When a WAKE word is heard, starts printing transcriptions to the console.
    - When a SLEEP word is heard, stops printing transcriptions (keeps listening silently).
    """
    def __init__(self):
        print(f"⏳ Loading Whisper model '{MODEL_SIZE}' on {DEVICE}...")
        self.model = whisper.load_model(MODEL_SIZE, device=DEVICE)

        # Transcription display state (False until wake word)
        self.active = False
        self._stop = False

        # Audio ring buffer
        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

        # Debounce times for wake/sleep
        self._last_wake  = 0.0
        self._last_sleep = 0.0

        # (Optional) avoid spammy repeats during ON mode
        self._last_printed = ""

    def _transcribe_chunk(self, audio_f32: np.ndarray) -> str:
        """
        audio_f32: float32 mono [-1,1], 16kHz
        Returns stripped text ('' if none).
        """
        result = self.model.transcribe(
            audio=audio_f32,
            language=LANGUAGE,
            verbose=False,
            condition_on_previous_text=False,  # stateless across chunks
            fp16=(DEVICE == "cuda"),
            temperature=0.0,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            word_timestamps=False
        )
        return (result.get("text") or "").strip()

    def _loop(self):
        """
        Worker that:
          - Every STEP_SECONDS, grabs the latest CHUNK_SECONDS from the ring buffer.
          - If inactive: only checks for wake words (prints nothing else).
          - If active: prints transcriptions and also checks for sleep words.
        """
        last_tick = 0.0
        while not self._stop:
            # Light tick loop
            time.sleep(0.02)
            now = time.time()
            if now - last_tick < STEP_SECONDS:
                continue
            last_tick = now

            # Copy the latest window from the ring buffer
            with self._lock:
                needed = int(SAMPLE_RATE * CHUNK_SECONDS)
                audio = self._buffer[-needed:] if self._buffer.size >= needed else self._buffer.copy()

            if audio.size == 0:
                continue

            audio = audio.astype(np.float32, copy=False)
            text = self._transcribe_chunk(audio)
            if not text:
                continue

            ltxt = text.lower()

            # Inactive → only listen for wake words; do not print the text
            if not self.active:
                if WAKE_RE.search(ltxt) and (now - self._last_wake) > WAKE_COOLDOWN_SEC:
                    self.active = True
                    self._last_wake = now
                    self._last_printed = ""
                    print("🟢 Heard wake word → Transcription ON")
                continue

            # Active → print text + look for sleep word
            # Avoid printing exact duplicates too often (whisper can repeat across short hops)
            if text != self._last_printed:
                print("🗣", text)
                self._last_printed = text

            if SLEEP_RE.search(ltxt) and (now - self._last_sleep) > SLEEP_COOLDOWN_SEC:
                self.active = False
                self._last_sleep = now
                print("🔴 Heard sleep word → Transcription OFF")

    def process_audio(self):
        print("🎤 Listening… Say a wake word to start (e.g., “hi/hello”), and a sleep word to stop (e.g., “bye”).")
        worker = threading.Thread(target=self._loop, daemon=True)
        worker.start()

        with MicStream() as mic:
            try:
                while not self._stop:
                    block = mic.read()            # (N,1) float32 in [-1,1]
                    if block is None:
                        time.sleep(0.005)
                        continue
                    block = np.squeeze(block)     # (N,)
                    if block.ndim != 1:
                        block = block.reshape(-1)

                    with self._lock:
                        # Append and cap to ~30s history
                        self._buffer = np.concatenate([self._buffer, block])
                        max_len = int(SAMPLE_RATE * 30)
                        if self._buffer.size > max_len:
                            self._buffer = self._buffer[-max_len:]
            except KeyboardInterrupt:
                pass
            finally:
                self._stop = True
                print("👋 Shutting down…")
