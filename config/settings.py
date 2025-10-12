SAMPLE_RATE = 16000
BLOCK_MS = 30

# Smaller windows for snappier wake detection
CHUNK_SECONDS = 3          # analyze last 3s each run
STEP_SECONDS = 2           # run every 2s

MODEL_SIZE = "small.en"    # use "small.en" for best English accuracy; or "small"
DEVICE = "cuda"             # "cuda" if you have NVIDIA GPU
LANGUAGE = "en"

# Wake/Sleep keywords (case-insensitive, whole-word match)
WAKE_WORDS = ["hi", "hello", "hey"]
SLEEP_WORDS = ["bye"]

# basic debouncing to avoid double-triggering
WAKE_COOLDOWN_SEC = 2.0
SLEEP_COOLDOWN_SEC = 2.0
