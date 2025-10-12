import queue
import numpy as np
import sounddevice as sd
from config.settings import SAMPLE_RATE, BLOCK_MS

class MicStream:
    def __init__(self):
        self.q = queue.Queue()
        self.blocksize = int(SAMPLE_RATE * BLOCK_MS / 1000)

    def _callback(self, indata, frames, time, status):
        if status:
            # Optionally print(status) for debugging
            pass
        # indata float32 [-1,1]
        self.q.put(indata.copy())

    def __enter__(self):
        self.stream = sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()
        return self

    def read(self):
        return self.q.get()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.stop()
        self.stream.close()
