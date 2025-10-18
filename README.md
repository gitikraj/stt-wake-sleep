# 🗣️ Wake/Sleep Controlled Real-time Speech-to-Text (Whisper)

This project enables **real-time voice transcription** using **OpenAI’s Whisper model**, with **wake/sleep word detection** for hands-free control.  
Say **“hi / hello / hey”** to start transcription, and **“bye”** to stop it — the system keeps listening in the background and toggles states automatically.

---

## 🚀 Features

- 🎧 **Always Listening** — The microphone stream runs continuously.  
- 🗣️ **Wake Word Activation** — Starts transcribing when you say *hi / hello / hey*.  
- 🤫 **Sleep Command** — Stops transcription when you say *bye*.  
- ⚡ **Real-Time Processing** — Audio chunks are processed instantly using Whisper.  
- 🧠 **Accurate Transcription** — Uses OpenAI’s Whisper model (`small`, `base`, or `tiny`).  
- 🧩 **Modular Design** — Cleanly separated audio streaming, model inference, and wake/sleep logic.  
- 🧵 **Threaded Processing** — Uses background threads for continuous listening without lag.

---

## 🧰 Tech Stack

| Component | Description |
|------------|--------------|
| **Python** | Core language for the pipeline |
| **Whisper** | Speech-to-text model by OpenAI |
| **PyTorch / Faster-Whisper** | Model inference |
| **Numpy** | Audio data handling |
| **Threading** | Background real-time processing |
| **Custom MicStream** | Continuous microphone capture |
