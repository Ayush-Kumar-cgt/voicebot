# Voice Assistant Flow Setup

This repository contains the microservices for a real-time voice, text, and chat assistant. 
The system is orchestrated by a central API and uses separate local services for Speech-to-Text (STT), LLM Chat, and Text-to-Speech (TTS).

## Pre-requisites

### 1. Install System Dependencies
If you are using Windows, you may need to install `ffmpeg` for audio manipulation via `pydub`.

### 2. Download kokoro weight model file:
https://drive.google.com/file/d/1gI49Z7kiEyNbKMpDA5COcNB9dv31Ez3D/view?usp=drive_link

### 3. Install Python Packages
Create a virtual environment and install the required packages:
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Add API Keys
Make sure your `.env` file contains your API keys:
```env
GROQ_API_KEY=your_groq_api_key
SARVAM_API_KEY=your_sarvam_api_key
```

## Running the Architecture

You need to spin up the 4 services, ideally in 4 separate terminal windows. 

**Terminal 1 (TTS Service):**
```bash
python tts_proxy.py
# Runs on localhost:8001
```

**Terminal 2 (Chat / LLM Service):**
```bash
python chat.py
# Runs on localhost:8002
```

**Terminal 3 (STT Service):**
```bash
python stt_sarvam.py
# Runs on localhost:8003
```

**Terminal 4 (Main Orchestrator & Frontend):**
```bash
python orchestrator.py
# Runs on localhost:8000
```

## Usage
Once all 4 services are successfully running:
1. Open your web browser.
2. Navigate to `http://localhost:8000`
3. Allow microphone permissions, speak into the mic, and experience the continuous real-time voice flow!
