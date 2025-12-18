# Chatterbox TTS Web UI

> [!IMPORTANT]
> This repo is not affiliated with Resemble AI

A web interface for the Chatterbox TTS model, built with Gradio. This application allows you to generate speech from text using voice cloning (via audio prompts) and provides tools to manage and save different voice "personas".

## Features

- **Text-to-Speech Generation**: Generate high-quality audio from text.
- **Voice Cloning**: Use a reference audio file ("Audio Prompt") to clone a specific voice.
- **Persona Management**: Save your favorite voice settings (reference audio + parameters) as named personas for easy reuse.
- **Advanced Controls**: Fine-tune generation with parameters like Temperature, Repetition Penalty, Min P, Top P, Exaggeration, and CFG Weight.

## Installation

1. Ensure you have Python 3.11 installed (didn't work on 3.12+ for me):

   ```bash
   uv python install 3.11
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

## Usage

Run the application:

```bash
uv run app.py
```

The interface will be available at `http://127.0.0.1:7860`.

## Interface Guide

### Main Inputs
- **Text to Speak**: The text you want to convert to speech.
- **Voice Reference (Audio Prompt)**: Upload an audio file of the voice you want to mimic.

### Advanced Parameters
- **Repetition Penalty**: Controls how much the model avoids repeating itself.
- **Min P / Top P**: Sampling parameters to control the diversity and quality of the output.
- **Exaggeration**: Adjusts the expressiveness of the speech.
- **CFG Weight**: Classifier-Free Guidance weight, influencing how strictly the model follows the conditioning.
- **Temperature**: Controls randomness (higher is more random/expressive, lower is more stable).

### Persona Manager
- **Load Persona**: Select a saved persona to automatically populate the audio reference and parameters.
- **Save New Persona**: Enter a name and click "Save Current Settings as Persona" to store the current configuration.

## Requirements

- Python 3.11
- CUDA-capable GPU is recommended for faster generation, but the application will fallback to CPU if CUDA is unavailable.
