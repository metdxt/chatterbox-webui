import os
import tempfile
import json
import shutil
import gradio as gr
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from perth import DummyWatermarker

# Constants
PERSONAS_DIR = os.path.join(os.path.dirname(__file__), "personas")
os.makedirs(PERSONAS_DIR, exist_ok=True)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")
model = ChatterboxTTS.from_pretrained(device=device)
model.watermarker = DummyWatermarker()
print("Model loaded.")


def generate_audio(
    text,
    repetition_penalty,
    min_p,
    top_p,
    audio_prompt,
    exaggeration,
    cfg_weight,
    temperature,
):
    if not text:
        raise gr.Error("Please enter text to generate.")

    print(f"Generating for text: {text[:50]}...")

    try:
        # audio_prompt path provided by Gradio Audio component (type="filepath")
        wav = model.generate(
            text,
            repetition_penalty=float(repetition_penalty),
            min_p=float(min_p),
            top_p=float(top_p),
            audio_prompt_path=audio_prompt,
            exaggeration=float(exaggeration),
            cfg_weight=float(cfg_weight),
            temperature=float(temperature),
        )

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            temp_path = fp.name

        # Move to CPU before saving if necessary
        ta.save(temp_path, wav.cpu(), model.sr)
        return temp_path

    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")


def get_persona_list():
    personas = []
    if os.path.exists(PERSONAS_DIR):
        for name in os.listdir(PERSONAS_DIR):
            if os.path.isdir(os.path.join(PERSONAS_DIR, name)):
                personas.append(name)
    return sorted(personas)


def save_persona(
    name,
    audio_prompt,
    repetition_penalty,
    min_p,
    top_p,
    exaggeration,
    cfg_weight,
    temperature,
):
    if not name:
        raise gr.Error("Please enter a name for the persona.")
    if not audio_prompt:
        raise gr.Error("Audio prompt is required to save a persona.")

    persona_path = os.path.join(PERSONAS_DIR, name)
    os.makedirs(persona_path, exist_ok=True)

    # Copy audio file
    _, ext = os.path.splitext(audio_prompt)
    if not ext:
        ext = ".wav" # Fallback
    
    saved_audio_path = os.path.join(persona_path, f"reference{ext}")
    shutil.copy2(audio_prompt, saved_audio_path)

    config = {
        "repetition_penalty": float(repetition_penalty),
        "min_p": float(min_p),
        "top_p": float(top_p),
        "exaggeration": float(exaggeration),
        "cfg_weight": float(cfg_weight),
        "temperature": float(temperature),
        "audio_filename": f"reference{ext}"
    }

    with open(os.path.join(persona_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return gr.Dropdown(choices=get_persona_list(), value=name), "Persona saved!"


def load_persona(name):
    if not name:
        return [gr.update()] * 7 # Return no updates

    persona_path = os.path.join(PERSONAS_DIR, name)
    config_path = os.path.join(persona_path, "config.json")
    
    if not os.path.exists(config_path):
        raise gr.Error(f"Config not found for persona: {name}")

    with open(config_path, "r") as f:
        config = json.load(f)

    audio_path = os.path.join(persona_path, config.get("audio_filename", "reference.wav"))
    
    return (
        audio_path,
        config.get("repetition_penalty", 1.2),
        config.get("min_p", 0.05),
        config.get("top_p", 1.0),
        config.get("exaggeration", 0.5),
        config.get("cfg_weight", 0.5),
        config.get("temperature", 0.8),
    )

def refresh_personas():
     return gr.Dropdown(choices=get_persona_list())


# Define the UI
with gr.Blocks(title="Chatterbox TTS") as demo:
    gr.Markdown("# Chatterbox TTS Web Interface")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Persona Manager")
            persona_dropdown = gr.Dropdown(label="Load Persona", choices=get_persona_list())
            refresh_btn = gr.Button("Refresh List", size="sm")
            
            gr.Markdown("### Save New Persona")
            new_persona_name = gr.Textbox(label="Persona Name", placeholder="e.g. My Narrator")
            save_persona_btn = gr.Button("Save Current Settings as Persona")
            save_msg = gr.Markdown("")

        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text to Speak", lines=5, placeholder="Enter text here..."
            )

            # Use Audio component for input, returning filepath
            audio_input = gr.Audio(
                label="Voice Reference (Audio Prompt)", type="filepath"
            )

            with gr.Accordion("Advanced Parameters", open=True):
                with gr.Row():
                    repetition_penalty = gr.Slider(
                        minimum=1.0, maximum=2.0, value=1.2, step=0.05, label="Repetition Penalty"
                    )
                    min_p = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.05, step=0.01, label="Min P"
                    )
                    top_p = gr.Slider(
                        minimum=0.0, maximum=1.0, value=1.0, step=0.01, label="Top P"
                    )
                with gr.Row():
                    exaggeration = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Exaggeration"
                    )
                    cfg_weight = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="CFG Weight"
                    )
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.8, step=0.05, label="Temperature"
                    )

            submit_btn = gr.Button("Generate Audio", variant="primary", size="lg")
            audio_output = gr.Audio(label="Generated Audio")

    # Event Wiring
    submit_btn.click(
        fn=generate_audio,
        inputs=[
            text_input,
            repetition_penalty,
            min_p,
            top_p,
            audio_input,
            exaggeration,
            cfg_weight,
            temperature,
        ],
        outputs=audio_output,
    )

    save_persona_btn.click(
        fn=save_persona,
        inputs=[
            new_persona_name,
            audio_input,
            repetition_penalty,
            min_p,
            top_p,
            exaggeration,
            cfg_weight,
            temperature,
        ],
        outputs=[persona_dropdown, save_msg],
    )

    persona_dropdown.change(
        fn=load_persona,
        inputs=[persona_dropdown],
        outputs=[
            audio_input,
            repetition_penalty,
            min_p,
            top_p,
            exaggeration,
            cfg_weight,
            temperature,
        ]
    )
    
    refresh_btn.click(fn=refresh_personas, inputs=[], outputs=[persona_dropdown])

if __name__ == "__main__":
    demo.launch()
