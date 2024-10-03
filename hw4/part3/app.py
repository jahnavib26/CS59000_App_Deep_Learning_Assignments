from transformers import pipeline
import gradio as gr

# Whisper model using Transformers pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")


# Function to handle the transcription
def transcribe(audio):
    result = asr(audio)["text"]
    return result
    
# Gradio interface
interface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    live=True
)

if __name__ == "__main__":
    interface.launch(share=True)