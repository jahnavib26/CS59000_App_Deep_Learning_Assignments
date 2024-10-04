from transformers import pipeline
import gradio as gr

# whisper-small model for transcription
asr_pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
 
def predict(input_audio):
    # ASR pipeline to transcribe the audio
    transcription = asr_pipeline(input_audio)["text"]
    return transcription
 
# Gradio app setup with audio input
gradio_app = gr.Interface(
    predict,
    inputs=gr.Audio(label="Upload an audio file", sources=["upload","microphone"], type="filepath"),
    outputs=gr.Textbox(label="Transcription"),
    title="Speech-to-Text with Whisper Small",
)
 
if __name__ == "__main__":
    gradio_app.launch(share=True)
