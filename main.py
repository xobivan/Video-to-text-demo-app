import streamlit as st
from moviepy.editor import VideoFileClip
from transformers import T5ForConditionalGeneration, T5Tokenizer
import whisper
import os

st.title("Speech Recognition from Video")
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("base")
    return model

@st.cache_resource
def load_t5_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

def extract_audio(video_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_file_path = "audio.wav"
    audio_clip.write_audiofile(audio_file_path, codec="pcm_s16le")
    video_clip.close()
    return audio_file_path

def rm_temp_files(file):
    os.remove(file)

def transcribe_audio(audio_file_path, model):
    result = model.transcribe(audio_file_path, fp16=False)
    return result['text']

def detect_language(text):
    if all(ord(char) < 128 for char in text):
        return "en"
    else:
        return "ru"

def generate_summary(model, tokenizer, text, max_length=200, min_length=30):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    whisper_model = load_whisper_model()
    t5_model, t5_tokenizer = load_t5_model()

    if uploaded_file:

        # Создаёт директорию temp если она не создана автоматически
        if not os.path.exists("temp"):
            os.makedirs("temp")

        # Сохраняет видео во временной директории
        video_path = os.path.join("temp", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.video(uploaded_file)
        st.write("Extracting audio...")
        audio_file_path = extract_audio(video_path)
        rm_temp_files(video_path)
        st.write("Transcribing...")
        text = transcribe_audio(audio_file_path, whisper_model)
        rm_temp_files(audio_file_path)
        st.write(text)

        if detect_language(text) == "ru":
            st.write("Warning: The summary model is optimized for English text!")
        
        st.write("Summarising...")
        summary = generate_summary(t5_model, t5_tokenizer, text)
        st.write("Summary:", summary)

if __name__ == "__main__":
    main()
