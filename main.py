import streamlit as st
from moviepy.editor import VideoFileClip
from transformers import BartForConditionalGeneration, BartTokenizer
import whisper
import os

def generate_detailed_summary(text):
    # Load the pretrained BART model and tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Tokenize the text and generate the summary
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Set the generation parameters
    max_length = 1000  # Increase the maximum length of the summary
    num_beams = 10    # Number of beams for decoding
    top_k = 50        # Top-k sampling parameter
    top_p = 0.95      # Top-p sampling parameter

    summary_ids = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=num_beams,
        top_k=top_k,
        top_p=top_p,
        early_stopping=True,
        num_return_sequences=1  # Generate only one summary
    )

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

st.title("Speech Recognition from Video")


uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)

    video_clip = None

    # Проверяет загружен ли файл
    if uploaded_file.name:

        # Создаёт директорию temp если она не создана автоматически
        if not os.path.exists("temp"):
            os.makedirs("temp")

        # Сохраняет видео во временной дирекстории
        video_path = os.path.join("temp", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Создание обьекта VideoFileClip
        video_clip = VideoFileClip(video_path)

    if video_clip is not None:
        st.write("Transcribing...")

        # Извлечение аудио из видео
        audio_clip = video_clip.audio
        audio_file_path = "audio.wav"
        audio_clip.write_audiofile(audio_file_path, codec="pcm_s16le")

        # Запуск модели
        model = whisper.load_model("base")
        result = model.transcribe(audio_file_path, fp16=False)
        summaries = generate_detailed_summary(result["text"])
        st.success("Transcription and complete:")
        st.write(summaries)

        # Удаление аудиофайла
        os.remove(audio_file_path)
        video_clip.close()
        os.remove(video_path)
    else:
        st.error("Error: No file uploaded")