import streamlit as st
from moviepy.editor import VideoFileClip
import whisper
import os

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

        st.success("Transcription complete:")
        st.write(result["text"])

        # Удаление аудиофайла
        os.remove(audio_file_path)
        video_clip.close()
        os.remove(video_path)
    else:
        st.error("Error: No file uploaded")

        #TODO Добавить модель - суммаризатор текста ( например эту https://huggingface.co/d0rj/rut5-base-summ ). Добавить её в приложение на стримлит. Кто то знает можно ли на стримлите накатить стили ? 