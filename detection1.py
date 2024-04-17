import streamlit as st
import numpy as np
from transformers import pipeline
import librosa

# Загрузка модели для обнаружения музыкальных инструментов
model = pipeline("audio-classification", model="dima806/musical_instrument_detection", framework="pt")

# Заголовок приложения
st.title("Музыкальный инструментово обнаружение")

# Загрузка аудиофайла
uploaded_file = st.file_uploader("Загрузите аудиофайл", type=["wav", "mp3"])

# Если файл загружен, применяем модель к аудиофайлу и выводим результат
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Преобразование аудиофайла в numpy ndarray
    audio, sr = librosa.load(uploaded_file, sr=None)
    prediction = model(audio)
    st.write("Результаты обнаружения:")
    for result in prediction:
        instrument = result["label"]
        probability = result["score"]
        st.write(f"- {instrument}: {probability}")
