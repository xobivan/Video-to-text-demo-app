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
    
    # Отображение результатов предсказания в виде таблицы
    st.write("Результаты обнаружения:")
    results_data = {"Инструмент": [], "Вероятность": []}
    for result in prediction:
        results_data["Инструмент"].append(result["label"])
        results_data["Вероятность"].append(result["score"])
    results_table = pd.DataFrame(results_data)
    st.write(results_table)
