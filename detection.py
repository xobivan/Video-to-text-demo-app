import streamlit as st
from transformers import pipeline

# Загрузка модели для обнаружения музыкальных инструментов
model = pipeline("audio-classification", model="dima806/musical_instrument_detection", framework="pt")

# Заголовок приложения
st.title("Музыкальный инструментово обнаружение")

# Загрузка аудиофайла
uploaded_file = st.file_uploader("Загрузите аудиофайл", type=["wav", "mp3"])

# Если файл загружен, применяем модель к аудиофайлу и выводим результат
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    prediction = model(uploaded_file)
    st.write("Результаты обнаружения:")
    for result in prediction:
        instrument = result["label"]
        probability = result["score"]
        st.write(f"- {instrument}: {probability}")
