import streamlit as st
from moviepy.editor import VideoFileClip
from transformers import BartForConditionalGeneration, BartTokenizer
import whisper
import os

@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

def generate_summary(model, tokenizer, text):
    inputs = tokenizer(str(text), return_tensors="pt", max_length=1024, truncation=True)
    max_length = 4000 
    num_beams = 30   
    top_k = 50        
    top_p = 0.95      
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=num_beams,
        top_k=top_k,
        top_p=top_p,
        early_stopping=True,
        num_return_sequences=1  
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

st.title("Speech Recognition from Video")

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

def save_video(uploaded_file):

    if uploaded_file is not None:
        st.video(uploaded_file)

        # Проверяет загружен ли файл
        if uploaded_file.name:

            # Создаёт директорию temp если она не создана автоматически
            if not os.path.exists("temp"):
                os.makedirs("temp")

            # Сохраняет видео во временной дирекстории
            video_path = os.path.join("temp", uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return video_path
        else:
            st.error("Error: No file uploaded")

@st.cache_data     
def extract_audio(video_path) -> str:
    video_clip = None

    # Создание обьекта VideoFileClip
    video_clip = VideoFileClip(video_path)

    if video_clip is not None:

        # Извлечение аудио из видео
        audio_clip = video_clip.audio
        audio_file_path = "audio.wav"
        audio_clip.write_audiofile(audio_file_path, codec="pcm_s16le")
        video_clip.close()
        rm_temp_files(video_path)

        return audio_file_path

@st.cache_resource
def transcribe_audio(audio_file_path) -> str:

    # Запуск модели
    model = whisper.load_model("base")
    result = model.transcribe(audio_file_path, fp16=False)
    rm_temp_files(audio_file_path)
    return result

def rm_temp_files(file):
        
    # Удаление файла
    os.remove(file)


def main():
    if uploaded_file:
        video_path = save_video(uploaded_file)
        if video_path:
            st.write("Extracting audio...")
            audio_file_path = extract_audio(video_path)
            if audio_file_path:
                st.write("Transcribing...")
                text = transcribe_audio(audio_file_path)
                if text:
                    model, tokenizer = load_model()
                    st.write("Summarising...")
                    summary = generate_summary(model, tokenizer, text)
                    st.success("Your video was transcribed and summarised")
                    st.write(summary)

if __name__ == "__main__":
    main()