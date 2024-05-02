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

def save_video(uploaded_file) -> str:

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
        
def extract_audio(video_path):
    video_clip = None

    # Создание обьекта VideoFileClip
    video_clip = VideoFileClip(video_path)

    if video_clip is not None:

        # Извлечение аудио из видео
        audio_clip = video_clip.audio
        audio_file_path = "audio.wav"
        audio_clip.write_audiofile(audio_file_path, codec="pcm_s16le")

        return audio_file_path, video_clip

def transcribe_audio(audio_file_path) -> str:
        
    # Запуск модели
    model = whisper.load_model("base")
    result = model.transcribe(audio_file_path, fp16=False)
    return result

def summarise_text(text) -> str:

    summarise = generate_detailed_summary(text["text"])
    st.write(summarise)

def rm_temp_files(audio_file_path, video_clip, video_path):
        
    # Удаление аудиофайла
    os.remove(audio_file_path)
    video_clip.close()
    os.remove(video_path)

def main():
    if uploaded_file:
        video_path = save_video(uploaded_file)
        if video_path:
            st.write("Extracting audio...")
            audio_file_path, video_clip = extract_audio(video_path)
            if audio_file_path:
                st.write("Transcribing...")
                text = transcribe_audio(audio_file_path)
                if text:
                    st.write("Summarising...")
                    summarise_text(text)
                    rm_temp_files(audio_file_path, video_clip, video_path)
                    st.success("Your video was transcibed and summarised")

if __name__ == "__main__":
    main()