import streamlit as st
from moviepy.editor import VideoFileClip
from transformers import BartForConditionalGeneration, BartTokenizer
import whisper
import os

st.title("Speech Recognition from Video")
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
    

@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

def generate_summary(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    max_length = 2000  
    num_beams = 20    
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


@st.cache_data
def extract_audio(video_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_file_path = "audio.wav"
    audio_clip.write_audiofile(audio_file_path, codec="pcm_s16le")
    video_clip.close()
    return audio_file_path

def rm_temp_files(file):
    os.remove(file)

@st.cache_resource
def transcribe_audio(audio_file_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file_path, fp16=False)
    return result['text']

def main():

    if uploaded_file:

        # Создаёт директорию temp если она не создана автоматически
        if not os.path.exists("temp"):
            os.makedirs("temp")

        # Сохраняет видео во временной дирекстории
        video_path = os.path.join("temp", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.video(uploaded_file)
        st.write("Extracting audio...")
        audio_file_path = extract_audio(video_path)
        rm_temp_files(video_path)
        st.write("Transcribing...")
        text = transcribe_audio(audio_file_path)
        rm_temp_files(audio_file_path)
        # st.write(text)

        model, tokenizer = load_model()
        st.write("Summarising...")
        summary = generate_summary(model, tokenizer, text)
        st.write("Summary:", summary)

if __name__ == "__main__":
    main()