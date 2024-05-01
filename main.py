import os
import streamlit as st
from moviepy.editor import VideoFileClip
import whisper

st.title("Speech Recognition from Video")

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
if uploaded_file is not None:
    st.video(uploaded_file)

    video_clip = None

    # Check if a file is uploaded
    if uploaded_file.name:

        # Create a temporary directory if it doesn't exist
        if not os.path.exists("temp"):
            os.makedirs("temp")

        # Save the video in the temporary directory
        video_path = os.path.join("temp", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Create a VideoFileClip object
        video_clip = VideoFileClip(video_path)

    if video_clip is not None:
        st.write("Transcribing...")

        # Extract audio from the video
        audio_clip = video_clip.audio
        audio_file_path = "audio.wav"
        audio_clip.write_audiofile(audio_file_path, codec="pcm_s16le")

        # Load the model
        model = whisper.load_model("base")
        result = model.transcribe(audio_file_path, fp16=False)

        st.success("Transcription complete:")
        st.write(result["text"])

        # Delete the audio file and video file
        os.remove(audio_file_path)
        video_clip.close()
        os.remove(video_path)
    else:
        st.error("Error: No file uploaded")

        # TODO: Add a text summarization model (for example, this one: https://huggingface.co/d0rj/rut5-base-summ).
        #  Integrate it into the Streamlit app. Is it possible to apply st
