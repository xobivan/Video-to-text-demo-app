import pytest
from unittest.mock import patch, MagicMock
from transformers import T5Tokenizer, T5ForConditionalGeneration
from main import load_whisper_model, load_t5_model, extract_audio, transcribe_audio, detect_language, generate_summary, rm_temp_files

@patch('main.whisper.load_model')
def test_load_whisper_model(mock_load_model):
    mock_load_model.return_value = "whisper_model"
    model = load_whisper_model()
    assert model == "whisper_model"
    mock_load_model.assert_called_once()

@patch('transformers.T5Tokenizer.from_pretrained')
@patch('transformers.T5ForConditionalGeneration.from_pretrained')
def test_load_t5_model(mock_model, mock_tokenizer):
    mock_model.return_value = "t5_model"
    mock_tokenizer.return_value = "t5_tokenizer"
    model, tokenizer = load_t5_model()
    assert model == "t5_model"
    assert tokenizer == "t5_tokenizer"
    mock_model.assert_called_once_with("t5-small")
    mock_tokenizer.assert_called_once_with("t5-small")

@patch('main.VideoFileClip')
def test_extract_audio(mock_video_file_clip):
    mock_audio_clip = mock_video_file_clip.return_value.audio
    mock_audio_clip.write_audiofile.return_value = None
    video_path = "test_video.mp4"
    audio_file_path = extract_audio(video_path)
    assert audio_file_path == "audio.wav"
    mock_video_file_clip.assert_called_once_with(video_path)
    mock_audio_clip.write_audiofile.assert_called_once_with("audio.wav", codec="pcm_s16le")
    mock_video_file_clip.return_value.close.assert_called_once()

@patch('os.remove')
def test_rm_temp_files(mock_remove):
    rm_temp_files("test_file")
    mock_remove.assert_called_once_with("test_file")

@patch('main.whisper.load_model')
def test_transcribe_audio(mock_load_model):
    mock_load_model.return_value = MagicMock()
    mock_model_instance = mock_load_model.return_value
    mock_model_instance.transcribe.return_value = {"text": "transcribed text"}
    
    model = load_whisper_model()
    audio_file_path = "audio.wav"
    text = transcribe_audio(audio_file_path, model)
    assert text == "transcribed text"
    mock_model_instance.transcribe.assert_called_once_with(audio_file_path, fp16=False)

def test_detect_language():
    assert detect_language("This is a test.") == "en"
    assert detect_language("Это тест.") == "ru"

@patch('transformers.T5Tokenizer.from_pretrained')
@patch('transformers.T5ForConditionalGeneration.from_pretrained')
def test_generate_summary(mock_model, mock_tokenizer):
    mock_model_instance = mock_model.return_value
    mock_tokenizer_instance = mock_tokenizer.return_value

    mock_tokenizer_instance.return_tensors = MagicMock(return_value={"input_ids": "mock_input_ids"})
    mock_model_instance.generate.return_value = ["summary_id"]
    mock_tokenizer_instance.decode.return_value = "summary"

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    summary = generate_summary(model, tokenizer, "test text")

    assert summary == "summary"
    mock_tokenizer_instance.decode.assert_called_once_with("summary_id", skip_special_tokens=True)
    mock_model_instance.generate.assert_called_once_with(
        mock_tokenizer_instance.return_tensors.return_value["input_ids"], max_length=200, min_length=30, num_beams=4, early_stopping=True
    )
