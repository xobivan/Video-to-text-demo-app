Выдаёт следующие ошибки

ValueError: The <class 'transformers.pipelines.audio_classification.AudioClassificationPipeline'> is only available in PyTorch.
Traceback:

File "/home/xobi_van/.local/lib/python3.10/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 535, in _run_script
    exec(code, module.__dict__)
File "/home/xobi_van/Projects/Instrument-Detection/detection.py", line 5, in <module>
    model = pipeline("audio-classification", model="dima806/musical_instrument_detection")
File "/home/xobi_van/.local/lib/python3.10/site-packages/transformers/pipelines/__init__.py", line 1070, in pipeline
    return pipeline_class(model=model, framework=framework, task=task, **kwargs)
File "/home/xobi_van/.local/lib/python3.10/site-packages/transformers/pipelines/audio_classification.py", line 99, in __init__
    raise ValueError(f"The {self.__class__} is only available in PyTorch.")


А если в строке model = pipeline("audio-classification", model="dima806/musical_instrument_detection", framework="pt") указать конкретный фреймворк "pt", то показывает следующее 

Pipeline cannot infer suitable model classes from dima806/musical_instrument_detection
Traceback:

File "/home/xobi_van/.local/lib/python3.10/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 535, in _run_script
    exec(code, module.__dict__)
File "/home/xobi_van/Projects/Instrument-Detection/detection.py", line 5, in <module>
    model = pipeline("audio-classification", model="dima806/musical_instrument_detection", framework="pt")
File "/home/xobi_van/.local/lib/python3.10/site-packages/transformers/pipelines/__init__.py", line 870, in pipeline
    framework, model = infer_framework_load_model(
File "/home/xobi_van/.local/lib/python3.10/site-packages/transformers/pipelines/base.py", line 250, in infer_framework_load_model
    raise ValueError(f"Pipeline cannot infer suitable model classes from {model}")


   ```
для     detection2.py после запуска на стримлите в терминале выдает такую ошибку:Wav2Vec2ForSequenceClassification: ['wav2vec2.encoder.pos_conv_embed.conv.weight_v', 'wav2vec2.encoder.pos_conv_embed.conv.weight_g']
- This IS expected if you are initializing Wav2Vec2ForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing Wav2Vec2ForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of Wav2Vec2ForSequenceClassification were not initialized from the model checkpoint at dima806/musical_instrument_detection and are newly initialized: ['wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1', 'wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of the model checkpoint at dima806/musical_instrument_detection were not used when initializing Wav2Vec2ForSequenceClassification: ['wav2vec2.encoder.pos_conv_embed.conv.weight_v', 'wav2vec2.encoder.pos_conv_embed.conv.weight_g']
- This IS expected if you are initializing Wav2Vec2ForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing Wav2Vec2ForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of Wav2Vec2ForSequenceClassification were not initialized from the model checkpoint at dima806/musical_instrument_detection and are newly initialized: ['wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1', 'wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.Это сообщение об ошибке связано с использованием модели Wav2Vec2ForSequenceClassification, которая, по-видимому, была инициализирована с предварительно обученной моделью Wav2Vec2, но не была дообучена на вашей конкретной задаче. Это сообщение об ошибке говорит о том, что некоторые веса модели не были использованы при инициализации, и некоторые веса были только что инициализированы.

Чтобы решить эту проблему, вам следует дообучить модель Wav2Vec2ForSequenceClassification на вашей конкретной задаче. Это позволит модели адаптироваться к вашим данным и улучшить качество предсказаний.

Вы можете провести дообучение модели, используя ваши собственные данные и задачу классификации звуков или музыкальных инструментов. Это позволит модели лучше адаптироваться к вашей конкретной задаче и улучшить качество предсказаний.

