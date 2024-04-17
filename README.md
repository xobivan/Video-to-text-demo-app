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