import json

import numpy as np
import requests
import torch
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read


@task(requests=Resources(gpu="1", mem="15Gi", cpu="2"))
def torch_transcribe(
    checkpoint: str,
    audio: FlyteFile,
    chunk_length: float,
    batch_size: int,
    return_timestamps: bool,
) -> str:
    pipe = pipeline(
        "automatic-speech-recognition",
        model=checkpoint,
        chunk_length_s=chunk_length,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    local_audio_path = audio.download()
    if local_audio_path.startswith("http://") or local_audio_path.startswith(
        "https://"
    ):
        # We need to actually check for a real protocol, otherwise it's impossible to use a local file
        # like http_huggingface_co.png
        inputs = requests.get(inputs).content
    else:
        with open(local_audio_path, "rb") as f:
            inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, 16000)

    if not isinstance(inputs, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")

    if len(inputs.shape) != 1:
        raise ValueError(
            "We expect a single channel audio input for AutomaticSpeechRecognitionPipeline"
        )

    prediction = pipe(
        inputs, batch_size=batch_size, return_timestamps=return_timestamps
    )
    return json.dumps(prediction)


@workflow
def torch_wf(
    checkpoint: str = "openai/whisper-large-v2",
    audio: FlyteFile = "https://datasets-server.huggingface.co/assets/librispeech_asr/--/all/train.clean.100/1/audio/audio.mp3",
    chunk_length: float = 30.0,
    batch_size: int = 16,
    return_timestamps: bool = False,
) -> str:
    return torch_transcribe(
        checkpoint=checkpoint,
        audio=audio,
        chunk_length=chunk_length,
        batch_size=batch_size,
        return_timestamps=return_timestamps,
    )
