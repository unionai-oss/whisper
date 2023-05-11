import json
from typing import Optional

import jax.numpy as jnp
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from whisper_jax import FlaxWhisperPipline


@task(
    requests=Resources(gpu="1", mem="15Gi", cpu="2"),
)
def jax_transcribe(
    audio: FlyteFile,
    chunk_length_s: Optional[float],
    stride_length_s: Optional[float],
    batch_size: Optional[int],
    language: Optional[str],
    task: Optional[str],
    return_timestamps: Optional[bool],
    checkpoint: str,
) -> str:
    pipeline = FlaxWhisperPipline(checkpoint, dtype=jnp.float16, batch_size=16)
    return json.dumps(
        pipeline(
            audio.download(),
            chunk_length_s,
            stride_length_s,
            batch_size,
            language,
            task,
            return_timestamps,
        )
    )


@workflow
def jax_simple_wf(
    audio: FlyteFile = "https://datasets-server.huggingface.co/assets/librispeech_asr/--/all/train.clean.100/1/audio/audio.mp3",
    chunk_length_s: Optional[float] = None,
    stride_length_s: Optional[float] = None,
    batch_size: Optional[int] = None,
    language: Optional[str] = None,
    task: Optional[str] = None,
    return_timestamps: Optional[bool] = None,
    checkpoint: str = "openai/whisper-large-v2",
):
    return jax_transcribe(
        audio=audio,
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        batch_size=batch_size,
        language=language,
        task=task,
        return_timestamps=return_timestamps,
        checkpoint=checkpoint,
    )
