from typing import Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F


def load_audio(
    audio_path: str,
    mono: bool,
    sample_rate: float,
    offset: float = 0.0,
    duration: float = None,
) -> np.ndarray:
    r"""Load audio.

    Args:
        audio_path: str
        mono: bool
        sample_rate: float
    """
    audio, _ = librosa.core.load(
        audio_path, sr=sample_rate, mono=mono, offset=offset, duration=duration
    )
    # (audio_samples,) | (channels_num, audio_samples)

    if audio.ndim == 1:
        audio = audio[None, :]
        # (1, audio_samples,)

    return audio


def load_random_segment(
    audio_path: str,
    random_state: int,
    segment_seconds: float,
    mono: bool,
    sample_rate: int,
) -> np.ndarray:
    r"""Randomly select an audio segment from a recording."""

    duration = librosa.get_duration(filename=audio_path)

    start_time = random_state.uniform(0.0, duration - segment_seconds)

    audio = load_audio(
        audio_path=audio_path,
        mono=mono,
        sample_rate=sample_rate,
        offset=start_time,
        duration=segment_seconds,
    )
    # (channels_num, audio_samples)

    return audio


def rms_norm(
    waveform: Union[np.ndarray, torch.Tensor],
    dim=0,
    level=0
    ) -> Union[np.ndarray, torch.Tensor]:
    """Root Mean Square normalization for audio in waveform.

    Args:
        waveform (np.ndarray): (*, L, *)
        dim (_type_, optional): the L dim. Defaults to int.
        level (int, optional): _description_. Defaults to 0.

    Returns:
        np.ndarray: _description_
    """
    r = 10 ** (level / 10.)
    if isinstance(waveform, np.ndarray):
        a = np.sqrt(waveform.shape[dim] * r ** 2 / np.sum(waveform ** 2, axis=dim, keepdims=True))
    elif isinstance(waveform, torch.Tensor):
        a = torch.sqrt(waveform.shape[dim] * r ** 2 / torch.sum(waveform ** 2, dim=dim, keepdim=True))
    else:
        raise NotImplementedError("rms_norm: Unsupport type for waveform: {}".format(type(waveform)))
    return a * waveform


def fix_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    length: int,
) -> Union[np.ndarray, torch.Tensor]:
    """_summary_

    Args:
        waveform (np.ndarray): (N, C, L) or (C, L)
    """
    origin_len = waveform.shape[-1]
    is_numpy = False
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
        is_numpy = True

    if origin_len > length:
        cutoff = origin_len - length
        waveform[..., :] = waveform[..., :-1-cutoff]
    else:
        padding = length - origin_len
        waveform = F.pad(waveform, (0, padding))

    if is_numpy:
        waveform = np.array(waveform)
    return waveform


def magnitude_to_db(x: float) -> float:
    eps = 1e-10
    return 20.0 * np.log10(max(x, eps))


def db_to_magnitude(x: float) -> float:
    return 10.0 ** (x / 20)


def float32_to_int16(x: np.float32) -> np.int16:

    x = np.clip(x, a_min=-1, a_max=1)

    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x: np.int16) -> np.float32:

    return (x / 32767.0).astype(np.float32)