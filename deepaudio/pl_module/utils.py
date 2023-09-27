from typing import List
from torch.optim.lr_scheduler import LambdaLR
import torch
import numpy as np
from torch import stft as stft_
from scipy.signal import correlate, correlation_lags


def stft(x, *args, **kwds):
    has_extra_shape = False
    if len(x.shape) >= 2:
        has_extra_shape = True
        extra_shape, origin_length = x.shape[:-1], x.shape[-1]
        x = x.reshape(-1, origin_length)
    y = stft_(x, *args, **kwds)
    if has_extra_shape:
        y = y.reshape(*extra_shape, *y.shape[1:])
    return y


class WarmupAndReduce(LambdaLR):
    r"""Warmup And Reduce LambdaLR Scheduler

    if step <= warmup_steps, then factor = step / warmup_steps

    else factor = (step - warmup_steps) / reduce_steps
    """
    def __init__(self, optimizer, warmup_steps, reduce_steps):
        def f(step):
            if step <= warmup_steps:
                return step / warmup_steps
            else:
                return 0.9 ** ((step - warmup_steps) // reduce_steps)
        super().__init__(optimizer, f)


def calculate_sdr(ref: torch.Tensor, est: torch.Tensor) -> float:
    s_true = ref
    s_artif = est - ref
    sdr = 10.0 * (
        torch.log10(torch.clip(s_true ** 2, 1e-8, torch.inf))
        - torch.log10(torch.clip(s_artif ** 2, 1e-8, torch.inf))
    ).mean()
    return sdr


def calculate_dild_in_db(input: torch.Tensor, target: torch.Tensor):
    # assert input.shape[0] == 2
    device = input.device
    # input: (batch_size, channels, timesteps)
    M_x = stft(
        input,
        n_fft=2048,
        window=torch.hann_window(2048, device=device),
        return_complex=True,
    ).abs()
    # M_x: (batch_size, channels, frequency_bins, frames)
    M_y = stft(
        target,
        n_fft=2048,
        window=torch.hann_window(2048, device=device),
        return_complex=True
    ).abs()

    ILD_x = 10 * torch.log10(
        torch.clip(M_x[:, 0, ...] - M_x[:, 1, ...], 1e-5, np.Inf)
    )
    # ILD_x: (frequency_bins, frames)
    ILD_y = 10 * torch.log10(
        torch.clip(M_y[:, 0, ...] - M_y[:, 1, ...], 1e-5, np.Inf)
    )
    dild = (ILD_x - ILD_y).abs()
    return dild.mean()


def calculate_ditd_in_radian(input: torch.Tensor, target: torch.Tensor):
    # assert input.shape[0] == 2
    device = input.device
    # input: (batch_size, channels, timesteps)
    P_x = stft(
        input,
        n_fft=2048,
        window=torch.hann_window(2048, device=device),
        return_complex=True
    ).angle()
    # M_x: (batch_size, channels, frequency_bins, frames)
    P_y = stft(
        target,
        n_fft=2048,
        window=torch.hann_window(2048, device=device),
        return_complex=True
    ).angle()
    ITD_x = P_x[:, 0, ...] - P_x[:, 1, ...]
    # ILD_x: (frequency_bins, frames)
    ITD_y = P_y[:, 0, ...] - P_y[:, 1, ...]
    return (ITD_x - ITD_y).abs().mean()


def calculate_itd_in_step(input: np.ndarray):
    # input: batch_size, channels, time_steps
    if len(input.shape) == 2:
        input = input[None, ...]

    lag_sum = 0
    for i in range(input.shape[0]):
        correlation = correlate(input[i, 0], input[i, 1])
        lags = correlation_lags(input.shape[-1], input.shape[-1])
        lag = lags[np.argmax(correlation)]
        lag_sum += lag
    return lag_sum / input.shape[0]


def calculate_ditd_in_step(input: torch.Tensor, target: torch.Tensor):
    # correlation
    device = input.device
    input = input.cpu().numpy()
    target = target.cpu().numpy()

    lag1 = calculate_itd_in_step(input)
    lag2 = calculate_itd_in_step(target)
    return torch.scalar_tensor(np.abs(lag1 - lag2), device=device)


def collate_list_of_dicts(list_of_dicts: List[dict]) -> dict:
    tgt = {}
    for src in list_of_dicts:
        for key, value in src.items():
            tgt[key] = tgt.get(key, 0) + value
    return tgt