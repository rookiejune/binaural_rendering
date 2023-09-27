from typing import Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def calculate_mag_and_phase(
    spectrogram: torch.Tensor,
    return_radian: bool=False,
    eps: float=1e-5,
    ) -> torch.Tensor:
    """Calculate magnitude and phase spectrogram for a given spectrogram.

    Args:
        spectrogram: Complex tensor (N, C, *shapes)

        return_radian: Whether the output should be radian, or sine and cosine.
            Radian may cause numerical unstable. (Default: ``False``)

        eps (float):

    Returns:
        phase: Real (`return_radian == True`) or complex (otherwise) tensor of shape (N, C, *shapes).
    """
    assert torch.is_complex(spectrogram)
    mag = spectrogram.abs().float()

    if return_radian:
        phase = spectrogram.angle()
    else:
        cos = spectrogram.real / (mag + eps)
        sin = spectrogram.imag / (mag + eps)
        phase = torch.complex(cos, sin)
    return mag, phase


class SpectralFeatureCrafter(nn.Module):
    def __init__(
        self,
        n_fft: int=2048,
        window: Union[Tensor, str]='hann',
        return_radian: bool=False,
        cut_last_frequency_bin: bool=True,
    ):
        """
        Args:
            n_fft (int, optional): _description_. Defaults to 2048.

            window (Tensor, optional): _description_. Defaults to None.

            return_radian (bool, optional): _description_. Defaults to False.

            cut_last_frequency_bin (bool, optional): _description_. Defaults to False.

        Input:
            A waveform of shape (batch_size, channels, time_steps).

        Output:
            Magnitude, phase spectrogram and complex spectrogram.
            All are of shape (batch_size, channels, frequencies, frames).
        """
        super().__init__()

        self.n_fft = n_fft
        if isinstance(window, str):
            if window == "hann":
                window = torch.hann_window(n_fft)
            else:
                raise Exception(f"Pass a string to assign window, "
                                "however, `{window}` is undefined.")
        elif isinstance(window, Tensor):
            pass
        else:
            raise Exception(f"It is unsupported to pass `{type(window)}` to assign window")
        self.register_buffer("window", window)
        self.cut_last_frequency_bin = cut_last_frequency_bin
        self.return_radian = return_radian

    def forward(
        self,
        x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # x: (batch_size, in_channels, time_steps)

        *_, self.time_steps = x.shape
        complex_spectrogram = self.stft(x)

        mag, phase = calculate_mag_and_phase(
            complex_spectrogram, return_radian=self.return_radian)

        if self.cut_last_frequency_bin:
            mag = mag[:, :, :-1, :]
            phase = phase[:, :, :-1, ...]
            complex_spectrogram = complex_spectrogram[:, :, :-1, :]
        return mag, phase, complex_spectrogram

    def stft(self, x: Tensor) -> Tensor:
        has_extra_shape = False
        if len(x.shape) >= 2:
            has_extra_shape = True
            extra_shape, L = x.shape[:-1], x.shape[-1]
            x = x.reshape(-1, L)

        y = torch.stft(
            x,
            n_fft=self.n_fft,
            window=self.window,
            return_complex=True,
        )
        if has_extra_shape:
            y = y.reshape(*extra_shape, *y.shape[1:])
        return y

    def istft(self, x: Tensor, length: int=None) -> Tensor:
        if self.cut_last_frequency_bin:
            x = F.pad(x, (0, 0, 0, 1))
        has_extra_shape = False
        if len(x.shape) >= 3:
            has_extra_shape = True
            extra_shape = x.shape[:-2]
            x  = x.reshape(-1, x.shape[-2], x.shape[-1])

        y = torch.istft(
            x,
            n_fft=self.n_fft,
            window=self.window,
            length=length
        )

        if has_extra_shape:
            y = y.reshape(*extra_shape, y.shape[-1])
        return y
