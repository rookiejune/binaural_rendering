import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


class L1_wav(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target)


class L1_sp(nn.Module):
    def __init__(
        self,
        **stft_keywords
    ):
        super(L1_sp, self).__init__()
        self.stft_keywords = stft_keywords

    def forward(self, input, target):
        """_summary_

        Args:
            input (_type_): (N, C, L)
            target (_type_): (N, C, L)
        """
        # N, C, F, T
        input_spec = torch.stft(
            rearrange(input, 'b c l -> (b c) l'),
            **self.stft_keywords
        )

        target_spec = torch.stft(
            rearrange(target, 'b c l -> (b c) l'),
            **self.stft_keywords
        )
        # Using complex spectrogram may cause trainning unstable!
        return F.l1_loss(input_spec.abs(), target_spec.abs())


class L1_wav_L1_sp(nn.Module):
    def __init__(
        self,
        **stft_keywords
    ):
        super().__init__()

        self.l1_wav = L1_wav()
        self.l1_sp = L1_sp(**stft_keywords)

    def forward(self, input, target):
        """_summary_

        Args:
            input (_type_): (N, C, L)
            target (_type_): (N, C, L)
        """
        # N, C, F, T
        return self.l1_wav(input, target) + self.l1_sp(input, target)


class SCL(nn.Module):
    def __init__(self, **stft_keywords) -> None:
        super().__init__()

        self.content_fn = L1_wav_L1_sp(**stft_keywords)
        self.stft_keywords = stft_keywords

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss_content = self.content_fn(input, target)
        input_spec = torch.stft(
            rearrange(input, 'b c l -> (b c) l'),
            **self.stft_keywords
        )

        target_spec = torch.stft(
            rearrange(target, 'b c l -> (b c) l'),
            **self.stft_keywords
        )

        input_mag, target_mag = map(
            lambda x: rearrange(x, '(b c) f t -> b c f t', c=2),
            (input_spec, target_spec)
        )

        input_ild = input_mag[:, 1] - input_mag[:, 0]
        target_ild = target_mag[:, 1] - target_mag[:, 0]
        return F.l1_loss(input_ild, target_ild) + loss_content