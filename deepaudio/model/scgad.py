import torch
import torch.nn as nn

from .feature_crafter import SpectralFeatureCrafter
from einops import rearrange
import torch.nn.functional as F


class SCGAD(nn.Module):
    def __init__(
        self,
        num_transformer_layers: int,
        num_gru_layers: int,
        hidden_dim_ratio: float=1.,
    ):
        super().__init__()

        self.n_fft=2048

        self.feature_crafter = SpectralFeatureCrafter(
            n_fft=self.n_fft,
            window='hann',
            return_radian=False,
            cut_last_frequency_bin=True
        )

        num_bins = 1024

        out_features = int(num_bins * hidden_dim_ratio)
        self.mag_gru = nn.GRU(
            input_size=num_bins,
            hidden_size=out_features // 2,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.x0_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=out_features,
                nhead=8,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_transformer_layers,
        )

        self.x1_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=out_features,
                nhead=8,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_transformer_layers,
        )
        self.x2_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=out_features,
                nhead=8,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_transformer_layers,
        )
        self.x3_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=out_features,
                nhead=8,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_transformer_layers,
        )

        self.x01_attn = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=out_features,
                nhead=8,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_transformer_layers,
        )

        self.x12_attn = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=out_features,
                nhead=8,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_transformer_layers,
        )

        self.x23_attn = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=out_features,
                nhead=8,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_transformer_layers,
        )

        self.x34_attn = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=out_features,
                nhead=8,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_transformer_layers,
        )

        self.complex_gru = nn.GRU(
            input_size=4 * 2 * num_bins,
            hidden_size=4 * out_features // 2,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(
            in_features=8 * out_features,
            out_features=3 * 2 * num_bins,
        )

    def forward(self, x):
        # x: (batch_size, in_channels, time_steps)
        origin_length = x.shape[-1]

        origin_mag, origin_phase, origin_complex = self.feature_crafter(x)
        x = origin_mag
        x = rearrange(x, "b c f t -> b c t f")

        x0, x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        x0, _ = self.mag_gru(x0)
        x0 = self.x0_self_attn(x0)

        x1, _ = self.mag_gru(x1)
        x1 = self.x1_self_attn(x1)

        x2, _ = self.mag_gru(x2)
        x2 = self.x2_self_attn(x2)

        x3, _ = self.mag_gru(x3)
        x3 = self.x3_self_attn(x3)
        # b t f

        x01 = self.x01_attn(x0, x1)
        x12 = self.x12_attn(x1, x2)
        x23 = self.x23_attn(x2, x3)
        x30 = self.x34_attn(x3, x0)
        spatial = torch.cat([x01, x12, x23, x30], dim=-1)

        # cos = rearrange(cos, "b c f t -> b c t f")
        # sin = rearrange(sin, "b c f t -> b c t f")

        complex = rearrange(
            torch.view_as_real(origin_complex),
            "b c f t p -> b t (c f p)")
        content, _ = self.complex_gru(complex)

        x = self.fc(torch.cat([spatial, content], dim=-1))
        x = rearrange(x, "b t (c f p) -> b p c f t", p=3, c=2)

        base_mag = origin_mag[:, 0:1]
        base_phase = origin_phase[:, 0:1]
        # print(base_phase.shape)

        mask_mag = x[:, 0]
        delta_cos, delta_sin = x[:, 1], x[:, 2]
        mask_mag = F.relu(mask_mag)
        delta_cos = torch.tanh(delta_cos)
        delta_sin = torch.tanh(delta_sin)
        y_mag = mask_mag * base_mag

        fcos, fsin = base_phase.real, base_phase.imag
        y_cos = fcos * delta_cos - fsin * delta_sin
        y_sin = fsin * delta_cos + delta_sin * fcos

        real = y_mag * y_cos
        imag = y_mag * y_sin

        complex = torch.complex(real, imag)

        y = self.feature_crafter.istft(
            complex,
            length=origin_length
        )
        return y
