import unittest
import torch
from deepaudio.model.scgad import SCGAD


class UnitTestCase(unittest.TestCase):
    def test_model_input_and_output(self):
        input = torch.randn(2, 4, 44100)
        model = SCGAD(
            num_transformer_layers=1,
            num_gru_layers=1,
            hidden_dim_ratio=1.
        )
        output = model(input)
        print(f'Input and output are of f{input.shape} and f{output.shape}, respectively.')


if __name__ == '__main__':
    unittest.main()