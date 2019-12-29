import torch
import torch.nn as nn

from model_dev.models.transformer.encoder import Encoder as Encoder
from model_dev.models.transformer.decoder import Decoder as Decoder


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # TODO Implement Transformer using Enocder and Decoder Module

    def forward(self, x: torch.Tensor):
        return x
