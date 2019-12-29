import torch
import torch.nn as nn

from model_dev.models.transformer.encoder import Encoder as Encoder
from model_dev.models.transformer.decoder import Decoder as Decoder
from model_dev.models.transformer.config import Config as Config


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # TODO Implement Transformer using Enocder and Decoder Module
        self.config = Config()
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def forward(self, x: torch.Tensor):
        return x
