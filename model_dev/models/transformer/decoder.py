
import torch
import torch.nn as nn

from model_dev.models.transformer.config import Config as Config


# TODO Implement Decoder
class Decoder(nn.Module):
    def __init__(self, config: Config):
        super(Decoder, self).__init__()

    def forward(self, x: torch.Tensor):
        return x