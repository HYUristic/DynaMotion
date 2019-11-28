import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init
import torchvision.transforms as transforms
import torchvision.datasets as dsets

class lstm_Model(nn.Module):
  def __init__(self, n_of_frames,n_of_cnn_frames, kernel_size, output_channel1, output_channel2):
    super(lstm_Model, self).__init__()

    self.n_of_frames = n_of_frames
    self.n_of_cnn_frames = n_of_cnn_frames
    self.kernel_size = kernel_size
    self.output_channel1 = output_channel1
    self.output_channel2 = output_channel2
    self.Rw1 = n_of_cnn_frames - kernel_size + 1
    self.Rh1 = 128 - kernel_size + 1
    self.Rw2 = (self.Rw1//2) - kernel_size + 1
    self.Rh2 = (self.Rh1//2) - kernel_size + 1
    self.Rw = self.Rw2//2
    self.Rh = self.Rh2//2
    self.sigmoid = nn.Sigmoid()

    self.layer1 = nn.Sequential(
        nn.Conv2d(1, output_channel1, kernel_size),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

    self.layer2 = nn.Sequential(
        nn.Conv2d(output_channel1, output_channel2, kernel_size),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

    self.layer3 = nn.LSTM(1*self.Rh*self.Rw, 128)

  def forward(self, x):
      out = self.layer1(x)
      out = self.layer2(out)
      out, (_, _) = self.layer3(out.view(x.size(0), 1, -1))
      out = self.sigmoid(out)

      return out

  def loss(self, output, target):
    loss = nn.MSELoss(output.view(1, -1), target)

    return loss
