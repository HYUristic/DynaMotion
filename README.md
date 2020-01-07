# DynaMotion
> Maximize you musical emotion
This project is to generate dynamic loudness of notes (velocity) from plain piano roll using deep learning technique. We are hoping to enhance emotional impact of every existing music and explore the possibilities of expression.

## Goal
- Train and test attention based models
- Make VST Instrument to apply this technology in real life

## Ongoing
- Training CNN-LSTM based model

## Done
- Pytorch based MIDI dataloader

## Usage
### MIDI Dataloader
Checkout '''model_dev/SimpleDatasetTutorial.ipynb''' for more details.
1. Set '''settings.py'''
<b>Arguments</b><br>
- dataset_root : Where your dataset is going to be downloaded and stored
- dataset_info : DO NOT MODIFY. Essential for downloading dataset from cloud
- quantization_period : Quantize MIDI signal to (x) ticks per bit. e.g. if 16, quantized to 1/16
- length : Total number of ticks per datum. e.g. if 128 and quantization_period is 16, 8 bars of piano roll per datum

2. Import Settings and Initialize
```python
from settings import Settings
from utils.datasets import MaestroDataset

settings = Settings()
dataset = MaestroDataset("train", settings=settings)

print('total number : {}'.format(len(dataset)))
```

3. Access dataset
<b>Output</b><br>
- piano_roll : Numpy_Array(shape=(length, 128), dtype=bool) Plain piano roll. 128 being total number of piano keys
- velocity : Numpy_Array(shape=(length, 128), dtype=float32) Velocity range [0, 1]. 128 being total number of piano keys
```python
index = 100
piano_roll, velocity = dataset[index]
```
