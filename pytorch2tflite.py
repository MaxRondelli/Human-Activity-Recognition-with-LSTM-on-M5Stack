import os 
import torch
import ai_edge_torch
import config as cfg
from model import LSTMModel

os.environ['PJRT_DEVICE'] = 'CPU'

BEST_MODEL = '/home/massimorondelli/Human-Activity-Recognition-with-Recurrent-Neural-Networks-on-IoT-Device/results/run_20240808_120746/models/best.pth'
INPUT_SIZE = cfg.n_input
HIDDEN_SIZE = cfg.n_hidden
NUM_LAYERS = cfg.n_layers
NUM_CLASSES = cfg.n_classes

model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
dummy_input = (torch.randn(2, NUM_LAYERS, 9),)

edge_model = ai_edge_torch.convert(model.eval(), dummy_input)
edge_model.export("edge_model.tflite")

# Run !xxd -i converted_model.tflite > model_data.cc to convert the model to .cc format 