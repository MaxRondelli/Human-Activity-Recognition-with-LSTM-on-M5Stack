import torch
from functions import evaluate
from model import LSTMModel
import config as cfg

BEST_MODEL = '/home/massimorondelli/Human-Activity-Recognition-with-Recurrent-Neural-Networks-on-IoT-Device/results/run_20240808_120746/models/best.pth'
input_size = cfg.n_input
hidden_size = cfg.n_hidden
num_layers = cfg.n_layers
num_classes = cfg.n_classes

model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
# Load your trained model
model.load_state_dict(torch.load(BEST_MODEL))
model.eval()

# Example of a dummy input: batch size = 1, sequence length = 100, input size = 9
dummy_input = torch.randn(2, num_layers, 9)

torch.onnx.export(
    model,
    (dummy_input,),  # Input as a tuple
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'},
                  'output': {0: 'batch_size'}}
)