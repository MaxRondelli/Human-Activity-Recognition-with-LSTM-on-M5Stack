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

class LSTMModelWrapper(torch.nn.Module):
    def __init__(self, lstm_model):
        super().__init__()
        self.lstm_model = lstm_model

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        batch_size, seq_length, _ = x.shape
        h0 = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=x.device)
        c0 = torch.zeros(NUM_LAYERS, batch_size, HIDDEN_SIZE, device=x.device)
        
        # Transpose x to (seq_length, batch_size, input_size)
        x = x.transpose(0, 1)
        
        output, (hn, cn) = self.lstm_model.lstm1(x, (h0, c0))
        return self.lstm_model.fc(output[-1, :, :])

original_model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
original_model.load_state_dict(torch.load(BEST_MODEL, map_location=torch.device('cpu'), weights_only=True))
original_model.eval()

model = LSTMModelWrapper(original_model)
model.eval()

# dummy input 
batch_size = 1
seq_length = 2
dummy_input = torch.randn(batch_size, seq_length, INPUT_SIZE)

edge_model = ai_edge_torch.convert(model, (dummy_input,))
edge_model.export("edge_model.tflite")
print("Model converted and exported successfully.")


"""
With the following bash line you create a C file for the model, used for the inference on the edge device. 
The .h model will be interpreted by tensorflow lite. 
"""
# xxd -i edge_model.tflite > edge_model.h 