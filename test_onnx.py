import torch
import numpy as np
import config as cfg
import onnxruntime as ort

# Load the ONNX model
ort_session = ort.InferenceSession('model.onnx')

# Dummy input data
num_layers = cfg.n_layers
dummy_input = np.random.randn(2, num_layers, 9).astype(np.float32)

# Run inference
outputs = ort_session.run(None, {"input": dummy_input})

print(outputs)