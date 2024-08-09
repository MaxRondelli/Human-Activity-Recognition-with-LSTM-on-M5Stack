import onnx
import warnings
from onnx_tf.backend import prepare

warnings.filterwarnings('ignore') # Ignore all the warning messages in this tutorial
model = onnx.load('assets/super_resolution.onnx') # Load the ONNX file
tf_rep = prepare(model) # Import the ONNX model to Tensorflow

print(tf_rep.inputs) # Input nodes to the model
print('-----')
print(tf_rep.outputs) # Output nodes from the model
print('-----')
print(tf_rep.tensor_dict) # All nodes in the model