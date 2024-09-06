# Human Activity Recognition with LSTM on M5Stack
This project implements human activity recognition using a Long Short-Term Memory (LSTM) neural network on an M5Stack Gray device. The model is trained on the UCI Human Activity Recognition dataset and deployed on the M5Stack for real-time inference using its built-in IMU sensor.

## Installation 
1. Clone this repository:
```bash
git clone https://github.com/MaxRondelli/Human-Activity-Recognition-with-LSTM-on-M5Stack.git
```
2. Install the required Python packages:
```bash
pip install -r requirements.txt
```
3. Set up the Arduino IDE with M5Stack support following the official M5Stack documentation.
4. Download and include the two libraries necessary to run inference on the IoT device.
```ino
#include <M5Stack.h>
#include <TensorFlowLite_ESP32.h>
```
TensorFlowLite for ESP32 is necessary to load and interpreter the LSTM model. 

## Usage
1. Download the UCI HAR dataset.
```bash
cd data
python download_dataset.py
```
2. Start the training.
```bash
python main.py
```
3. The best model and all the statistics about the training will be saved in the folder results/that_specific_run/.
4. Convert the `best.pth` model to TensorFlowLite format. Look at the given path for the variable `BEST_MODEL` in the class `pytorch2tflite.py`
```bash
python pytorch2tflite.py
```
5. Convert the `edge_model.tflite` to `.h` format. It must be in the following format to be load on the M5Stack device.
```bash
xxd -i edge_model.tflite > edge_model.h
```
6. Once you have the converted model in `.h` format, add the attribute `const` to the model the variables in the model.
So change from:
```c
unsigned char edge_model_tflite[] = { model_inside_the_brackets }
unsigned int edge_model_tflite_len = 70912;
```
to
```h
const unsigned char edge_model_tflite[] = { model_inside_the_brackets }
const unsigned int edge_model_tflite_len = 70912;
```
7. To run the inference on M5Stack, move the `edge_model.h` inside the same folder where the `.ino` file is. Include the model inside the file and run the code. 

## Performance
Model's performance on the left and M5Stack inference on the right.
<div align="center">
  <img src="https://github.com/MaxRondelli/Human-Activity-Recognition-with-LSTM-on-M5Stack/blob/main/results/run_20240808_120746/confusion_matrix.png?raw=True" height="300px" alt="Model's Confusion Matrix">
  <img src="https://github.com/MaxRondelli/Human-Activity-Recognition-with-LSTM-on-M5Stack/blob/main/results/run_20240808_120746/inference-pic.jpeg" height="300px" alt="M5Stack Inference">
</div>
