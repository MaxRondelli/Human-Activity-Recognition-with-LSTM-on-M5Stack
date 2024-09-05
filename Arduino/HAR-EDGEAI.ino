#include <M5Stack.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "edge_model.h"
#include "Functions.h"

const int SEQUENCE_LENGTH = 64;
const int NUM_FEATURES = 118;
const int NUM_LABELS = 6;

const char* LABELS[] = {
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
};

// Global variables for TensorFlow Lite
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;
// memory buffer for tflite interpreter
constexpr int kTensorArenaSize = 40 * 1024; 
uint8_t tensorArena[kTensorArenaSize];

// Global variables for sensor data processing
float gravityX = 0, gravityY = 0, gravityZ = 0;

void setup() {
  M5.begin();
  M5.Power.begin();
  
  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.setTextColor(WHITE);
  M5.Lcd.setTextSize(2);
  M5.Lcd.setCursor(0, 0);
  M5.Lcd.println("HAR Inference");

  tflModel = tflite::GetModel(edge_model_tflite);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    M5.Lcd.println("Model schema mismatch!");
    return;
  }

  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, kTensorArenaSize, &tflErrorReporter);

  TfLiteStatus allocate_status = tflInterpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    M5.Lcd.println("AllocateTensors() failed");
    return;
  }

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  M5.IMU.Init();

  M5.Lcd.println("Setup complete");
  delay(500);
}

void loop() {
  float accX[SEQUENCE_LENGTH], accY[SEQUENCE_LENGTH], accZ[SEQUENCE_LENGTH];
  float gyroX[SEQUENCE_LENGTH], gyroY[SEQUENCE_LENGTH], gyroZ[SEQUENCE_LENGTH];
  float bodyAccX[SEQUENCE_LENGTH], bodyAccY[SEQUENCE_LENGTH], bodyAccZ[SEQUENCE_LENGTH];
  float features[NUM_FEATURES];
  
  for (int i = 0; i < SEQUENCE_LENGTH; i++) {
    M5.IMU.getAccelData(&accX[i], &accY[i], &accZ[i]);
    M5.IMU.getGyroData(&gyroX[i], &gyroY[i], &gyroZ[i]);
    
    gravityX = lowPassFilter(accX[i], gravityX);
    gravityY = lowPassFilter(accY[i], gravityY);
    gravityZ = lowPassFilter(accZ[i], gravityZ);

    bodyAccX[i] = accX[i] - gravityX;
    bodyAccY[i] = accY[i] - gravityY;
    bodyAccZ[i] = accZ[i] - gravityZ;

    delay(20); 
  }

  int featureIndex = 0;

  for (int axis = 0; axis < 3; axis++) {
    float* acc = (axis == 0) ? bodyAccX : (axis == 1) ? bodyAccY : bodyAccZ;
    float* gyro = (axis == 0) ? gyroX : (axis == 1) ? gyroY : gyroZ;

    features[featureIndex++] = calculateMean(acc, SEQUENCE_LENGTH);
    features[featureIndex++] = calculateStd(acc, SEQUENCE_LENGTH, features[featureIndex-1]);
    features[featureIndex++] = calculateMAD(acc, SEQUENCE_LENGTH);
    features[featureIndex++] = calculateIQR(acc, SEQUENCE_LENGTH);
    features[featureIndex++] = calculateEntropy(acc, SEQUENCE_LENGTH);
    
    float arCoeffs[4];
    calculateARCoefficients(acc, arCoeffs, SEQUENCE_LENGTH, 4);
    for (int j = 0; j < 4; j++) {
      features[featureIndex++] = arCoeffs[j];
    }

    features[featureIndex++] = calculateMean(gyro, SEQUENCE_LENGTH);
    features[featureIndex++] = calculateStd(gyro, SEQUENCE_LENGTH, features[featureIndex-1]);
    features[featureIndex++] = calculateMAD(gyro, SEQUENCE_LENGTH);
    features[featureIndex++] = calculateIQR(gyro, SEQUENCE_LENGTH);
    features[featureIndex++] = calculateEntropy(gyro, SEQUENCE_LENGTH);
    
    calculateARCoefficients(gyro, arCoeffs, SEQUENCE_LENGTH, 4);
    for (int j = 0; j < 4; j++) {
      features[featureIndex++] = arCoeffs[j];
    }
  }

  // Frequency domain features
  for (int i = 0; i < 13; i++) {
    features[featureIndex++] = 0.5; 
  }

  // Correlation features
  float corrAcc[3], corrGyro[3];
  calculateCorrelation(bodyAccX, bodyAccY, bodyAccZ, corrAcc, SEQUENCE_LENGTH);
  calculateCorrelation(gyroX, gyroY, gyroZ, corrGyro, SEQUENCE_LENGTH);
  for (int i = 0; i < 3; i++) {
    features[featureIndex++] = corrAcc[i];
    features[featureIndex++] = corrGyro[i];
  }

  // Prepare input tensor
  for (int i = 0; i < NUM_FEATURES; i++) {
    tflInputTensor->data.f[i] = features[i];
  }

  // Run inference
  TfLiteStatus invoke_status = tflInterpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    M5.Lcd.println("Invoke failed!");
    return;
  }

  int maxIndex = 0;
  float maxValue = tflOutputTensor->data.f[0];
  float sum = 0.0f;
  float minValue = tflOutputTensor->data.f[0];

  for (int i = 0; i < NUM_LABELS; i++) {
    float value = tflOutputTensor->data.f[i];
    sum += value;
    if (value > maxValue) {
      maxIndex = i;
      maxValue = value;
    }
    if (value < minValue) {
      minValue = value;
    }
  }

  // confidence
  float confidence;
  if (sum == 0 || maxValue == minValue) {
    confidence = 0.0f;
  } else if (sum < 1e-6) {
    confidence = ((maxValue - minValue) / (sum - NUM_LABELS * minValue)) * 100.0f;
  } else {
    confidence = (maxValue / sum) * 100.0f;
  }
  confidence = std::max(0.0f, std::min(100.0f, confidence));

  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.setTextSize(2);
  M5.Lcd.setCursor(0, 0);
  M5.Lcd.println("Predicted Activity:");
  M5.Lcd.setTextSize(3);
  M5.Lcd.setTextColor(YELLOW);
  M5.Lcd.println(LABELS[maxIndex]);
  M5.Lcd.setTextSize(2);
  M5.Lcd.setTextColor(WHITE);
  M5.Lcd.print("Confidence: ");
  M5.Lcd.print(confidence, 1);
  M5.Lcd.println("%");

  delay(1000);
}