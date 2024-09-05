#ifndef FUNCTION_H
#define FUNCTION_H
#include <Arduino.h>

// Function declarations
float lowPassFilter(float input, float prev_output, float alpha = 0.8);
void medianFilter(float* input, float* output, int size);
float calculateMagnitude(float x, float y, float z);
float calculateMean(float* array, int size);
float calculateStd(float* array, int size, float mean);
float calculateMAD(float* array, int size);
float calculateEnergy(float* array, int size);
float calculateIQR(float* array, int size);
float calculateEntropy(float* array, int size);
void calculateCorrelation(float* x, float* y, float* z, float* result, int size);
void calculateJerk(float* input, float* output, int size);
void calculateARCoefficients(float* input, float* output, int size, int order);

// Function implementations
float lowPassFilter(float input, float prev_output, float alpha) {
    return alpha * prev_output + (1 - alpha) * input;
}

float calculateMean(float* data, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

float calculateStd(float* data, int size, float mean) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += (data[i] - mean) * (data[i] - mean);
    }
    return sqrt(sum / size);
}

float calculateMAD(float* data, int size) {
    float median = data[size / 2]; 
    float* deviations = new float[size];
    for (int i = 0; i < size; i++) {
        deviations[i] = fabs(data[i] - median);
    }
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (deviations[j] > deviations[j + 1]) {
                float temp = deviations[j];
                deviations[j] = deviations[j + 1];
                deviations[j + 1] = temp;
            }
        }
    }
    float mad = deviations[size / 2];
    delete[] deviations;
    return mad;
}

float calculateIQR(float* data, int size) {
    return data[3 * size / 4] - data[size / 4];
}

float calculateEntropy(float* data, int size) {
    float entropy = 0;
    for (int i = 0; i < size; i++) {
        if (data[i] > 0) {
            entropy -= data[i] * log2(data[i]);
        }
    }
    return entropy;
}

void calculateARCoefficients(float* input, float* output, int size, int order) {
    for (int i = 0; i < order; i++) {
        output[i] = 0;
        for (int j = 0; j < size - i - 1; j++) {
            output[i] += input[j] * input[j + i + 1];
        }
        output[i] /= size - i - 1;
    }
}

void calculateCorrelation(float* x, float* y, float* z, float* result, int size) {
    float meanX = calculateMean(x, size);
    float meanY = calculateMean(y, size);
    float meanZ = calculateMean(z, size);

    float sumXY = 0, sumXZ = 0, sumYZ = 0;
    float sumX2 = 0, sumY2 = 0, sumZ2 = 0;

    for (int i = 0; i < size; i++) {
        float dx = x[i] - meanX;
        float dy = y[i] - meanY;
        float dz = z[i] - meanZ;

        sumXY += dx * dy;
        sumXZ += dx * dz;
        sumYZ += dy * dz;

        sumX2 += dx * dx;
        sumY2 += dy * dy;
        sumZ2 += dz * dz;
    }
    result[0] = sumXY / sqrt(sumX2 * sumY2);
    result[1] = sumXZ / sqrt(sumX2 * sumZ2);
    result[2] = sumYZ / sqrt(sumY2 * sumZ2);
}
#endif
