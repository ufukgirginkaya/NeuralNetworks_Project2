# Human Activity Recognition Using Fully Recurrent Neural Networks

## Overview
This project implements a Human Activity Recognition (HAR) system using a fully recurrent neural network (RNN) trained with the Backpropagation Through Time (BPTT) algorithm. The dataset includes sensor data for six different human activities, and the task is to classify these activities based on the time-series sensor inputs.

The activities are:
1. Downstairs motion
2. Jogging motion
3. Sitting motion
4. Standing motion
5. Upstairs motion
6. Walking motion

---

## Dataset
The dataset includes:
- **Training Data**: 3000 samples (500 samples per activity).
- **Test Data**: 600 samples (100 samples per activity).

Each sample is a time series of length 150 units, with measurements from three sensors.

### Data Files
- `trX` and `tstX`: Sensor measurements for training and testing.
- `trY` and `tstY`: Corresponding labels for the training and testing data.

---

## Methodology
### Network Architecture
- **Hidden Layer**:
  - Contains \( N \) neurons.
  - Weight matrices: \( W_{HH} \) (\( N \times N \)) and \( W_{IH} \) (\( N \times (3+1) \)), where \( +1 \) accounts for bias terms.
  - Activation function: Hyperbolic tangent (\( \tanh \)).

- **Output Layer**:
  - Contains 6 neurons for the 6 classes.
  - Weight matrix: \( W_{HO} \) (\( 6 \times (N+1) \)), where \( +1 \) accounts for bias terms.
  - Activation function: Sigmoid.
  - Loss function: Multi-category cross-entropy.

---

## Training and Evaluation
### Hyperparameters
- Learning rate (\( \eta \)): 0.05, 0.1.
- Hidden layer size (\( N \)): 50, 100.
- Mini-batch sizes: 10, 30.
- Epochs: 50.

---

## Usage
### Requirements
- Python 3.8+
- Libraries: `numpy`


