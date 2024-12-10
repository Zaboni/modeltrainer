# Network Traffic Analysis with Deep Learning

This project implements a deep learning model for detecting network attacks using the CICIDS2017 dataset. The system processes network traffic data and classifies it as either benign or malicious.

## Overview

The project processes network traffic data from multiple days (Monday, Tuesday, Thursday, and Friday) containing various types of network activities including normal traffic and different types of attacks (DDoS and Web Attacks).

## Features

- Data preprocessing and cleaning
- Handling of missing values and extreme outliers
- Feature normalization
- Binary classification (Benign vs. Malicious)
- Neural network implementation using TensorFlow
- C-friendly model export capabilities

## Project Structure

```
csvtraining/
├── main.py              # Main processing and training script
├── feature_names.txt    # List of features used in the model
├── network_config.txt   # Neural network architecture configuration
├── model_weights.bin    # Trained model weights in binary format
├── scaler_params.bin    # Feature scaling parameters
├── sample_input.bin     # Sample input for testing
└── cicids2017_cleaned.csv # Preprocessed dataset
```

## Data Processing

The system performs several preprocessing steps:
1. Combines multiple CSV files from different days
2. Removes irrelevant columns (Flow ID, IPs, Ports, etc.)
3. Converts labels to binary (0 = BENIGN, 1 = MALICIOUS)
4. Handles missing values and infinities
5. Removes extreme outliers using IQR method
6. Normalizes features using MinMaxScaler

## Model Architecture

The neural network consists of:
- Input layer: 78 features
- Hidden layer 1: 64 neurons (ReLU activation)
- Dropout layer (20%)
- Hidden layer 2: 32 neurons (ReLU activation)
- Dropout layer (20%)
- Output layer: 1 neuron (Sigmoid activation)

## Model Export

The model is exported in a C-friendly format:
- Weights and biases saved as binary files
- Network architecture saved in text format
- Feature names preserved for reference
- Scaling parameters saved for preprocessing new data

## Usage

To run the training pipeline:

```bash
python main.py
```

The script will:
1. Load and preprocess the data
2. Train the neural network
3. Save the model and necessary parameters
4. Generate a sample prediction

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- tensorflow

## Dataset

This project uses the CICIDS2017 dataset, which includes:
- Monday-WorkingHours.pcap_ISCX.csv
- Tuesday-WorkingHours.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
