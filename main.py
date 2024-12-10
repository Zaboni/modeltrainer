import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load selected files
monday = pd.read_csv("./csv/Monday-WorkingHours.pcap_ISCX.csv")
tuesday = pd.read_csv("./csv/Tuesday-WorkingHours.pcap_ISCX.csv")
friday_ddos = pd.read_csv("./csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
thursday_web = pd.read_csv("./csv/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")

# Combine datasets
combined = pd.concat([monday, tuesday, friday_ddos, thursday_web], ignore_index=True)
print(f"Combined dataset shape: {combined.shape}")

# Print column names to identify the correct label column
print("\nAvailable columns:")
print(combined.columns.tolist())

# Drop irrelevant columns
columns_to_drop = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
combined = combined.drop(columns=columns_to_drop, errors='ignore')

# Convert labels to binary (0 = BENIGN, 1 = MALICIOUS)
label_column = ' Label' if ' Label' in combined.columns else 'label' if 'label' in combined.columns else 'Label'
combined[label_column] = combined[label_column].apply(lambda x: 0 if x == 'BENIGN' else 1)

# Handle missing values and infinities
combined = combined.replace([np.inf, -np.inf], np.nan)
combined = combined.fillna(0)

# Replace extreme values with reasonable limits
numerical_columns = combined.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    # Calculate percentiles while ignoring infinite values
    q1 = combined[col].quantile(0.25)
    q3 = combined[col].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 3 * iqr
    lower_limit = q1 - 3 * iqr

    # Cap extreme values
    combined[col] = combined[col].clip(lower_limit, upper_limit)

# Normalize numerical features
scaler = MinMaxScaler()
combined[numerical_columns] = scaler.fit_transform(combined[numerical_columns])

# Save the cleaned dataset
combined.to_csv("cicids2017_cleaned.csv", index=False)
print(f"Preprocessed dataset shape: {combined.shape}")

# Save the scaler parameters in binary format
with open('scaler_params.bin', 'wb') as f:
    scaler.min_.astype(np.float32).tofile(f)
    scaler.scale_.astype(np.float32).tofile(f)
print("Scaler parameters saved in binary format as scaler_params.bin")

# Save feature names for reference
feature_names = numerical_columns.tolist()
with open('feature_names.txt', 'w') as f:
    for feature in feature_names:
        f.write(f"{feature}\n")
print("Feature names saved to feature_names.txt")

# Load preprocessed dataset
data = pd.read_csv("cicids2017_cleaned.csv")

# Features and labels
X = data.drop(label_column, axis=1)  # Features
y = data[label_column]  # Labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile and train model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                   epochs=10,
                   batch_size=32,
                   validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")
# Save weights in C-friendly binary format
weights = []
biases = []

# Get weights and biases from each dense layer
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        layer_weights = layer.get_weights()
        weights.append(layer_weights[0].astype(np.float32))  # Weights
        biases.append(layer_weights[1].astype(np.float32))   # Biases

# Save network architecture info
with open('network_config.txt', 'w') as f:
    f.write(f"Input size: {X.shape[1]}\n")
    f.write("Layer sizes: 64,32,1\n")
    f.write("Activation functions: relu,relu,sigmoid\n")

# Save weights and biases in binary format
with open('model_weights.bin', 'wb') as f:
    # Save weights
    for w in weights:
        w.tofile(f)
    # Save biases
    for b in biases:
        b.tofile(f)

print("Model weights and architecture saved in C-friendly format")

# Test prediction with sample data
test_sample = X_test.iloc[0:1]
prediction = model.predict(test_sample)
print(f"Sample prediction: {prediction[0][0]:.4f}")

# Save a sample input for testing
test_sample.astype(np.float32).to_numpy().tofile('sample_input.bin')
print("Sample input saved for C testing")