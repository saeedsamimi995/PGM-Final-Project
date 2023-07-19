# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from pgm_final_project.config import load_config_from_file

# Step 1: Preparing the data
config = load_config_from_file()
data = pd.read_csv(config.resolve_data_path())
df = pd.DataFrame(data)
df

X = df.iloc[:, :-1]  
y = df.iloc[:, -1]  
# Convert target variable to categorical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
input_dim = X.shape[1]  # Number of input features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

number_of_epochs = 100
autoencoder = tf.keras.Sequential([
    tf.keras.layers.Dense(units=256, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=input_dim, activation='sigmoid')
])

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
autoencoder.fit(X_train, y_train, epochs=number_of_epochs, batch_size=32)

encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[2].output)
encoded_train = encoder.predict(X_train)
encoded_test = encoder.predict(X_test)

#train the MLP:
mlp = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(64,)),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=5, activation='softmax')
])

# Compile and train the MLP
mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mlp.fit(encoded_train, y_train, epochs=number_of_epochs, batch_size=32, validation_data=(encoded_test, y_test))

# Train the MLP and store the loss per epoch
train_loss_mlp = []
test_loss_mlp = []

for epoch in range(number_of_epochs):
    history = mlp.fit(encoded_train, y_train, epochs=1, batch_size=32, validation_data=(encoded_test, y_test))
    train_loss_mlp.append(history.history['loss'][0])
    test_loss_mlp.append(history.history['val_loss'][0])

# Get predictions from the MLP
train_predictions = mlp.predict(encoded_train)
test_predictions = mlp.predict(encoded_test)

# Convert predictions to class labels
train_predicted_labels = np.argmax(train_predictions, axis=1)
test_predicted_labels = np.argmax(test_predictions, axis=1)

# Calculate the confusion matrices
train_confusion_mat = confusion_matrix(y_train, train_predicted_labels)
test_confusion_mat = confusion_matrix(y_test, test_predicted_labels)

# Plotting
plt.figure(figsize=(12, 8))

# Plot train error
plt.subplot(2, 2, 1)
plt.plot(range(number_of_epochs), train_loss_mlp, label='Train MLP Error')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Errors')
plt.legend()

## Plot test error
plt.subplot(2, 2, 3)
plt.plot(range(number_of_epochs), test_loss_mlp, label='Test MLP Error')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Errors')
plt.legend()

# Plot the confusion matrix for training set
labels = ['normal', 'luma', 'lumb', 'her2', 'basal']

plt.subplot(2, 2, 2)
plt.imshow(train_confusion_mat, cmap='Blues')
plt.title('Confusion Matrix (Training Set)')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(np.arange(5), labels)
plt.yticks(np.arange(5), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, train_confusion_mat[i, j], ha='center', va='center', color='black')

# Plot the confusion matrix for testing set
plt.subplot(2, 2, 4)
plt.imshow(test_confusion_mat, cmap='Blues')
plt.title('Confusion Matrix (Testing Set)')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(np.arange(5), labels)
plt.yticks(np.arange(5), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, test_confusion_mat[i, j], ha='center', va='center', color='black')

plt.tight_layout()
plt.show()