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

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Convert target variable to categorical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
input_dim = X.shape[1]  # Number of input features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

number_of_epochs = 100

# Step 2: Implement the Bayesian MLP
def create_bayesian_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=256, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.5),  # Add dropout layer for uncertainty
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Add dropout layer for uncertainty
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Add dropout layer for uncertainty
        tf.keras.layers.Dense(units=5, activation='softmax')  # Output layer with 5 classes
    ])
    return model

bayesian_mlp = create_bayesian_model(input_dim)

# Step 3: Update the loss function for the Bayesian MLP
def bayesian_loss(y_true, y_pred, sigma=1.0):
    """Custom loss function for Bayesian Neural Network"""
    y_true = tf.cast(y_true, tf.float32)#................................................................................................
    # Negative log-likelihood loss
    log_likelihood = tf.reduce_sum(-0.5 * np.log(2 * np.pi * sigma**2) - (tf.square(y_true - y_pred) / (2 * sigma**2)))
    # Add regularization term for weight uncertainty
    regularizer = tf.reduce_sum(tf.abs(y_pred))
    return -(log_likelihood + regularizer)

# Compile and train the Bayesian MLP
bayesian_mlp.compile(optimizer='adam', loss=bayesian_loss)
bayesian_mlp.fit(X_train, y_train, epochs=number_of_epochs, batch_size=32)

# Step 4: Implement Monte Carlo Dropout during prediction for the Bayesian MLP
def monte_carlo_dropout_prediction(model, X, n_samples=50):
    """Perform Monte Carlo Dropout prediction"""
    y_preds = []
    for _ in range(n_samples):
        y_pred = model.predict(X)
        y_preds.append(y_pred)
    y_preds = np.array(y_preds)
    y_mean = np.mean(y_preds, axis=0)
    y_std = np.std(y_preds, axis=0)
    return y_mean, y_std





# Get the predictions from the Bayesian MLP
train_predictions_mean, train_predictions_std = monte_carlo_dropout_prediction(bayesian_mlp, X_train)
test_predictions_mean, test_predictions_std = monte_carlo_dropout_prediction(bayesian_mlp, X_test)

# Train the Bayesian MLP and store the loss per epoch
train_loss_mlp = []
test_loss_mlp = []
for epoch in range(number_of_epochs):
    history = bayesian_mlp.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
    train_loss_mlp.append(history.history['loss'][0])
    test_loss_mlp.append(history.history['val_loss'][0])
# Convert predictions to class labels
train_predicted_labels = np.argmax(train_predictions_mean, axis=1)
test_predicted_labels = np.argmax(test_predictions_mean, axis=1)

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