import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the Saved Model
model = tf.keras.models.load_model("digits_recognition_cnn.h5")
print("✅ Model loaded successfully!")

# Load MNIST Test Data
mnist_dataset = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist_dataset.load_data()

# Normalize and Reshape Data
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# Evaluate Model on Test Data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"✅ Model Accuracy on Test Data: {test_acc:.4f}")

# Make Predictions
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Display Some Predictions
num_images = 10  # Change this to show more images
plt.figure(figsize=(10, 5))

for i in range(num_images):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.xlabel(f"Pred: {predicted_labels[i]} (True: {y_test[i]})", color="green" if predicted_labels[i] == y_test[i] else "red")

plt.show()
