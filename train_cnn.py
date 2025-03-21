import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST Dataset
mnist_dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()

# Normalize the Data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN (Adding Channel Dimension)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Define CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, kernel_size=5, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(16, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Save Model
model.save("digits_recognition_cnn.h5")
print("✅ Model saved successfully!")

# Load and Verify Model
loaded_model = tf.keras.models.load_model("digits_recognition_cnn.h5")
test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
print(f"✅ Loaded Model Accuracy: {test_acc:.4f}")

# Make Predictions
predictions = loaded_model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Display First Prediction
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted Label: {predicted_labels[0]}")
plt.show()
