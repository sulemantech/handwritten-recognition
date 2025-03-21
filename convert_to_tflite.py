import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("digits_recognition_cnn.h5")
print("✅ Model loaded successfully!")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
tflite_model_path = "digits_recognition_cnn.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ Model successfully converted to TensorFlow Lite: {tflite_model_path}")
