from flask import Flask, request, jsonify
from tensorflow import keras
from tensorflow.image import resize
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load your Keras model
model_path = "xception_v3_05_0.606.keras"
try:
    model = keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)  # Stop execution, as no model was loaded


# Assume you have the class names somehow,
#  either from training data or manually defined.
# You can replace this part with however your class names are stored.

#  Method 1: If your model has an attribute with class names
if hasattr(model, 'class_names') and model.class_names:
    classes = model.class_names #  Use class_names from the model
else:
    # Method 2: Manual list (if your model doesn't store them)
    classes = [
      "arabic",
      "art_deco",
      "baroque",
      "beaux_arts",
      "brutalist",
      "byzantine",
      "colonial",
      "gothic",
      "modernist",
      "palladian",
      "postmodern"
    ] # Replace with your class names.

def preprocess_image(image_file):
    """Preprocesses an image for model input."""
    image = Image.open(image_file).convert("RGB")
    image = np.array(image) / 255.0
    image = resize(image, [150, 150])
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    """Handles prediction requests."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]

    try:
        img_arr = preprocess_image(image_file)
        prediction = model.predict(img_arr)[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = classes[predicted_class_index] # Use the class_name
        named_probabilities = dict(zip(classes, prediction.tolist()))  # Combine names and probabilities
        return jsonify({
          "predicted_class": predicted_class_name,
          "probabilities": named_probabilities
          }) # Return class name and the probabilities
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)