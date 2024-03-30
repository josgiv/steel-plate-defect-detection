import os
from PIL import Image
import numpy as np
import tensorflow as tf


# Load your model here
def load_model():
    # Load your trained model here
    # Example:
    # model = tf.keras.models.load_model("your_model_path")
    pass


model = load_model()


# Function to preprocess and predict image
def predict_image(image_path):
    # Preprocess the image
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values

    # Predict using the loaded model
    prediction = model.predict(np.expand_dims(img_array, axis=0))

    # Convert prediction to output format
    # Example: convert prediction probabilities to one-hot encoded labels
    output_labels = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
    predicted_label = output_labels[np.argmax(prediction)]

    return predicted_label


@app.route("/upload", methods=["POST"])
def upload():
    # Handle file upload
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"})

    # Save uploaded image
    image_path = os.path.join("uploads", file.filename)
    file.save(image_path)

    # Predict defect from uploaded image
    defect_prediction = predict_image(image_path)

    # Return prediction as response
    return jsonify({"defect_prediction": defect_prediction})
