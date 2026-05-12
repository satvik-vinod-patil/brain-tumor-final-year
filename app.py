import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

app = Flask(__name__)

# Class labels mapping
CLASS_LABELS = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor",
    2: "Normal",
    3: "Pituitary Tumor"
}

CLASS_DESCRIPTIONS = {
    0: "A type of tumor that occurs in the brain and spinal cord.",
    1: "A tumor that arises from the meninges surrounding the brain.",
    2: "No tumor detected in the MRI scan.",
    3: "A tumor that develops in the pituitary gland."
}

# Load model at startup
model = None

def load_model():
    global model
    try:
        import tensorflow as tf
        model_path = os.path.join(os.path.dirname(__file__), 'global_round_11.h5')
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully.")
        else:
            print("⚠️ Model file not found.")
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")

def preprocess_image(image_bytes):
    import tensorflow as tf
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((380, 380), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image_bytes):
    img_array = preprocess_image(image_bytes)
    predictions = model.predict(img_array, verbose=0)[0]
    class_idx = int(np.argmax(predictions))
    confidence = float(predictions[class_idx]) * 100
    return {
        "class_index": class_idx,
        "label": CLASS_LABELS[class_idx],
        "description": CLASS_DESCRIPTIONS[class_idx],
        "confidence": round(confidence, 2),
        "all_probabilities": {
            CLASS_LABELS[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(predictions))
        }
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({"error": "Invalid file type. Please upload an image."}), 400

    try:
        image_bytes = file.read()
        result = predict(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    load_model()
    print("🚀 Starting Brain Tumor Detection server at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
