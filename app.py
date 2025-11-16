import io
import logging
import traceback
from flask import Flask, request, render_template_string, redirect, url_for
from werkzeug.utils import secure_filename
import joblib
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

# Configuration
MODEL_PATH = "model/savedmodel.pth"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 4 * 1024 * 1024  # 4 MB max upload

# Flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy-loaded model holder
clf = None

def get_model():
    global clf
    if clf is None:
        logger.info("Loading model from %s", MODEL_PATH)
        model_data = joblib.load(MODEL_PATH)
        if isinstance(model_data, dict) and "model" in model_data:
            clf = model_data["model"]
        else:
            clf = model_data
        logger.info("Model loaded successfully")
    return clf

HTML = """
<!doctype html>
<title>Olivetti Face Classifier</title>
<h1>Upload an image (64x64 grayscale will work best)</h1>
<form action="/predict" method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if pred is defined %}
  <h2>Predicted class: {{ pred }}</h2>
{% endif %}
"""

def allowed_file(filename: str) -> bool:
    if not filename:
        return False
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_fileobj(file_stream):
    img = Image.open(file_stream).convert("L")
    img = img.resize((64, 64))
    arr = np.array(img).astype("float32") / 255.0
    return arr.flatten().reshape(1, -1)

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return "No file part in the request", 400

        file = request.files["file"]
        filename = secure_filename(file.filename)

        if filename == "":
            return "No selected file", 400

        if not allowed_file(filename):
            return "Unsupported file type. Allowed: png, jpg, jpeg", 400

        try:
            arr = preprocess_image_fileobj(file.stream)
        except UnidentifiedImageError:
            return "Uploaded file is not a valid image", 400
        except Exception as e:
            logger.exception("Failed during image preprocessing")
            return f"Image preprocessing error: {str(e)}", 500

        try:
            model = get_model()
        except Exception:
            logger.exception("Failed to load model")
            return "Internal Server Error: model load failed (check server logs)", 500

        try:
            pred = model.predict(arr)[0]
            pred = int(pred)
        except Exception:
            logger.exception("Prediction failed")
            return "Internal Server Error: prediction failed (check server logs)", 500

        return render_template_string(HTML, pred=pred)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Unhandled exception:\n%s", tb)
        return "Internal Server Error (see server logs)", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
