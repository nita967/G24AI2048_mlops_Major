# app.py
import os
import io
from flask import Flask, request, render_template_string, redirect, url_for
import joblib
from PIL import Image
import numpy as np

MODEL_PATH = "model/savedmodel.pth"

app = Flask(__name__)
model_data = joblib.load(MODEL_PATH)
clf = model_data['model']

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

def preprocess_image(file_stream):
    img = Image.open(file_stream).convert("L")  # grayscale
    img = img.resize((64,64))
    arr = np.array(img).astype("float32") / 255.0
    flat = arr.flatten().reshape(1, -1)
    return flat

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    f = request.files['file']
    arr = preprocess_image(f.stream)
    pred = clf.predict(arr)[0]
    return render_template_string(HTML, pred=int(pred))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

