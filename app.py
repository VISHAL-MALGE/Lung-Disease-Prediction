from flask import Flask, render_template, request, jsonify
import os
from predict import predict_disease_from_image  
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to fix cross-origin issues

UPLOAD_FOLDER = 'uploads'  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    # Call your ML prediction function ✅
    result = predict_disease_from_image(image_path)

    print("Final JSON Response:", result)

    return jsonify(result)  # ✅ Returns a list of detected diseases

if __name__ == '__main__':
    app.run(debug=True)
