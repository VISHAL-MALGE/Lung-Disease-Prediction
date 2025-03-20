from flask import Flask, render_template, request
import os

app = Flask(__name__)

# Ensure the uploads folder exists
UPLOAD_FOLDER = "webApp/upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == '':
        return "No selected file", 400

    # Save file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    return f"✅ File '{file.filename}' uploaded successfully!"

if __name__ == "__main__":
    app.run(debug=True)
