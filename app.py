from flask import Flask, jsonify, request
import cv2
import numpy as np
import pickle
from werkzeug.utils import secure_filename
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained number plate recognition model
model = None
MODEL_PATH = "./model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return "Number Plate Recognition Flask Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is included in the request
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if video_file and allowed_file(video_file.filename):
            filename = secure_filename(video_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(file_path)

            # Process the video to extract frames and predict number plates
            recognized_plates = process_video(file_path)
            os.remove(file_path)  # Clean up uploaded file after processing

            return jsonify({'plates': recognized_plates}), 200
        else:
            return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video(video_path):
    """
    Process the video frame by frame to recognize number plates.
    """
    cap = cv2.VideoCapture(video_path)
    recognized_plates = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if the video ends

        # Preprocess the frame (resize, grayscale, etc., as required by your model)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (128, 128))  # Adjust size based on model input
        frame_input = resized_frame.flatten().reshape(1, -1)  # Flatten if the model expects 1D input

        # Predict number plate using the loaded model
        plate_prediction = model.predict(frame_input)

        # Append the result if it’s a new plate
        plate_text = plate_prediction[0]
        if plate_text not in recognized_plates:
            recognized_plates.append(plate_text)

    cap.release()
    return recognized_plates

if __name__ == "__main__":
    # For Render, use host='0.0.0.0' and port as an environment variable
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
