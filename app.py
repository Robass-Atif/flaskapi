from flask import Flask, jsonify, request
import cv2
import time
import requests
import numpy as np
from werkzeug.utils import secure_filename
import os

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def recognize_plate(image_data):
    url = "https://api.platerecognizer.com/v1/plate-reader/"
    headers = {
        "Authorization": "Token 2c640e797a7e5a17bc6e523c63cfcee58aec7274"  # Replace with your Plate Recognizer API token
    }
    files = {'upload': image_data}  # Send image data to the API

    retries = 3
    for i in range(retries):
        try:
            response = requests.post(url, files=files, headers=headers)
            if response.status_code == 201:
                result = response.json()
                if "results" in result and result["results"]:
                    highest_score_plate = max(result["results"], key=lambda x: x["score"])
                    plate = highest_score_plate["plate"]
                    score = highest_score_plate["score"]
                    return {"plate": plate, "score": score}
                break
        except requests.exceptions.RequestException as e:
            print(f"Error occurred: {e}")
            time.sleep(2)  # Retry after 2 seconds
    return None

# Route for video prediction
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

            # Process the video to extract frames and recognize plates
            recognized_plates = process_video(file_path)
            os.remove(file_path)  # Clean up uploaded file after processing

            return jsonify({'plates': recognized_plates}), 200
        else:
            return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Function to process the video and send frames to the Plate Recognizer API
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    recognized_plates = set()
    frame_skip = 60  # Process every 50th frame (can adjust as needed)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if the video ends

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Convert the frame to JPEG format
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_data = img_encoded.tobytes()  # Convert to byte data

        # Send frame to the Plate Recognizer API
        plate_info = recognize_plate(img_data)

        if plate_info:
            plate_number = plate_info["plate"]
            score = plate_info["score"]
            
            # Check if plate is already recognized to avoid duplicates
            if plate_number not in recognized_plates and score > 0.9:
                recognized_plates.add(plate_number)
                print(plate_number, score)

                # You can add the plate score here if needed (plate_number and score)
                # For example, to display plate and score on the frame
                # cv2.putText(frame, f"{plate_number} ({score})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cap.release()
    return [{"plate": plate, "score": score} for plate, score in zip(recognized_plates, [1]*len(recognized_plates))]  # Adjust to include score

# Run Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
