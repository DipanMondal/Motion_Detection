from yolo_manager import YoloManager
from mediapipe_manager import MedipapipeManager
from motion import MOTION
import cv2
import numpy as np
from flask import *
import io
import os
import uuid


UPLOAD_FOLDER = 'processed_files'
ym = YoloManager()
mm = MedipapipeManager()
mo = MOTION("skeleton.json",
            r"data/custom_skeleton_pose_data/models/svm_motion_model1.pkl",
            r"data/custom_skeleton_pose_data/models/scaler.pkl")

app = Flask(__name__)


@app.route('/')
def main():
    return render_template("index.html")


@app.route('/upload_image', methods=['POST'])
def success():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file:
            try:
                # Read the image directly from the file object
                file_bytes = np.frombuffer(file.read(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                frame = cv2.resize(frame,(640,640))

                if frame is None:
                    return jsonify({"error": "Unable to process the image file"}), 400

                # Process the image (optional)
                img_shape = frame.shape  # Example: (height, width, channels)
                print(img_shape)

                # Detect humans using YOLO
                detections = ym.detect(frame)
                # mediapipe detection
                coordinates, skeletons = mm.getSkeletons(detections, frame)
                # motion detection
                motions = mo.getMotion(coordinates, skeletons)

                # Annotate the bounding box and motion label
                for cords, motion in zip(coordinates, motions):
                    cv2.rectangle(frame, cords[0], cords[1], (0, 255, 0), 2)
                    cv2.putText(frame, motion, (cords[0][0], cords[0][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (58, 125, 255), 2)

                # Encode the frame to an in-memory file
                _, buffer = cv2.imencode('.png', frame)
                file_stream = io.BytesIO(buffer)

                # Return the annotated frame as a downloadable image
                return send_file(
                    file_stream,
                    mimetype='image/png',
                    as_attachment=True,
                    download_name='annotated_frame.png'
                )

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        return jsonify({"error": "Unknown error occurred"}), 500


# Route to handle video upload and processing
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No file part in the request", 400

    file = request.files['video']

    if file.filename == '':
        return "No file selected", 400

    if file:
        try:
            # Save the uploaded video temporarily
            video_filename = f"{uuid.uuid4().hex}.mp4"
            input_video_path = os.path.join(UPLOAD_FOLDER, video_filename)
            file.save(input_video_path)

            # Process the video
            output_video_filename = f"processed_{video_filename}"
            output_video_path = os.path.join(UPLOAD_FOLDER, output_video_filename)

            # OpenCV video processing
            cap = cv2.VideoCapture(input_video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform frame analysis (replace this with your custom logic)
                # Example: YOLO, Mediapipe, motion detection
                detections = ym.detect(frame)
                coordinates, skeletons = mm.getSkeletons(detections, frame)
                motions = mo.getMotion(coordinates, skeletons)

                for cords, motion in zip(coordinates, motions):
                    cv2.rectangle(frame, cords[0], cords[1], (0, 255, 0), 2)
                    cv2.putText(frame, motion, (cords[0][0], cords[0][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (58, 125, 255), 2)

                # Write the annotated frame to the output video
                out.write(frame)

            # Release resources
            cap.release()
            out.release()

            # Render the download page for the processed video
            return render_template("video_download.html", filename=output_video_filename)

        except Exception as e:
            return str(e), 500

    return "Unknown error occurred", 500


# Route to serve the processed file for download
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)


