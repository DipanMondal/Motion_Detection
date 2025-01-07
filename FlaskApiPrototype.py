from yolo_manager import YoloManager
from mediapipe_manager import MedipapipeManager
from motion import MOTION
import cv2
import numpy as np
from flask import *
import io


ym = YoloManager()
mm = MedipapipeManager()
mo = MOTION("skeleton.json",
            r"data/custom_skeleton_pose_data/models/svm_motion_model1.pkl",
            r"data/custom_skeleton_pose_data/models/scaler.pkl")

app = Flask(__name__)


@app.route('/')
def main():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
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

                if frame is None:
                    return jsonify({"error": "Unable to process the image file"}), 400

                # Process the image (optional)
                img_shape = frame.shape  # Example: (height, width, channels)

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


if __name__ == '__main__':
    app.run(debug=True)


