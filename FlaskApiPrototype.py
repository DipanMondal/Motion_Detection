from yolo_manager import YoloManager, ForkliftYoloManager
from mediapipe_manager import MedipapipeManager
from motion import MOTION
from optical_flow import OpticalFlow
import cv2
import numpy as np
from flask import *
import io
import os
import uuid
import socket


UPLOAD_FOLDER = 'processed_files'
ym = YoloManager()
mm = MedipapipeManager()
mo = MOTION("skeleton.json",
            r"data/custom_skeleton_pose_data/models/svm_motion_model1.pkl",
            r"data/custom_skeleton_pose_data/models/scaler.pkl")
fym = ForkliftYoloManager()

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


@app.route('/upload_forklift_image', methods=['POST'])
def upload_forklift_image():
    if request.method == 'POST':
        if 'forklift_image' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['forklift_image']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file:
            try:
                # Read the image directly from the file object
                file_bytes = np.frombuffer(file.read(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                frame = cv2.resize(frame, (640, 640))

                if frame is None:
                    return jsonify({"error": "Unable to process the image file"}), 400

                # Detect forklifts using YOLO
                detections = fym.detect(frame)

                # Annotate the bounding box
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Encode the frame to an in-memory file
                _, buffer = cv2.imencode('.png', frame)
                file_stream = io.BytesIO(buffer)

                # Return the annotated frame as a downloadable image
                return send_file(
                    file_stream,
                    mimetype='image/png',
                    as_attachment=True,
                    download_name='annotated_forklift_frame.png'
                )

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        return jsonify({"error": "Unknown error occurred"}), 500


@app.route('/upload_forklift_video', methods=['POST'])
def upload_forklift_video():
    if 'forklift_video' not in request.files:
        return "No file part in the request", 400

    file = request.files['forklift_video']

    if file.filename == '':
        return "No file selected", 400

    if file:
        try:
            # Save the uploaded video temporarily
            video_filename = f"{uuid.uuid4().hex}.mp4"
            input_video_path = os.path.join(UPLOAD_FOLDER, video_filename)
            file.save(input_video_path)

            # Process the video
            output_video_filename = f"processed_forklift_{video_filename}"
            output_video_path = os.path.join(UPLOAD_FOLDER, output_video_filename)

            # OpenCV video processing
            cap = cv2.VideoCapture(input_video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            # Initialize variables for optical flow
            ret, prev_frame = cap.read()
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            hsv_mask = np.zeros_like(prev_frame)
            hsv_mask[..., 1] = 255

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect forklifts using YOLO
                detections = fym.detect(frame)

                # Detect humans using YOLO
                human_detections = ym.detect(frame)

                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, "Forklift", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                for human in human_detections:
                    hx1, hy1, hx2, hy2, hconf, hcls = human
                    if int(hcls) == 0:  # Class 0 corresponds to "person"
                        cv2.rectangle(frame, (int(hx1), int(hy1)), (int(hx2), int(hy2)), (255, 0, 0), 2)
                        cv2.putText(frame, "Person", (int(hx1), int(hy1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Collision avoidance: Check proximity to humans
                for forklift in detections:
                    fx1, fy1, fx2, fy2, fconf, fcls = forklift
                    for human in human_detections:
                        hx1, hy1, hx2, hy2, hconf, hcls = human
                        if int(hcls) == 0:  # Class 0 corresponds to "person"
                            # Calculate distance between forklift and human
                            forklift_center = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)
                            human_center = ((hx1 + hx2) / 2, (hy1 + hy2) / 2)
                            distance = np.linalg.norm(np.array(forklift_center) - np.array(human_center))

                            # Issue alert if too close
                            if distance < 100:  # Threshold distance in pixels
                                cv2.putText(frame, "Alert: Proximity Warning!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Optical flow analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv_mask[..., 0] = ang * 180 / np.pi / 2
                hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb_flow = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

                # Overlay optical flow on the frame
                frame = cv2.addWeighted(frame, 0.7, rgb_flow, 0.3, 0)

                # Write the annotated frame to the output video
                out.write(frame)
                prev_gray = gray

            # Release resources
            cap.release()
            out.release()

            # Render the download page for the processed video
            return render_template("video_download.html", filename=output_video_filename)

        except Exception as e:
            return str(e), 500

    return "Unknown error occurred", 500


@app.route('/upload_combined_image', methods=['POST'])
def upload_combined_image():
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
                frame = cv2.resize(frame, (640, 640))

                if frame is None:
                    return jsonify({"error": "Unable to process the image file"}), 400

                # Detect humans using YOLO
                human_detections = ym.detect(frame)
                # Detect forklifts using YOLO
                forklift_detections = fym.detect(frame)

                # Annotate the bounding boxes
                for det in human_detections:
                    x1, y1, x2, y2, conf, cls = det
                    if int(cls) == 0:  # Class 0 corresponds to "person"
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, "Person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                for det in forklift_detections:
                    x1, y1, x2, y2, conf, cls = det
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, "Forklift", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Encode the frame to an in-memory file
                _, buffer = cv2.imencode('.png', frame)
                file_stream = io.BytesIO(buffer)

                # Return the annotated frame as a downloadable image
                return send_file(
                    file_stream,
                    mimetype='image/png',
                    as_attachment=True,
                    download_name='annotated_combined_frame.png'
                )

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        return jsonify({"error": "Unknown error occurred"}), 500


@app.route('/upload_combined_video', methods=['POST'])
def upload_combined_video():
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
            output_video_filename = f"processed_combined_{video_filename}"
            output_video_path = os.path.join(UPLOAD_FOLDER, output_video_filename)

            # OpenCV video processing
            cap = cv2.VideoCapture(input_video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            # Initialize optical flow
            optical_flow = OpticalFlow()
            ret, prev_frame = cap.read()
            optical_flow.initialize(prev_frame)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect humans using YOLO
                human_detections = ym.detect(frame)
                # Detect forklifts using YOLO
                forklift_detections = fym.detect(frame)

                # Annotate the bounding boxes
                for det in human_detections:
                    x1, y1, x2, y2, conf, cls = det
                    if int(cls) == 0:  # Class 0 corresponds to "person"
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, "Person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                for det in forklift_detections:
                    x1, y1, x2, y2, conf, cls = det
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, "Forklift", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Optical flow analysis
                rgb_flow = optical_flow.compute_flow(frame)

                # Overlay optical flow on the frame
                frame = cv2.addWeighted(frame, 0.7, rgb_flow, 0.3, 0)

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
    try:
        app.run(debug=True)
    except socket.error as e:
        print(f"Socket error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")



