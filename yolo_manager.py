from ultralytics import YOLO
import torch
import cv2


class YoloManager:
    def __init__(self):
        # Automatically select device: GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO("yolov8n.pt").to(self.device)

    def preprocess(self, frame):
        # Resize the frame to the model's expected input size
        input_size = 640  # Standard size for YOLOv8
        resized_frame = cv2.resize(frame, (input_size, input_size))

        # Convert to a tensor with BCHW format
        frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # Normalize pixel values to the range [0, 1]
        frame_tensor /= 255.0
        return frame_tensor

    def detect(self, frame):
        # Preprocess the frame
        frame_tensor = self.preprocess(frame)

        # Perform inference
        results = self.model(frame_tensor)

        # Extract detections and move them to CPU
        detections = results[0].boxes.data.cpu().numpy()
        return detections


if __name__ == "__main__":
    ob = YoloManager()
