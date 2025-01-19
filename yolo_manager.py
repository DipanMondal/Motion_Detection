from ultralytics import YOLO
import torch
import cv2
import os


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
        print("Using YoloManager model")
        # Preprocess the frame
        frame_tensor = self.preprocess(frame)

        # Perform inference
        results = self.model(frame_tensor)

        # Extract detections and move them to CPU
        detections = results[0].boxes.data.cpu().numpy()
        return detections


class ForkliftYoloManager:
    def __init__(self):
        self.device = 'cpu'  # or 'cuda' if you have a GPU
        model_path = r"C:\Users\sabar\Downloads\Motion_Detection\Motion_Detection\forklift_yolov8n .pt"  # Update this path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model file {model_path} does not exist.")
        
        self.model = YOLO(model_path).to(self.device)

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
        print("Using ForkliftYoloManager model")
        # Preprocess the frame
        frame_tensor = self.preprocess(frame)

        # Perform inference
        results = self.model(frame_tensor)

        # Extract detections and move them to CPU
        detections = results[0].boxes.data.cpu().numpy()
        return detections


if __name__ == "__main__":
    ob = YoloManager()