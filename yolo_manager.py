from ultralytics import YOLO


class YoloManager:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        results = self.model(frame)
        detections = results[0].boxes.data.cpu().numpy()
        return detections


if __name__ == "__main__":
    ob = YoloManager()