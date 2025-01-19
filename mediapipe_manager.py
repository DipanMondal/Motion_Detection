import mediapipe as mp
import json
import cv2
import numpy as np


class MedipapipeManager:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        read_it = open("skeleton.json","r")
        self.skeleton_indices = json.load(read_it)

    def getSkeletons(self,detections,frame):
        coordinates = []
        skeletons = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0:  # Class 0 corresponds to "person"
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                human_frame = frame[y1:y2, x1:x2]  # Crop bounding box for the detected person

                # Process the cropped frame with MediaPipe
                rgb_human_frame = cv2.cvtColor(human_frame, cv2.COLOR_BGR2RGB)
                results_pose = self.pose.process(rgb_human_frame)

                if results_pose.pose_landmarks:
                    landmarks = results_pose.pose_landmarks.landmark

                    skeleton = [0.0] * 99
                    for i, lm in enumerate(landmarks):
                        idx = i * 3
                        skeleton[idx], skeleton[idx + 1], skeleton[idx + 2] = lm.x, lm.y, lm.z

                    skeleton = np.array(skeleton)

                    skeletons.append(skeleton)
                    coordinates.append([(x1,y1),(x2,y2)])

        return coordinates,skeletons


if __name__ == '__main__':
    ob = MedipapipeManager()