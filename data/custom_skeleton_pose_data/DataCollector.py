import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


class DataCollection:
    def __init__(self, cam=0):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils
        self.data = []
        self.labels = []
        self.cap = cv2.VideoCapture(cam)

    def save(self, path, file_name):
        fold = path + '/' + file_name
        df = pd.DataFrame(self.data)
        df['label'] = self.labels
        df.to_csv(fold, index=False)

    def append(self, path, file_name):
        fold = path+'/'+file_name
        df = pd.read_csv(fold)
        df2 = pd.DataFrame(self.data)
        df2['label'] = self.labels
        DF = pd.concat([df, df2],axis=0)
        DF.to_csv(fold, index=False)

    def run(self, label, records=1000):
        selected_columns = [33, 34, 35, 36, 37, 38, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86]

        while self.cap.isOpened() and records > 0:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                skeleton = [0.0] * 99
                for i, lm in enumerate(landmarks):
                    idx = i * 3
                    skeleton[idx], skeleton[idx + 1], skeleton[idx + 2] = lm.x, lm.y, lm.z

                features = []
                for idx in selected_columns:
                    features.append(skeleton[idx])

                # Append skeleton data
                self.data.append(features)
                self.labels.append(label)  # Replace with the action being performed

                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Skeleton Extraction", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print("remaining : ",records)
            records -= 1

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
