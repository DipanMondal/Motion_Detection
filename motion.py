import json

import joblib


class MOTION:
    def __init__(self,skeleton_path,model_path,scaler_path):
        read_it = open(skeleton_path, "r")
        self.selected_indices = json.load(read_it)
        self.svm_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def getMotion(self,coordinates,skeletons):
        motions = []

        for coordinate, skeleton in zip(coordinates,skeletons):
            features = skeleton[self.selected_indices]
            features = features.reshape(1, -1)
            features = self.scaler.transform(features)

            motion = self.svm_model.predict(features)[0]

            motions.append(motion)

        return motions


if __name__ == '__main__':
    mo = MOTION("skeleton.json",
                r"data/custom_skeleton_pose_data/models/svm_motion_model1.pkl",
                r"data/custom_skeleton_pose_data/models/scaler.pkl")
