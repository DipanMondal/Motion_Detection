import cv2
import numpy as np

class OpticalFlow:
    def __init__(self):
        self.prev_gray = None
        self.hsv_mask = None

    def initialize(self, frame):
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.hsv_mask = np.zeros_like(frame)
        self.hsv_mask[..., 1] = 255

    def compute_flow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.hsv_mask[..., 0] = ang * 180 / np.pi / 2
        self.hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(self.hsv_mask, cv2.COLOR_HSV2BGR)
        self.prev_gray = gray
        return rgb_flow
