import numpy as np
import dlib

"""
    Wrapper class for dlib's landmark detector.
    This class is used to detect 68 landmarks on a face image.
    To donwnload the detector model, visit: http://dlib.net/files/mmod_human_face_detector.dat.bz2
    To download the predictor model, visit: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""
class DlibLandmarkDetector:
    detector = None
    predictor = None
    def __init__(self, path_to_dlib_detector_model, path_to_dlib_predictor_model: str):
        self.detector = dlib.cnn_face_detection_model_v1(path_to_dlib_detector_model)
        self.predictor = dlib.shape_predictor(path_to_dlib_predictor_model)

    def detect_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        Detects 68 landmarks on a face image.
        :param image: Face image HxWx3 (RGB ordering)
        :return: Detected landmarks
        """
        boxes = self.detector(image, 1)
        assert len(boxes) == 1, "Expected exactly one face in the image, but found {}".format(len(boxes))
        box = boxes[0]
        prediction = self.predictor(image, box)
        landmarks = np.array([[p.x, p.y] for p in prediction.parts()])
        assert len(landmarks) == 68, "Expected 68 landmarks, but found {}".format(landmarks.shape)
        return landmarks
