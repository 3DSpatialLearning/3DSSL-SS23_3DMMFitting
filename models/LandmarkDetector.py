import numpy as np
from torch import nn
import dlib

"""
    Wrapper class for dlib's landmark detector.
    This class is used to detect 68 landmarks on a face image.
    To download the detector model, visit: http://dlib.net/files/mmod_human_face_detector.dat.bz2 or use default one
    To download the predictor model, visit: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    After having that donwloaded, place it in data/checkpoints/
"""


class DlibLandmarkDetector(nn.Module):
    def __init__(self, path_to_dlib_predictor_model: str = "data/checkpoints/shape_predictor_68_face_landmarks.dat"):
        super(DlibLandmarkDetector, self).__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path_to_dlib_predictor_model)

    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Detects 68 landmarks on a face image.
        :param image: Face image HxWx3 (RGB ordering)
        :return: Detected landmarks
        """
        if type(image) is not np.ndarray:
            boxes = self.detector(image.squeeze().numpy(), 1)
        else:
            boxes = self.detector(image, 1)
        assert len(boxes) == 1, "Expected exactly one face in the image, but found {}".format(len(boxes))
        box = boxes[0]
        prediction = self.predictor(image, box)
        landmarks = np.array([[p.x, p.y] for p in prediction.parts()])
        assert len(landmarks) == 68, "Expected 68 landmarks, but found {}".format(landmarks.shape)
        return landmarks
