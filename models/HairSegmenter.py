import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HairSegmenter:
    """
    A class that performs hair segmentation on images using the MediaPipe framework.

    Install the pre-trained hair segmentation model from https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite
    Then place it under data/hair_segmentation

    Example usage:
        segmenter = HairSegmenter()
        hair_mask = segmenter.segment(image)  # Perform hair segmentation
    """

    def __init__(self, model_path: str = "data/hair_segmentation/hair_segmenter.tflite"):
        """
        Initializes the HairSegmenter class.

        Args:
            model_path (str): Path to the hair segmentation model (default: "data/hair_segmentation/hair_segmentation.tflite").
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def segment(self, input_image: np.ndarray, visualize: bool = False) -> np.ndarray:
        """
        Performs hair segmentation on the input image.

        Args:
            img (np.ndarray): Input image as a NumPy array with shape (width, height, 3) (RGB image) or (width, height, 4) (RGBA image).
            visualize (bool): Flag to visualize the hair mask (default: False).

        Returns:
            np.ndarray: Hair mask as a binary NumPy array with shape (width, height).
        """
        img_shape_len = len(input_image.shape)
        assert img_shape_len == 3, "Expected len(img.shape) == 3 (width, height, channel), but found {}.".format(
            img_shape_len)
        img_channel = input_image.shape[2]
        assert img_channel == 3 or img_channel == 4, "Expected image to have 3 or 4 channels, but found {}".format(
            img_channel)

        if input_image.shape[2] == 3:  # Check if image has 3 channels
            rgba_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGBA)
            rgba_image[:, :, 3] = 255
        else:
            rgba_image = input_image

        # create MP Image object from numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba_image)

        # Retrieve the masks for the segmented image
        segmentation_result = self.segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask
        mask = category_mask.numpy_view()

        if (visualize):
            rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            rgb_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
            debug_image = cv2.addWeighted(rgb_image, 0.5, rgb_mask * 255, 0.5, 0)
            cv2.imshow('Hair Mask', debug_image)
            cv2.waitKey(0)

        return mask
