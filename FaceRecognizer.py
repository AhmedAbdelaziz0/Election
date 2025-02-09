import cv2
import numpy as np
from utils.ediffiqa import eDifFIQA
from utils.register import get_database
from utils.utils import resize_with_padding
from abc import abstractmethod

class BaseFaceRecognizer:
    def __init__(self, detection_threshold=0.9, quality_threshold=0.4):
        self.face_detector = cv2.FaceDetectorYN.create(
            "models/face_detection_yunet_2023mar.onnx",
            "",
            [640, 640],
            detection_threshold,
        )
        self.face_recognizer = cv2.FaceRecognizerSF.create(
            "models/face_recognition_sface_2021dec.onnx", ""
        )
        self.model_quality = eDifFIQA(
            modelPath="models/ediffiqa_tiny_jun2024.onnx",
        )

        self.quality_threshold = quality_threshold

    def extract_faces(self, image):
        names = []
        locations = []
        resized_image, scale_factor, padding = resize_with_padding(image, (640, 640))
        _, faces = self.face_detector.detect(resized_image)
        if faces is not None:
            for face_info in faces:
                x, y, w, h = face_info[0:4]
                # Scale the coordinates back to the original image size
                x = int((x - padding[0]) / scale_factor)
                y = int((y - padding[2]) / scale_factor)
                w = int(w / scale_factor)
                h = int(h / scale_factor)
                locations.append((x, y, w, h))
                aligned_face = self.face_recognizer.alignCrop(resized_image, face_info)
                quality = self.model_quality.infer(aligned_face)
                if quality < self.quality_threshold:
                    names.append(f"Face Quality not met: {quality}")
                    continue
                feature = self.face_recognizer.feature(aligned_face)
                name = self.search(feature)
                names.append(name)
        return names, locations

    @abstractmethod
    def search(self, feature):
        raise NotImplementedError


class FaceRecognizer(BaseFaceRecognizer):
    def __init__(
        self,
        detection_threshold=0.9,
        distance_threshold=0.2,
        quality_threshold=0.4,
    ):
        super().__init__(detection_threshold, quality_threshold)
        self.distance_threshold = distance_threshold

        self.features, self.names = get_database(
            "database", "models", quality_threshold
        )

    def search(self, feature):
        distance = (
            1 - np.dot(feature / np.linalg.norm(feature), self.features.T)
        ).squeeze()
        closest = np.argmin(distance)

        return (
            self.names[closest]
            if distance[closest] < self.distance_threshold
            else "Unknown"
        )
