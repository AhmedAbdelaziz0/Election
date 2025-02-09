import cv2
import numpy as np
import random
import os
from utils.ediffiqa import eDifFIQA
from utils.utils import load_database, resize_with_padding


def get_database(database_path, models_path, quality_threshold = 0.3):
    SFace = cv2.FaceRecognizerSF.create(
        f"{models_path}/face_recognition_sface_2021dec.onnx", config=""
    )

    YuNet = cv2.FaceDetectorYN.create(
        f"{models_path}/face_detection_yunet_2023mar.onnx", "", (640, 640), 0.7
    )

    model_quality = eDifFIQA(
        modelPath=f"{models_path}/ediffiqa_tiny_jun2024.onnx",
    )

    database = load_database(database_path)
    names = []
    features = []
    for person_name, images in database.items():
        for image in images:

            image, _, _ = resize_with_padding(image, (640, 640))
            _, faces = YuNet.detect(image)
            if faces is not None:
                for face in faces:
                    aligned_face = SFace.alignCrop(image, face)

                    quality = model_quality.infer(aligned_face)
                    if quality < quality_threshold:
                        print(
                            f"Quality threshold not met, {person_name}, {quality}"
                        )
                        continue
                    feature = SFace.feature(aligned_face)
                    names.append(person_name)
                    features.append(feature[0])

    features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]
    features = np.array(features)
    return features, names

def register_faces(database_path, models_path):
    SFace = cv2.FaceRecognizerSF.create(
        f"{models_path}/face_recognition_sface_2021dec.onnx", config=""
    )

    YuNet = cv2.FaceDetectorYN.create(
        f"{models_path}/face_detection_yunet_2023mar.onnx", "", (640, 640), 0.7
    )

    model_quality = eDifFIQA(
        modelPath=f"{models_path}/ediffiqa_tiny_jun2024.onnx",
    )
    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            break

        output = cv2.resize(frame, (640, 640))
        faces, results = YuNet.detect(output)

        if results is not None:
            for det in results:
                aligned_image = SFace.alignCrop(output, results)
                quality = model_quality.infer(aligned_image)

                bbox = det[0:4].astype(np.int32)
                cv2.rectangle(
                    output,
                    (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    output,
                    f"{quality:.3f}",
                    (bbox[0], bbox[1]),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0, 0, 255),
                )

        cv2.imshow("frame", output)


        # Check for 'q' key press to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.destroyWindow("frame")
            name = input("Enter name: ")
            os.makedirs(f"{database_path}/{name}", exist_ok=True)
            cv2.imwrite(f"{database_path}/{name}/{random.randint(0, 10000)}.jpg", frame)

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_faces("../database", "../models")
