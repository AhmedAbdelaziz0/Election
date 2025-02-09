import sys

import cv2

from utils.register import register_faces
from utils.utils import annotate
from FaceRecognizer import FaceRecognizer


print("Please specify either 'register' or 'recognize'")
if len(sys.argv) > 1 and sys.argv[1] == "register":
    register_faces("./database", "./models")
    exit(0)
else:
    face_recognizer = FaceRecognizer(
        quality_threshold=0.3,
        detection_threshold=0.7,
        distance_threshold=0.3,
    )

cap = cv2.VideoCapture(0)
scale = 2
skip_frames = 2
frame_count = -1
names = []
locations = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        break

    frame_count += 1
    if frame_count % skip_frames == 0:
        names, locations = face_recognizer.extract_faces(frame)

    annotate(frame, names, locations)

    frame = cv2.resize(
        frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    )
    cv2.imshow("Face Recognition", frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
