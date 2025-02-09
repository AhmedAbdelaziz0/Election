import cv2
import os
import hashlib
import pickle


def resize_with_padding(image, target_size):
    height, width = image.shape[:2]
    width_ratio = target_size[0] / width
    height_ratio = target_size[1] / height

    scale_factor = min(width_ratio, height_ratio)

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    resized_image = cv2.resize(image, (new_width, new_height))

    pad_left = (target_size[0] - new_width) // 2
    pad_right = target_size[0] - new_width - pad_left
    pad_top = (target_size[1] - new_height) // 2
    pad_bottom = target_size[1] - new_height - pad_top

    padded_image = cv2.copyMakeBorder(
        resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT
    )

    return padded_image, scale_factor, (pad_left, pad_right, pad_top, pad_bottom)


def load_database(dataset):
    database = {}

    for person_name in os.listdir(dataset):
        person_path = os.path.join(dataset, person_name)

        if os.path.isdir(person_path):
            images = []
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                image = cv2.imread(image_path)
                image, _, _ = resize_with_padding(image, (640, 640))
                images.append(image)
            database[person_name] = images

    return database


def annotate(image, names, locations):
    for name, (x, y, w, h) in zip(names, locations):
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            image,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )


def compute_directory_hash(directory_path):
    sha256 = hashlib.sha256()
    for root, _, files in os.walk(directory_path):
        for file_name in sorted(files):  # Sort to ensure consistent order
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, "rb") as f:
                    while chunk := f.read(4096):  # Read in chunks
                        sha256.update(chunk)
            except Exception as e:
                print(f"Could not read file {file_path}: {e}")
    return sha256.hexdigest()


def check_for_changes(directory_path):
    previous_hash = None
    if os.path.exists(".database_hash.pkl"):
        with open(".database_hash.pkl", "rb") as f:
            previous_hash = pickle.load(f)

    current_hash = compute_directory_hash(directory_path)
    if current_hash != previous_hash:
        with open(".database_hash.pkl", "wb") as f:
            pickle.dump(current_hash, f)

        return True
    else:
        return False
