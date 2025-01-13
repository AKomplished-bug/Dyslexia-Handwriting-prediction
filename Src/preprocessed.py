import os
import cv2
import numpy as np

def preprocess_and_save(data_dir, output_file):
    images, labels = [], []
    label_mapping = {'Normal': 0, 'Reversal': 1, 'Corrected': 2}

    for label_name, label_value in label_mapping.items():
        class_dir = os.path.join(data_dir, label_name)
        print(f"Checking for folder: {class_dir}")
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist. Skipping.")
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            print(f"Processing image: {img_path}")

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to read image: {img_path}")
                continue

            image = cv2.resize(image, (224, 224)) / 255.0
            images.append(image)
            labels.append(label_value)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    print(f"Total images: {len(images)}, Total labels: {len(labels)}")

    np.savez_compressed(output_file, images=images, labels=labels)
    print(f"Preprocessed data saved to {output_file} (Total samples: {len(labels)})")


def organize_and_preprocess(train_dir, test_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing train data from: {train_dir}")
    preprocess_and_save(train_dir, os.path.join(output_dir, "train_data.npz"))

    print(f"Processing test data from: {test_dir}")
    preprocess_and_save(test_dir, os.path.join(output_dir, "test_data.npz"))

    print("All datasets processed successfully!")


if __name__ == "__main__":
    # Dynamically resolve project root and paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_dir = os.path.join(project_root, "Data/train")
    test_dir = os.path.join(project_root, "Data/test")
    output_dir = os.path.join(project_root, "Data/processed")

    print(f"Resolved Train directory: {train_dir}")
    print(f"Resolved Test directory: {test_dir}")

    if not os.path.exists(train_dir):
        print(f"Error: Train directory {train_dir} does not exist.")
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} does not exist.")

    organize_and_preprocess(train_dir, test_dir, output_dir)
