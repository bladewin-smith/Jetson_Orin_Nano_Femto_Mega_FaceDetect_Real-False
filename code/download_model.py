from pathlib import Path
from urllib.request import urlretrieve


MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    f"face_detection_yunet/{MODEL_NAME}"
)


def main():
    target = Path(__file__).resolve().parent / MODEL_NAME
    if target.exists() and target.stat().st_size > 1024:
        print(f"{MODEL_NAME} already exists: {target}")
        return

    print(f"Downloading {MODEL_NAME}...")
    urlretrieve(MODEL_URL, target)
    print(f"Saved to {target}")


if __name__ == "__main__":
    main()
