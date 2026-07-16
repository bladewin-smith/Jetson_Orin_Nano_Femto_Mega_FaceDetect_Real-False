from pathlib import Path

import cv2 as cv
import numpy as np

from orbbec_capture import OrbbecCamera


MODEL_CANDIDATES = (
    "face_detection_yunet_2023mar.onnx",
    "face_detection_yunet_2022mar.onnx",
)
FACE_SCORE_THRESHOLD = 0.75
SPOOF_STD_THRESHOLD_MM = 5.0


def should_exit():
    key = cv.waitKey(1) & 0xFF
    return key in (27, ord("q"))


def main():
    try:
        camera = OrbbecCamera()
    except Exception as exc:
        print(f"Fail to open Orbbec camera: {exc}")
        return

    try:
        detector, model_path = create_face_detector()
    except FileNotFoundError as exc:
        print(exc)
        camera.release()
        return

    missed_frames = 0
    warned_detector_error = False
    try:
        while True:
            ret, bgr_image, depth_map = camera.read()
            if not ret:
                missed_frames += 1
                if missed_frames % 30 == 0:
                    print(f"Waiting for complete RGB-D frames... missed {missed_frames}")
                if should_exit():
                    break
                continue
            missed_frames = 0

            detector.setInputSize((bgr_image.shape[1], bgr_image.shape[0]))
            try:
                faces = detector.detect(bgr_image)
            except cv.error as exc:
                if not warned_detector_error:
                    print(f"FaceDetectorYN failed with model '{model_path}': {exc}")
                    print(
                        "If you are using OpenCV 4.x, download the OpenCV Zoo "
                        "face_detection_yunet_2023mar.onnx model and put it in this folder."
                    )
                    warned_detector_error = True
                cv.imshow("Demo", bgr_image)
                if should_exit():
                    break
                continue
            flags = anti_spoofing(faces, depth_map)
            visualize(faces, flags, bgr_image)
            show_status(faces, bgr_image)
            cv.imshow("Depth", depth_to_colormap(depth_map))
            cv.imwrite("anti-spoofing.jpg", bgr_image)
            cv.imshow("Demo", bgr_image)
            if should_exit():
                break
    finally:
        camera.release()
        cv.destroyAllWindows()


def create_face_detector():
    for model_path in MODEL_CANDIDATES:
        path = Path(model_path)
        if not path.exists():
            continue

        if "2022mar" in model_path:
            print(
                "Warning: using old YuNet 2022 model. OpenCV 4.10 may fail at detect(). "
                "Prefer face_detection_yunet_2023mar.onnx."
            )

        detector = cv.FaceDetectorYN.create(
            str(path),
            "",
            (640, 480),
            score_threshold=FACE_SCORE_THRESHOLD,
        )
        print(f"Using face detector model: {model_path}")
        print(f"Face score threshold: {FACE_SCORE_THRESHOLD}")
        return detector, model_path

    raise FileNotFoundError(
        "No YuNet model found. Download face_detection_yunet_2023mar.onnx into "
        "the code directory before running anti-spoofing.py."
    )


def anti_spoofing(faces, depth_map):
    face_flags = []
    if faces[1] is None:
        return face_flags

    h, w = depth_map.shape
    for face in faces[1]:
        coords = face[:-1].astype(np.int32)
        dists = []
        for i in range(2, 7):
            x = coords[2 * i]
            y = coords[2 * i + 1]
            if 0 <= x < w and 0 <= y < h:
                dist = float(depth_map[y, x])
                if dist > 0:
                    dists.append(dist)

        std = np.std(dists) if len(dists) == 5 else -1.0
        print("std: {:.2f}".format(std))

        if std < 0:
            face_flags.append(0)
        elif std < SPOOF_STD_THRESHOLD_MM:
            face_flags.append(-1)
        else:
            face_flags.append(1)

    return face_flags


def visualize(faces, face_flags, bgr_image):
    color = {
        -1: (0, 0, 255),
        0: (255, 0, 0),
        1: (0, 255, 0),
    }
    label = {
        -1: "False",
        0: "Undetermined",
        1: "True",
    }

    if faces[1] is None:
        return

    thickness = 2
    for idx, face in enumerate(faces[1]):
        coords = face[:-1].astype(np.int32)
        flag = face_flags[idx]
        face_color = color[flag]

        cv.rectangle(
            bgr_image,
            (coords[0], coords[1]),
            (coords[0] + coords[2], coords[1] + coords[3]),
            face_color,
            thickness,
        )
        cv.putText(
            bgr_image,
            label[flag],
            (coords[0], coords[1] - 5),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            face_color,
            thickness,
        )

        cv.circle(bgr_image, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
        cv.circle(bgr_image, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
        cv.circle(bgr_image, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
        cv.circle(bgr_image, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
        cv.circle(bgr_image, (coords[12], coords[13]), 2, (0, 255, 255), thickness)


def show_status(faces, bgr_image):
    face_count = 0 if faces[1] is None else len(faces[1])
    text = f"Faces: {face_count}"
    if face_count == 0:
        text += "  Move closer / face camera / improve light"
    cv.putText(
        bgr_image,
        text,
        (10, 24),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv.LINE_AA,
    )


def depth_to_colormap(depth_map):
    depth_8u = depth_map * 255.0 / 5000.0
    depth_8u = np.clip(depth_8u, 0, 255).astype(np.uint8)
    return cv.applyColorMap(depth_8u, cv.COLORMAP_JET)


if __name__ == "__main__":
    main()
