import cv2 as cv
import numpy as np
import os
import time

from orbbec_capture import OrbbecCamera


WINDOW_NAME = "Distance"
thres = [700, 900]


SAVE_PATH = "/home/jetson/ws/Facial-authenticity-judgment/code"


def set_thre1(thre):
    thres[0] = thre


def set_thre2(thre):
    thres[1] = thre


def should_exit():
    key = cv.waitKey(1) & 0xFF
    return key in (27, ord("q"))


def main():
    
    os.makedirs(SAVE_PATH, exist_ok=True)

    cv.namedWindow(WINDOW_NAME)
    cv.createTrackbar("dist1", WINDOW_NAME, thres[0], 3000, set_thre1)
    cv.createTrackbar("dist2", WINDOW_NAME, thres[1], 3000, set_thre2)

    try:
        camera = OrbbecCamera()
    except Exception as exc:
        print(f"Fail to open Orbbec camera: {exc}")
        return

    missed_frames = 0
    frame_idx = 0
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

            color_depth_map = cv.normalize(
                depth_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1
            )
            color_depth_map = cv.applyColorMap(color_depth_map, cv.COLORMAP_JET)
            cv.imshow("Depth", color_depth_map)

            segment_image, body_contour = segment_body(depth_map, thres)
            cv.imshow("Segmentation", segment_image)

            if body_contour is not None:
                cv.drawContours(bgr_image, body_contour, -1, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow("Body Contour", bgr_image)

            key = cv.waitKey(1) & 0xFF
            if key == ord('s'):
                timestamp = int(time.time())
                depth_file = os.path.join(SAVE_PATH, f"depth_{timestamp}.jpg")
                segment_file = os.path.join(SAVE_PATH, f"segment_{timestamp}.jpg")
                contour_file = os.path.join(SAVE_PATH, f"contour_{timestamp}.jpg")

                cv.imwrite(depth_file, color_depth_map)
                cv.imwrite(segment_file, segment_image)
                cv.imwrite(contour_file, bgr_image)

                print(f"Saved images:\n  {depth_file}\n  {segment_file}\n  {contour_file}")

            elif key in (27, ord('q')):
                break

            frame_idx += 1

    finally:
        camera.release()
        cv.destroyAllWindows()


def segment_body(depth_image, threshold):
    lo, hi = sorted(threshold)
    output = cv.inRange(depth_image, lo, hi)
    output = cv.dilate(output, None)
    output = cv.dilate(output, None)

    contours, _ = cv.findContours(output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return output, None

    body_contour = max(contours, key=cv.contourArea)
    output = np.zeros(depth_image.shape, dtype=np.uint8)
    cv.fillPoly(output, (body_contour,), 255)
    return output, (body_contour,)


if __name__ == "__main__":
    main()
