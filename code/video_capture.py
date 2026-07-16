import cv2 as cv
import numpy as np

from orbbec_capture import OrbbecCamera


def should_exit():
    key = cv.waitKey(1) & 0xFF
    return key in (27, ord("q"))


def main():
    try:
        camera = OrbbecCamera()
    except Exception as exc:
        print(f"Fail to open Orbbec camera: {exc}")
        return

    missed_frames = 0
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

            depth_map_8u = depth_map * 255.0 / 5000.0
            depth_map_8u = np.clip(depth_map_8u, 0, 255).astype(np.uint8)
            color_depth_map = cv.applyColorMap(depth_map_8u, cv.COLORMAP_JET)

            cv.imshow("BGR", bgr_image)
            cv.imshow("Depth: Gray", depth_map_8u)
            cv.imshow("Depth: ColorMap", color_depth_map)

            if should_exit():
                break
    finally:
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
