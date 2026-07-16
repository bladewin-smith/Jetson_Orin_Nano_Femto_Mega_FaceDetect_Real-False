import gc
import os
import time

import cv2 as cv
import numpy as np
from pyorbbecsdk import (
    AlignFilter,
    Config,
    FormatConvertFilter,
    OBAlignMode,
    OBConvertFormat,
    OBError,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBSensorType,
    OBStreamType,
    Pipeline,
)


DEFAULT_OUTPUT_SIZE = (640, 480)


class OrbbecCamera:
    """Small RGB-D wrapper around pyorbbecsdk2."""

    def __init__(self, timeout_ms=1000, align_mode=None, output_size=None):
        self.timeout_ms = timeout_ms
        self.align_mode = (align_mode or os.getenv("ORBBEC_ALIGN_MODE", "software")).lower()
        self.output_size = output_size or _parse_output_size(
            os.getenv("ORBBEC_OUTPUT_SIZE"), DEFAULT_OUTPUT_SIZE
        )
        self.pipeline = None
        self.config = None
        self.align_filter = None
        self.started = False

        self._start_with_fallbacks()
        print(f"Output RGB-D frame size: {self.output_size[0]}x{self.output_size[1]}")

    def _enable_frame_sync(self):
        if hasattr(self.pipeline, "enable_frame_sync"):
            self.pipeline.enable_frame_sync()

    def _start_with_fallbacks(self):
        modes = self._mode_order(self.align_mode)
        last_error = None
        for mode in modes:
            try:
                self._start_mode(mode)
                self.started = True
                return
            except Exception as exc:
                last_error = exc
                print(f"Start mode '{mode}' failed: {exc}")
                self.release()
                if _is_device_busy_error(exc):
                    raise RuntimeError(
                        "Orbbec camera device is busy. Close other camera programs "
                        "such as video_capture.py, body_segmentation.py, OrbbecViewer, "
                        "ROS camera nodes, or any process using /dev/video*."
                    ) from exc
                time.sleep(0.2)

        raise RuntimeError(f"Failed to start Orbbec camera: {last_error}")

    def _mode_order(self, mode):
        if mode == "auto":
            return ("software", "none", "default", "hardware")
        if mode in ("software", "none", "default", "hardware"):
            return (mode, "none", "default") if mode != "default" else ("default",)
        print(f"Unknown ORBBEC_ALIGN_MODE '{mode}', using software.")
        return ("software", "none", "default")

    def _start_mode(self, mode):
        self.pipeline = Pipeline()
        self.config = Config()
        self.align_filter = None

        if mode == "default":
            self.config = None
            self._enable_frame_sync()
            self.pipeline.start()
            print("Using Orbbec SDK default stream configuration.")
            return

        if mode == "hardware":
            if not self._try_configure_hardware_d2c():
                raise RuntimeError("hardware D2C profile is unavailable")
        elif mode == "software":
            self._configure_color_depth_streams()
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            print("Using software depth-to-color alignment.")
        elif mode == "none":
            self._configure_color_depth_streams()
            print("Using RGB-D streams without depth-to-color alignment.")
        else:
            raise ValueError(f"unsupported align mode: {mode}")

        self._enable_frame_sync()
        self.pipeline.start(self.config)

    def _configure_color_depth_streams(self):
        color_profile = self._get_color_profile()
        depth_profile = self._get_depth_profile()
        self.config.enable_stream(color_profile)
        self.config.enable_stream(depth_profile)
        self.config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
        )

    def _try_configure_hardware_d2c(self):
        try:
            color_profiles = self.pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
            for idx in range(len(color_profiles)):
                color_profile = color_profiles[idx]
                if color_profile.get_format() != OBFormat.RGB:
                    continue

                depth_profiles = self.pipeline.get_d2c_depth_profile_list(
                    color_profile, OBAlignMode.HW_MODE
                )
                if len(depth_profiles) == 0:
                    continue

                self.config.enable_stream(color_profile)
                self.config.enable_stream(depth_profiles[0])
                self.config.set_align_mode(OBAlignMode.HW_MODE)
                self.config.set_frame_aggregate_output_mode(
                    OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
                )
                print("Using hardware depth-to-color alignment.")
                return True
        except Exception as exc:
            print(f"Hardware alignment is unavailable: {exc}")

        return False

    def _get_color_profile(self):
        profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        for width, height, fps in ((640, 480, 30), (640, 480, 0), (0, 0, 0)):
            try:
                return profiles.get_video_stream_profile(
                    width, height, OBFormat.RGB, fps
                )
            except OBError:
                continue
        return profiles.get_default_video_stream_profile()

    def _get_depth_profile(self):
        profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        for width, height, fps in ((640, 480, 30), (640, 400, 30), (0, 0, 0)):
            try:
                return profiles.get_video_stream_profile(
                    width, height, OBFormat.Y16, fps
                )
            except OBError:
                continue
        return profiles.get_default_video_stream_profile()

    def read(self, attempts=5):
        for _ in range(max(1, attempts)):
            ret, bgr_image, depth_map = self._read_once()
            if ret:
                return ret, bgr_image, depth_map
        return False, None, None

    def _read_once(self):
        if self.pipeline is None:
            return False, None, None

        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return False, None, None

        if self.align_filter is not None:
            aligned = self.align_filter.process(frames)
            if aligned is None:
                return False, None, None
            frames = aligned.as_frame_set()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return False, None, None

        bgr_image = frame_to_bgr_image(color_frame)
        depth_map = depth_frame_to_mm(depth_frame)
        if bgr_image is None or depth_map is None:
            return False, None, None

        color_h, color_w = bgr_image.shape[:2]
        if depth_map.shape[:2] != (color_h, color_w):
            depth_map = cv.resize(
                depth_map, (color_w, color_h), interpolation=cv.INTER_NEAREST
            )

        bgr_image, depth_map = resize_rgbd(bgr_image, depth_map, self.output_size)
        return True, bgr_image, depth_map

    def release(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            finally:
                self.started = False
                self.pipeline = None
                self.config = None
                self.align_filter = None
                gc.collect()


def depth_frame_to_mm(depth_frame):
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    return data.reshape((height, width)).astype(np.float32) * scale


def resize_rgbd(bgr_image, depth_map, output_size):
    target_w, target_h = output_size
    if bgr_image.shape[:2] == (target_h, target_w):
        return bgr_image, depth_map

    bgr_image = cv.resize(bgr_image, (target_w, target_h), interpolation=cv.INTER_AREA)
    depth_map = cv.resize(depth_map, (target_w, target_h), interpolation=cv.INTER_NEAREST)
    return bgr_image, depth_map


def _parse_output_size(value, default_size):
    if not value:
        return default_size

    try:
        width_text, height_text = value.lower().replace("*", "x").split("x", 1)
        width = int(width_text.strip())
        height = int(height_text.strip())
        if width > 0 and height > 0:
            return width, height
    except ValueError:
        pass

    print(f"Invalid ORBBEC_OUTPUT_SIZE '{value}', using {default_size[0]}x{default_size[1]}.")
    return default_size


def _is_device_busy_error(exc):
    message = str(exc).lower()
    return "device or resource busy" in message or "resource busy" in message


def frame_to_bgr_image(frame):
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if color_format == OBFormat.RGB:
        image = data.reshape((height, width, 3))
        return cv.cvtColor(image, cv.COLOR_RGB2BGR)

    if color_format == OBFormat.BGR:
        return data.reshape((height, width, 3))

    if color_format == OBFormat.MJPG:
        return cv.imdecode(data, cv.IMREAD_COLOR)

    if color_format == OBFormat.YUYV:
        image = data.reshape((height, width, 2))
        return cv.cvtColor(image, cv.COLOR_YUV2BGR_YUY2)

    if color_format == OBFormat.UYVY:
        image = data.reshape((height, width, 2))
        return cv.cvtColor(image, cv.COLOR_YUV2BGR_UYVY)

    if color_format == OBFormat.I420:
        image = data.reshape((height * 3 // 2, width))
        return cv.cvtColor(image, cv.COLOR_YUV2BGR_I420)

    if color_format == OBFormat.NV12:
        image = data.reshape((height * 3 // 2, width))
        return cv.cvtColor(image, cv.COLOR_YUV2BGR_NV12)

    if color_format == OBFormat.NV21:
        image = data.reshape((height * 3 // 2, width))
        return cv.cvtColor(image, cv.COLOR_YUV2BGR_NV21)

    rgb_frame = _try_convert_to_rgb(frame)
    if rgb_frame is None:
        print(f"Unsupported color format: {color_format}")
        return None
    return frame_to_bgr_image(rgb_frame)


def _try_convert_to_rgb(frame):
    convert_formats = {
        OBFormat.I420: OBConvertFormat.I420_TO_RGB888,
        OBFormat.MJPG: OBConvertFormat.MJPG_TO_RGB888,
        OBFormat.YUYV: OBConvertFormat.YUYV_TO_RGB888,
        OBFormat.NV21: OBConvertFormat.NV21_TO_RGB888,
        OBFormat.NV12: OBConvertFormat.NV12_TO_RGB888,
        OBFormat.UYVY: OBConvertFormat.UYVY_TO_RGB888,
    }
    convert_format = convert_formats.get(frame.get_format())
    if convert_format is None:
        return None

    convert_filter = FormatConvertFilter()
    convert_filter.set_format_convert_format(convert_format)
    return convert_filter.process(frame)
