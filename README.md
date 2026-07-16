# Facial Authenticity Judgment on Jetson Orin Nano

本仓库是对原 `Facial-authenticity-judgment` 项目的 Jetson 迁移版本，目标平台为：

- NVIDIA Jetson Orin Nano
- Ubuntu 22.04 / JetPack 6.x
- Orbbec Femto Mega RGB-D 相机
- Python + OpenCV + `pyorbbecsdk2`

项目功能包括：

- RGB-D 数据流预览
- 基于深度图的人体距离分割
- 基于 YuNet 人脸关键点 + 深度标准差的简单活体判断

## 为什么迁移

原项目使用：

```python
cv.VideoCapture(0, cv.CAP_OBSENSOR)
```

在 Jetson + Femto Mega 环境中，OpenCV 的 `CAP_OBSENSOR` 后端容易出现 `grab()` 持续失败、硬件 D2C 对齐不出帧等问题。因此本版本将相机采集层迁移为 Orbbec 官方 Python SDK：

```python
pyorbbecsdk2
```

新的公共采集模块为：

```text
code/orbbec_capture.py
```

三个业务脚本都通过它获取统一的 `bgr_image` 和 `depth_map`。

## 文件说明

```text
code/
├── orbbec_capture.py              # Orbbec RGB-D 采集封装
├── video_capture.py               # RGB + Depth 预览
├── body_segmentation.py           # 深度阈值人体分割
├── anti-spoofing.py               # 人脸活体判断
├── download_yunet_model.py        # 下载新版 YuNet 模型
├── requirements.txt               # Python 依赖
├── face_detection_yunet_2022mar.onnx
└── *.jpg                          # 示例输出图
```

## 环境安装

进入代码目录：

```bash
cd ~/ws/Facial-authenticity-judgment/code
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 OpenCV 4.x，建议下载新版 YuNet 模型：

```bash
python download_yunet_model.py
```

也可以手动下载：

```bash
wget -O face_detection_yunet_2023mar.onnx \
https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## 运行

同一时间只能运行一个相机脚本。先确认没有其它程序占用相机：

```bash
sudo fuser -v /dev/video*
```

RGB-D 预览：

```bash
python video_capture.py
```

人体分割：

```bash
python body_segmentation.py
```

活体判断：

```bash
python anti-spoofing.py
```

按 `q` 或 `Esc` 退出程序。

## 关键实现点

### 1. 统一使用 `pyorbbecsdk2`

相机采集由 `orbbec_capture.py` 封装，避免每个脚本单独初始化 Orbbec Pipeline。

### 2. 默认软件 depth-to-color 对齐

Femto Mega 在 Jetson 上使用硬件 D2C 对齐时可能成功启动但无法持续出帧，因此默认使用软件对齐：

```bash
ORBBEC_ALIGN_MODE=software python video_capture.py
```

可选模式：

```bash
ORBBEC_ALIGN_MODE=none python video_capture.py
ORBBEC_ALIGN_MODE=default python video_capture.py
ORBBEC_ALIGN_MODE=auto python video_capture.py
```

### 3. 输出尺寸统一为 640x480

采集层会将 RGB 和 Depth 统一调整为 `640x480`，降低 Jetson 上显示和处理压力。

可通过环境变量调整：

```bash
ORBBEC_OUTPUT_SIZE=320x240 python body_segmentation.py
```

### 4. 避免多进程抢占相机

如果看到：

```text
Device or resource busy
```

说明已有程序占用 `/dev/video*`。先关闭其它脚本或 Viewer：

```bash
pkill -f video_capture.py
pkill -f body_segmentation.py
pkill -f anti-spoofing.py
```

### 5. YuNet 模型版本

旧版 `face_detection_yunet_2022mar.onnx` 在 OpenCV 4.10 上可能报：

```text
Layer with requested id=-1 not found
```

因此活体脚本会优先使用：

```text
face_detection_yunet_2023mar.onnx
```

### 6. 人脸检测阈值

原项目阈值 `0.99` 对真实相机画面偏严格，本版本默认改为：

```python
FACE_SCORE_THRESHOLD = 0.6
```

如果画面显示 `Faces: 0`，可以继续降低到 `0.3` 试验。

## 当前限制

本项目的活体判断是一个教学级规则 Demo：取人脸 5 个关键点的深度值，计算标准差；如果深度变化过小，则倾向判断为平面攻击。

它可以用于理解 RGB-D 活体检测流程，但不建议直接作为生产级安全方案。

## 常见问题

### `Fail to grab data from camera`

偶发空帧属于 RGB-D 流同步中的常见现象。当前代码会自动重试，并只在连续多次拿不到完整帧时提示。

### `Device or resource busy`

相机被其它进程占用。使用：

```bash
sudo fuser -v /dev/video*
```

找出 PID 后结束对应进程。

### 只有 RGB 图，没有框

看左上角 `Faces: N`：

- `Faces: 0`：人脸检测没出来，调光照、距离或降低阈值
- `Faces: 1` 但没有稳定判断：查看终端里的 `std` 深度标准差

## 致谢

本项目基于原 `Facial-authenticity-judgment` 思路迁移，采集层改为 Orbbec 官方 Python SDK，并针对 Jetson + Femto Mega 的实际运行问题做了兼容处理。

