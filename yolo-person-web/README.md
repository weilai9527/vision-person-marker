# YOLO 人员检测网页

这是一个本地 Flask 小网站，支持图片和视频里的人员检测。图片上传后会使用本地 YOLO 模型框选行人；视频上传后会逐帧处理，生成带检测框的视频结果。

## 主要目录

```text
yolo-person-web/
  app.py                 Flask 入口
  config.py              路径和模型参数配置
  models.py              数据库模型
  services/              检测、绘图、视频处理等服务模块
  templates/             页面模板
  static/uploads/        图片上传目录
  static/results/        图片检测结果
  static/videos/         视频上传目录
  static/video_results/  视频检测结果
  instance/              数据库、日志和本地配置
```

## 安装依赖

```powershell
cd yolo-person-web
python -m pip install -r requirements.txt
```

## 运行

```powershell
cd yolo-person-web
python app.py
```

启动后访问：

```text
http://127.0.0.1:5000
```

也可以在项目根目录双击 `start.bat`。

## 常用配置

- `YOLO_MODEL_PATH`：YOLO 模型路径，默认使用上级 `models/yolov8s.pt`
- `YOLO_CONF`：检测置信度阈值，默认 `0.25`
- `YOLO_IOU`：NMS IOU 阈值，默认 `0.50`
- `YOLO_IMGSZ`：推理尺寸，默认 `1280`
- `YOLO_DEVICE`：推理设备，默认 `auto`。安装 CUDA 版 PyTorch 时会自动使用第 1 张 GPU；也可以手动设为 `0`、`1` 或 `cpu`
- `VIDEO_MAX_SIZE`：视频最大上传大小，默认 `500` MB

## GPU 加速

项目的 YOLO 推理支持 GPU。当前环境如果是 CPU 版 PyTorch，程序会自动退回 CPU。要启用 GPU，需要安装和显卡驱动匹配的 CUDA 版 PyTorch，然后保持 `YOLO_DEVICE=auto` 或手动设置为 `YOLO_DEVICE=0`。

```powershell
python -m pip install --upgrade --force-reinstall -r requirements-gpu.txt
```

视频读取、绘制检测框和写出 mp4 仍由 OpenCV 处理，通常还是 CPU；速度提升主要来自 YOLO 推理阶段。

可清理目录：`static/uploads/`、`static/results/`、`static/videos/`、`static/video_results/`、`instance/logs/`。
