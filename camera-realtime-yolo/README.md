# 摄像头实时 YOLO 检测

这是从原 `yolo-person-web` 项目中独立提取出来的摄像头实时检测小项目，只保留：

- 浏览器摄像头采集
- 实时人员/车辆检测切换
- YOLO 检测接口
- 浏览器端检测框绘制
- 本地录像与下载

## 目录结构

```text
camera-realtime-yolo/
  app.py                 Flask 入口，默认端口 5001
  config.py              YOLO 模型和实时检测参数
  yolo_detector.py       YOLO 懒加载与图片帧检测
  requirements.txt       依赖
  start.bat              Windows 双击启动
  templates/index.html   页面
  static/app.css         样式
  static/app.js          摄像头、检测、绘框、录像逻辑
```

## 安装依赖

```powershell
cd camera-realtime-yolo
python -m pip install -r requirements.txt
```

## 运行

```powershell
python app.py
```

启动后访问：

```text
http://127.0.0.1:5001
```

浏览器摄像头权限要求 `localhost`、`127.0.0.1` 或 HTTPS 环境。

## 模型配置

默认使用上级目录的模型：

```text
../models/yolov8s.pt
```

可以用环境变量替换：

```powershell
$env:YOLO_MODEL_PATH="C:\path\to\your-model.pt"
python app.py
```

常用参数：

```powershell
$env:CAMERA_CONF="0.22"
$env:CAMERA_IOU="0.50"
$env:CAMERA_IMGSZ="960"
$env:CAMERA_MIN_AREA="35"
$env:YOLO_DEVICE="auto"
```

如果安装了 CUDA 版 PyTorch，`YOLO_DEVICE=auto` 会优先使用 GPU；也可以手动设置 `YOLO_DEVICE=0` 或 `YOLO_DEVICE=cpu`。
