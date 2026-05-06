# yolodemo 项目说明

这个目录现在分成两部分：

```text
D:\yolodemo
├─ models\               模型权重文件
│  ├─ yolov8s.pt          默认模型，检测精度更高但稍慢
│  └─ yolo11n.pt          备用模型，体积更小、速度更快
└─ yolo-person-web\       人数检测网页项目
   ├─ app.py              后端入口
   ├─ start.bat           双击启动
   ├─ README.md           使用说明
   ├─ requirements.txt    依赖列表
   ├─ templates\          HTML 页面
   └─ static\             上传图片和检测结果
```

## 平时主要看哪里

1. 想运行项目：看 `yolo-person-web\README.md`。
2. 想改检测逻辑：改 `yolo-person-web\app.py`。
3. 想改网页样式和文字：改 `yolo-person-web\templates\index.html`。
4. 想换模型：把模型放到 `models\`，再设置 `YOLO_MODEL_PATH`。

## 可以不用管的文件

- `static\uploads\`：运行时上传的图片。
- `static\results\`：运行时生成的检测结果图。
- `__pycache__\`：Python 缓存。
- `server.out.log` / `server.err.log`：服务日志。
