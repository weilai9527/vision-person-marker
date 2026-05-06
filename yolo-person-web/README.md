# 大模型人数检测网页

这是一个本地 Flask 小网站：上传一张图片后，程序会调用兼容 OpenAI Chat Completions 格式的视觉大模型 API，判断图片中的人数，并显示人数、模型说明和原图。

## 目录说明

```text
D:\yolodemo\yolo-person-web\
   app.py              Flask 后端，负责上传、调用大模型 API 和保存结果
   start.bat           Windows 启动脚本
   requirements.txt    Python 依赖
   templates\
      index.html       网页界面
   static\
      uploads\         上传的原图
      results\         用于页面展示的图片副本
```

## 安装依赖

```powershell
cd D:\yolodemo\yolo-person-web
D:\yolo\yolo_env\Scripts\python.exe -m pip install -r requirements.txt
```

## 配置 API

至少需要设置 `LLM_API_KEY`。如果你的服务商不是 OpenAI 官方接口，也需要设置 `LLM_API_URL` 和 `LLM_MODEL`。

```powershell
$env:LLM_API_KEY="你的 API Key"
$env:LLM_API_URL="https://api.openai.com/v1/chat/completions"
$env:LLM_MODEL="gpt-4o-mini"
```

可选配置：

- `LLM_API_URL`：大模型接口地址，默认 `https://api.openai.com/v1/chat/completions`
- `LLM_MODEL`：视觉模型名称，默认 `gpt-4o-mini`
- `LLM_TIMEOUT`：接口超时时间，默认 `60` 秒

## 运行

```powershell
cd D:\yolodemo\yolo-person-web
D:\yolo\yolo_env\Scripts\python.exe app.py
```

启动后打开浏览器访问：

```text
http://127.0.0.1:5000
```

## 可以清理的文件

- `static\uploads\`：历史上传图片，不影响程序代码。
- `static\results\`：历史展示图片，不影响程序代码。
- `__pycache__\`：Python 自动生成的缓存。
- `server.out.log` / `server.err.log`：启动日志。
