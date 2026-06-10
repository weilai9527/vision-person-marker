# 基础镜像：使用轻量级的 Python 3.10
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 环境变量：不生成 .pyc 文件，且让日志直接输出不缓冲
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装 OpenCV 所需的系统依赖库
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先拷贝 requirements 缓存依赖层
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 安装 Gunicorn 用于生产环境运行
RUN pip install --no-cache-dir gunicorn

# 拷贝项目全量代码
COPY . /app/

# 使用单进程多线程模式启动（配合我们的 FileLock 和 ThreadPoolExecutor）
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]