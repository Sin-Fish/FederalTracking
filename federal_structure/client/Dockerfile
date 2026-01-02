FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ --timeout 600 --retries 3 --trusted-host pypi.tuna.tsinghua.edu.cn

# 复制客户端代码
COPY . .

# 暴露端口（虽然客户端不直接监听端口，但保留用于可能的调试）
EXPOSE 8001

# 启动客户端的命令
CMD ["python", "client.py"]