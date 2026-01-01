# 使用官方Python基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制服务器端的所有代码文件到容器中
COPY . /app

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露服务器端口
EXPOSE 8000

# 启动服务器
CMD ["python", "server.py", "--host", "0.0.0.0"]