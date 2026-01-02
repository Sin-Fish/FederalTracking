# 使用官方Python基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制requirements.txt先进行依赖安装（利用Docker缓存层）
COPY requirements.txt /app/requirements.txt

# 配置pip使用国内镜像源并安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ --timeout 600 --retries 5

# 复制服务器端的所有代码文件到容器中
COPY . /app

# 暴露服务器端口
EXPOSE 8000

# 启动服务器
CMD ["python", "server.py", "--host", "0.0.0.0"]