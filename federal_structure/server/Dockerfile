# 使用官方Python运行时作为基础镜像
FROM python:3.9-slim

# 设置环境变量以禁用tokenizers并行处理警告
ENV TOKENIZERS_PARALLELISM=false

# 设置工作目录
WORKDIR /app

# 复制requirements.txt先进行依赖安装（利用Docker缓存层）
COPY requirements.txt /app/requirements.txt

# 配置pip使用国内镜像源并安装依赖，跳过哈希验证
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY . /app

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]