FROM python:3.9-slim

# 设置环境变量以禁用tokenizers并行处理警告
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建下载模型的脚本
RUN echo "import os" > download_model.py && \
    echo "# 设置环境变量以禁用tokenizers并行处理警告" >> download_model.py && \
    echo "os.environ.setdefault(\"TOKENIZERS_PARALLELISM\", \"false\")" >> download_model.py && \
    echo "" >> download_model.py && \
    echo "from sentence_transformers import SentenceTransformer" >> download_model.py && \
    echo "" >> download_model.py && \
    echo "def download_model():" >> download_model.py && \
    echo "    model_name = \"all-MiniLM-L6-v2\"" >> download_model.py && \
    echo "    model_cache_dir = \"./models\"" >> download_model.py && \
    echo "" >> download_model.py && \
    echo "    print(\"[CLIENT] 开始下载模型 {}...\".format(model_name))" >> download_model.py && \
    echo "" >> download_model.py && \
    echo "    try:" >> download_model.py && \
    echo "        os.makedirs(model_cache_dir, exist_ok=True)" >> download_model.py && \
    echo "" >> download_model.py && \
    echo "        model = SentenceTransformer(model_name)" >> download_model.py && \
    echo "" >> download_model.py && \
    echo "        model_path = os.path.join(model_cache_dir, model_name)" >> download_model.py && \
    echo "        model.save(model_path)" >> download_model.py && \
    echo "" >> download_model.py && \
    echo "        print(\"[CLIENT] 模型已成功下载并保存至 {}\".format(model_path))" >> download_model.py && \
    echo "        return True" >> download_model.py && \
    echo "    except Exception as e:" >> download_model.py && \
    echo "        print(\"[CLIENT] 下载模型失败: {}\".format(e))" >> download_model.py && \
    echo "        return False" >> download_model.py && \
    echo "" >> download_model.py && \
    echo "if __name__ == \"__main__\":" >> download_model.py && \
    echo "    download_model()" >> download_model.py

# 预先下载模型
RUN python download_model.py

# 复制客户端代码
COPY . .

# 暴露端口（虽然客户端不直接监听端口，但保留用于可能的调试）
EXPOSE 8001

# 启动客户端的命令
CMD ["python", "client.py"]