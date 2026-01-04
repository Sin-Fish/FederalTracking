import os
# 设置环境变量以禁用tokenizers并行处理警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer

def download_model():
    """下载嵌入模型到本地"""
    model_name = "all-MiniLM-L6-v2"
    model_cache_dir = "./models"
    
    print(f"[SERVER] 开始下载模型 {model_name}...")
    
    try:
        # 确保模型目录存在
        os.makedirs(model_cache_dir, exist_ok=True)
        
        # 下载模型
        model = SentenceTransformer(model_name)
        
        # 保存到本地目录
        model_path = os.path.join(model_cache_dir, model_name)
        model.save(model_path)
        
        print(f"[SERVER] 模型已成功下载并保存至 {model_path}")
        return True
    except Exception as e:
        print(f"[SERVER] 下载模型失败: {e}")
        return False

if __name__ == "__main__":
    download_model()