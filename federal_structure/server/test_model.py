import os
import json
import pickle
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 设置环境变量以禁用tokenizers并行处理警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

from ml_models import SimpleLSTM


class ModelTester:
    def __init__(self, test_data_path: str, model_path: str):
        self.test_data_path = test_data_path
        self.model_path = model_path
        self.embedding_model = None
        self.aggregated_models = None
        self.test_data = []
        
    def load_test_data(self):
        """加载测试数据"""
        print(f"[TEST] 从 {self.test_data_path} 加载测试数据...")
        
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        self.test_data.append(record)
                    except json.JSONDecodeError:
                        continue
        
        print(f"[TEST] 成功加载 {len(self.test_data)} 条测试记录")
        return self.test_data

    def load_aggregated_models(self):
        """加载聚合后的模型"""
        print(f"[TEST] 从 {self.model_path} 加载聚合模型...")
        
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.aggregated_models = data['models']
        self.prototype_labels = data['prototype_labels']
        self.prototype_centers = data['prototypes']  # 使用正确的键名
        self.timestamp = data['timestamp']
        
        print(f"[TEST] 成功加载 {len(self.aggregated_models)} 个聚合模型")
        print(f"[TEST] 模型标签: {self.prototype_labels}")
        
        return self.aggregated_models

    def initialize_embedding_model(self):
        """初始化嵌入模型"""
        print("[TEST] 初始化嵌入模型...")
        
        # 尝试从本地加载
        local_model_path = "./models/all-MiniLM-L6-v2"
        if os.path.exists(local_model_path):
            self.embedding_model = SentenceTransformer(local_model_path)
            print(f"[TEST] 从本地加载模型: {local_model_path}")
        else:
            # 从网络下载
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[TEST] 从网络下载模型: all-MiniLM-L6-v2")
        
        return self.embedding_model

    def process_raw_data(self):
        """处理原始数据，生成行为字符串"""
        print("[TEST] 处理原始数据，生成行为字符串...")
        
        behavior_strings = []
        for record in self.test_data:
            try:
                # 提取行为信息
                if "activity" in record:
                    activity = record["activity"]
                    behavior = f"{activity.get('window_title', '')} - {activity.get('process_name', '')}"
                elif "action" in record:
                    behavior = record["action"]
                else:
                    behavior = str(record)
                
                if behavior.strip():
                    behavior_strings.append(behavior.strip())
            except Exception as e:
                print(f"[TEST] 处理记录时出错: {e}")
                continue
        
        print(f"[TEST] 生成 {len(behavior_strings)} 个行为字符串")
        return behavior_strings

    def test_embeddings(self):
        """测试嵌入功能"""
        print("[TEST] 测试嵌入功能...")
        
        # 初始化模型
        self.initialize_embedding_model()
        
        # 处理数据
        behavior_strings = self.process_raw_data()
        
        if not behavior_strings:
            print("[TEST] 没有有效的行为字符串用于测试")
            return
        
        # 生成嵌入
        print(f"[TEST] 生成 {len(behavior_strings)} 个行为字符串的嵌入...")
        embeddings = self.embedding_model.encode(behavior_strings, show_progress_bar=True)
        
        print(f"[TEST] 嵌入形状: {embeddings.shape}")
        print(f"[TEST] 嵌入功能测试完成")
        
        return embeddings

    def embed_behavior_strings(self, behavior_strings: List[str]):
        """将行为字符串转换为嵌入向量"""
        print(f"[TEST] 为 {len(behavior_strings)} 个行为字符串生成嵌入向量...")
        print("[TEST] 注意：对于大型数据集，此过程可能需要一些时间...")
        
        if not self.embedding_model:
            self.initialize_embedding_model()
        
        # 为了提高效率，如果数据集太大，我们可以采样一部分进行测试
        sample_size = min(2000, len(behavior_strings))  # 限制最大样本数
        if len(behavior_strings) > sample_size:
            print(f"[TEST] 数据集过大，将采样 {sample_size} 条记录进行测试")
            import random
            sampled_behaviors = random.sample(behavior_strings, sample_size)
        else:
            sampled_behaviors = behavior_strings
        
        print(f"[TEST] 使用 {len(sampled_behaviors)} 条记录生成嵌入向量...")
        embeddings = self.embedding_model.encode(sampled_behaviors)
        print(f"[TEST] 嵌入向量形状: {embeddings.shape}")
        
        return embeddings

    def evaluate_model(self, model: SimpleLSTM, test_embeddings: np.ndarray, prototype_center: np.ndarray):
        """评估单个模型"""
        print(f"[TEST] 评估模型...")
        
        if prototype_center is None:
            print("[TEST] 原型中心为None，跳过此模型的评估")
            return {
                "mse": float('inf'),
                "mae": float('inf'),
                "cosine_similarity": float('nan')
            }
        
        # 准备测试数据
        # 使用原型中心作为目标值进行评估
        n_samples = min(len(test_embeddings), 1000)  # 限制测试样本数量以提高效率
        test_data = test_embeddings[:n_samples]
        
        # 转换为PyTorch张量
        test_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)  # 添加序列维度
        
        # 设置模型为评估模式
        model.eval()
        
        with torch.no_grad():
            predictions = model(test_tensor)
        
        # 将预测结果与原型中心比较
        target = np.tile(prototype_center, (n_samples, 1))
        predictions_np = predictions.numpy()
        
        # 计算评估指标
        mse = mean_squared_error(target, predictions_np)
        mae = mean_absolute_error(target, predictions_np)
        
        # 计算预测值与目标值之间的余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        cosine_sim = np.mean([
            cosine_similarity([target[i]], [predictions_np[i]])[0][0]
            for i in range(len(target))
        ])
        
        print(f"[TEST] 模型评估结果:")
        print(f"  - 均方误差 (MSE): {mse:.6f}")
        print(f"  - 平均绝对误差 (MAE): {mae:.6f}")
        print(f"  - 余弦相似度: {cosine_sim:.6f}")
        
        return {
            "mse": mse,
            "mae": mae,
            "cosine_similarity": cosine_sim
        }

    def test_all_models(self):
        """测试所有聚合模型"""
        print("="*60)
        print("开始测试聚合模型")
        print("="*60)
        
        # 加载数据和模型
        self.load_test_data()
        self.load_aggregated_models()
        behavior_strings = self.process_raw_data()
        test_embeddings = self.embed_behavior_strings(behavior_strings)
        
        # 测试每个模型
        results = {}
        
        for proto_id, model_state in self.aggregated_models.items():
            print(f"\n[TEST] 测试原型 {proto_id} 的模型...")
            
            # 创建模型实例
            # 从模型状态中推断参数
            fc_weight_shape = np.array(model_state['fc.weight']).shape
            output_size = fc_weight_shape[0]
            hidden_size = fc_weight_shape[1]
            
            lstm_ih_weight_shape = np.array(model_state['lstm.weight_ih_l0']).shape
            input_size = lstm_ih_weight_shape[1]
            
            model = SimpleLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
            
            # 加载模型权重
            model.load_state_dict(model_state)
            
            # 获取对应的原型中心
            prototype_center = None
            if self.prototype_centers is not None and proto_id < len(self.prototype_centers):
                prototype_center = self.prototype_centers[proto_id]
            
            # 评估模型
            eval_result = self.evaluate_model(model, test_embeddings, prototype_center)
            results[proto_id] = eval_result
        
        # 输出总体结果
        print("\n" + "="*60)
        print("模型测试总结")
        print("="*60)
        
        valid_results = {k: v for k, v in results.items() 
                        if v['mse'] != float('inf') and not np.isnan(v['cosine_similarity'])}
        
        if valid_results:
            total_mse = sum([r['mse'] for r in valid_results.values()]) / len(valid_results)
            total_mae = sum([r['mae'] for r in valid_results.values()]) / len(valid_results)
            total_cosine = sum([r['cosine_similarity'] for r in valid_results.values()]) / len(valid_results)
            
            print(f"总体平均指标 (基于 {len(valid_results)} 个有效模型):")
            print(f"  - 平均均方误差 (MSE): {total_mse:.6f}")
            print(f"  - 平均绝对误差 (MAE): {total_mae:.6f}")
            print(f"  - 平均余弦相似度: {total_cosine:.6f}")
        else:
            print("没有有效的模型评估结果")
        
        print(f"\n聚合模型时间戳: {self.timestamp}")
        print(f"测试数据来源: {self.test_data_path}")
        print(f"测试数据量: {len(self.test_data)} 条记录")
        
        return results


def test_aggregated_model(test_data_path: str, model_path: str):
    """测试聚合模型的主函数"""
    tester = ModelTester(test_data_path, model_path)
    return tester.test_all_models()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python test_model.py <test_data_path> <model_path>")
        print("示例: python test_model.py ./test_data/activity_log_20251228.jsonl ./aggregated_models/aggregated_models_20251229_011536.pkl")
        sys.exit(1)
    
    test_data_path = sys.argv[1]
    model_path = sys.argv[2]
    
    test_aggregated_model(test_data_path, model_path)