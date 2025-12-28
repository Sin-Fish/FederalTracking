import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import threading
import os
import pickle
import zipfile

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin

from data_processor import DataProcessor
from models import SimpleLSTM


class CommunicationModule:
    """通信模块，处理与服务器的通信"""
    
    def __init__(self, server_url: str, client_id: str, data_processor: DataProcessor):
        self.server_url = server_url.rstrip("/")
        self.client_id = client_id
        self.data_processor = data_processor
        
        # 联邦学习状态
        self.global_prototypes: Optional[np.ndarray] = None
        self.prototype_labels: List[str] = []
        self.prototype_mapping: Dict[int, List[int]] = {}  # 原型->数据索引映射
        
        # 通信状态
        self.is_registered = False
        self.last_heartbeat = time.time()
        self.request_timeout = 30  # 增加请求超时时间
        
        # 嵌入模型
        self.embedding_model = None
        self.embedding_dim = 384
        
        # 本地模型缓存路径
        self.model_cache_path = "./models/embedding_model_cache"
        self.model_zip_path = "./models/embedding_model.zip"
    
    def check_local_model(self, server_model_hash: str) -> bool:
        """检查本地是否已有指定哈希的模型"""
        hash_file_path = f"{self.model_cache_path}_hash.txt"
        if not os.path.exists(hash_file_path):
            return False
            
        try:
            with open(hash_file_path, 'r') as f:
                cached_hash = f.read().strip()
                return cached_hash == server_model_hash
        except Exception:
            return False
    
    def download_model_from_server(self, server_model_hash: str) -> bool:
        """从服务器下载模型"""
        print(f"[CLIENT] 从服务器下载模型，哈希: {server_model_hash[:16]}...")
        
        try:
            response = requests.get(
                f"{self.server_url}/api/model/download",
                timeout=self.request_timeout
            )
            
            if response.status_code != 200:
                print(f"[CLIENT] 下载模型失败: HTTP {response.status_code}")
                return False
            
            # 获取服务器返回的模型哈希
            response_hash = response.headers.get('X-Model-Hash', '')
            if response_hash != server_model_hash:
                print(f"[CLIENT] 模型哈希验证失败: 期望 {server_model_hash[:16]}, 实际 {response_hash[:16]}")
                return False
            
            # 保存下载的模型压缩包
            os.makedirs(os.path.dirname(self.model_zip_path), exist_ok=True)
            with open(self.model_zip_path, 'wb') as f:
                f.write(response.content)
            
            # 解压模型
            with zipfile.ZipFile(self.model_zip_path, 'r') as zip_ref:
                zip_ref.extractall("./models/")
            
            # 保存哈希值用于后续验证
            hash_file_path = f"{self.model_cache_path}_hash.txt"
            with open(hash_file_path, 'w') as f:
                f.write(server_model_hash)
            
            print(f"[CLIENT] 模型下载并解压成功")
            return True
            
        except Exception as e:
            print(f"[CLIENT] 下载模型时发生错误: {e}")
            return False
    
    def load_embedding_model(self):
        """加载嵌入模型，优先从服务器获取或使用本地缓存"""
        print("[CLIENT] 准备加载嵌入模型...")
        
        # 向服务器请求模型信息
        try:
            response = requests.get(
                f"{self.server_url}/api/system/model_info",
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                model_info = response.json()
                model_hash = model_info.get('hash', '')
                
                # 检查本地是否有匹配的模型
                if self.check_local_model(model_hash):
                    print("[CLIENT] 本地已存在匹配的模型，准备加载...")
                    # 加载本地缓存模型（这里只是模拟，实际需要根据具体模型类型实现）
                    print("[CLIENT] 本地模型验证通过")
                    # 对于SentenceTransformer模型，我们仍需要加载它
                    from sentence_transformers import SentenceTransformer
                    self.embedding_model = SentenceTransformer('./models/all-MiniLM-L6-v2')
                    print("[CLIENT] 嵌入模型加载成功")
                else:
                    print(f"[CLIENT] 本地无匹配模型，从服务器下载 (Hash: {model_hash[:16]}...)")
                    
                    # 从服务器下载模型
                    if self.download_model_from_server(model_hash):
                        # 下载成功后，加载模型
                        from sentence_transformers import SentenceTransformer
                        self.embedding_model = SentenceTransformer('./models/all-MiniLM-L6-v2')
                        print("[CLIENT] 从服务器下载的模型加载成功")
                    else:
                        print("[CLIENT] 从服务器下载模型失败")
                        # 尝试使用随机向量作为备选方案
                        print("[CLIENT] 使用随机向量作为嵌入（备用方案）")
            else:
                print(f"[CLIENT] 获取模型信息失败: {response.status_code}")
                print("[CLIENT] 使用随机向量作为嵌入（备用方案）")
        except Exception as e:
            print(f"[CLIENT] 加载嵌入模型时发生错误: {e}")
            print("[CLIENT] 使用随机向量作为嵌入（备用方案）")
    
    def embed_behavior_strings(self, strings: List[str]) -> np.ndarray:
        """将行为字符串转换为嵌入向量"""
        self.load_embedding_model()
        
        if self.embedding_model is not None:
            print(f"[CLIENT] 嵌入 {len(strings)} 个行为字符串...")
            embeddings = self.embedding_model.encode(strings, show_progress_bar=False)
            return embeddings
        else:
            # 如果无法加载模型，则使用随机向量作为占位符
            print(f"[CLIENT] 使用随机向量嵌入 {len(strings)} 个行为字符串...")
            embeddings = np.random.rand(len(strings), self.embedding_dim).astype(np.float32)
            return embeddings
    
    def register_to_server(self) -> bool:
        """向服务器报到"""
        print("[CLIENT] 向服务器报到...")
        
        # 确保数据已处理
        if not self.data_processor.behavior_strings:
            self.data_processor.process_raw_data()
        
        fingerprint = self.data_processor.get_data_fingerprint()
        
        payload = {
            "client_id": self.client_id,
            "data_fingerprint": fingerprint,
            "embedding_version": "all-MiniLM-L6-v2",
            "sample_count": len(self.data_processor.behavior_strings)
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/api/client/register",
                json=payload,
                timeout=self.request_timeout  # 使用更长的超时时间
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"[CLIENT] 报到成功: {data}")
                self.is_registered = True
                return True
            else:
                print(f"[CLIENT] 报到失败: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"[CLIENT] 报到超时 (超过{self.request_timeout}秒)")
            return False
        except Exception as e:
            print(f"[CLIENT] 报到时发生错误: {e}")
            return False
    
    def submit_high_freq_data(self, top_k: int = 50) -> bool:
        """提交高频数据用于联邦聚类"""
        print("[CLIENT] 提交高频数据...")
        
        high_freq = self.data_processor.get_high_frequency_actions(top_k)
        
        payload = {
            "client_id": self.client_id,
            "high_freq_actions": high_freq,
            "sample_count": len(self.data_processor.behavior_strings)
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/api/data/collect",
                json=payload,
                timeout=self.request_timeout  # 使用更长的超时时间
            )
            
            if response.status_code == 200:
                print(f"[CLIENT] 高频数据提交成功: {response.json()}")
                return True
            else:
                print(f"[CLIENT] 高频数据提交失败: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"[CLIENT] 提交高频数据超时 (超过{self.request_timeout}秒)")
            return False
        except Exception as e:
            print(f"[CLIENT] 提交高频数据时发生错误: {e}")
            return False
    
    def fetch_prototypes(self) -> bool:
        """从服务器获取全局原型"""
        print("[CLIENT] 获取全局原型...")
        
        try:
            response = requests.get(
                f"{self.server_url}/api/system/prototypes",
                timeout=self.request_timeout  # 使用更长的超时时间
            )
            
            if response.status_code == 200:
                data = response.json()
                self.global_prototypes = np.array(data["prototypes"])
                self.prototype_labels = data["labels"]
                
                print(f"[CLIENT] 成功获取 {len(self.prototype_labels)} 个全局原型")
                
                # 立即进行数据映射
                self.map_to_prototypes(self.global_prototypes)
                return True
            else:
                print(f"[CLIENT] 获取原型失败: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"[CLIENT] 获取原型超时 (超过{self.request_timeout}秒)")
            return False
        except Exception as e:
            print(f"[CLIENT] 获取原型时发生错误: {e}")
            return False
    
    def map_to_prototypes(self, prototypes: np.ndarray):
        """将本地数据映射到全局原型"""
        print("[CLIENT] 将本地数据映射到全局原型...")
        
        # 计算本地数据的嵌入
        if self.data_processor.behavior_embeddings is None:
            self.data_processor.behavior_embeddings = self.embed_behavior_strings(self.data_processor.behavior_strings)
        
        # 为每个本地向量找到最近的原型
        prototype_indices = pairwise_distances_argmin(self.data_processor.behavior_embeddings, prototypes)
        
        # 构建映射
        self.prototype_mapping.clear()
        for idx, proto_idx in enumerate(prototype_indices):
            if proto_idx not in self.prototype_mapping:
                self.prototype_mapping[proto_idx] = []
            self.prototype_mapping[proto_idx].append(idx)
        
        # 打印统计信息
        print("[CLIENT] 数据分布统计:")
        for proto_idx in sorted(self.prototype_mapping.keys()):
            count = len(self.prototype_mapping[proto_idx])
            percentage = count / len(self.data_processor.behavior_strings) * 100
            label = self.prototype_labels[proto_idx] if proto_idx < len(self.prototype_labels) else f"原型{proto_idx}"
            print(f"  {label}: {count} 条数据 ({percentage:.1f}%)")
    
    def send_status_update(self, status: str, progress: Optional[float] = None) -> bool:
        """向服务器发送状态更新"""
        payload = {
            "client_id": self.client_id,
            "status": status,
            "progress": progress
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/api/client/status",
                json=payload,
                timeout=self.request_timeout  # 使用更长的超时时间
            )
            
            if response.status_code == 200:
                self.last_heartbeat = time.time()
                return True
            return False
            
        except requests.exceptions.Timeout:
            print(f"[CLIENT] 状态更新超时 (超过{self.request_timeout}秒)")
            return False
        except Exception as e:
            print(f"[CLIENT] 状态更新失败: {e}")
            return False
    
    def submit_model_updates(self, local_models: Dict[int, SimpleLSTM], current_round: int, epochs_per_round: int):
        """提交模型更新到服务器"""
        print("[CLIENT] 提交模型更新...")
        
        for proto_id, model in local_models.items():
            data_size = len(self.prototype_mapping[proto_id])
            
            # 获取模型状态
            state_dict = model.get_state_dict()
            
            payload = {
                "client_id": self.client_id,
                "prototype_id": proto_id,
                "model_state_dict": state_dict,
                "data_size": data_size,
                "metadata": {
                    "round": current_round,
                    "epochs": epochs_per_round,
                    "data_points": data_size,
                    "submitted_at": datetime.now().isoformat()
                }
            }
            
            try:
                response = requests.post(
                    f"{self.server_url}/api/model/update",
                    json=payload,
                    timeout=self.request_timeout  # 使用更长的超时时间
                )
                
                if response.status_code == 200:
                    print(f"[CLIENT] 原型{proto_id}的模型更新提交成功")
                else:
                    print(f"[CLIENT] 原型{proto_id}的模型更新提交失败: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"[CLIENT] 原型{proto_id}的模型更新提交超时 (超过{self.request_timeout}秒)")
            except Exception as e:
                print(f"[CLIENT] 提交模型更新时发生错误: {e}")
    
    def start_heartbeat(self, interval: int = 30):
        """启动心跳线程"""
        def heartbeat_loop():
            while True:
                time.sleep(interval)
                # 状态应该是由主程序控制的，这里只是发送上次已知状态
                # 在实际应用中，这里应该获取当前状态
                self.send_status_update("idle")  # 临时使用idle状态
        
        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        print(f"[CLIENT] 心跳线程已启动，间隔: {interval}秒")