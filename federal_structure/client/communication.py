import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import threading

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sentence_transformers import SentenceTransformer

from .data_processor import DataProcessor
from .models import SimpleLSTM


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
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"[CLIENT] 报到成功: {data}")
                self.is_registered = True
                return True
            else:
                print(f"[CLIENT] 报到失败: {response.status_code} - {response.text}")
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
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"[CLIENT] 高频数据提交成功")
                return True
            else:
                print(f"[CLIENT] 高频数据提交失败: {response.status_code}")
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
                timeout=10
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
                print(f"[CLIENT] 获取原型失败: {response.status_code}")
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
                timeout=5
            )
            
            if response.status_code == 200:
                self.last_heartbeat = time.time()
                return True
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
                    timeout=30
                )
                
                if response.status_code == 200:
                    print(f"[CLIENT] 原型{proto_id}的模型更新提交成功")
                else:
                    print(f"[CLIENT] 原型{proto_id}的模型更新提交失败: {response.status_code}")
                    
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