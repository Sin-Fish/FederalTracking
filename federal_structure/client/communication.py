import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import threading
import os
import pickle
import zipfile
import shutil

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
        self.request_timeout = 60  # 增加请求超时时间，以适应长轮询
        
        # 嵌入模型
        self.embedding_model = None
        self.embedding_dim = 384
        
        # 本地模型缓存路径
        self.model_cache_path = "./model_cache"
        self.model_name = "all-MiniLM-L6-v2"
    
    def check_local_model(self, server_model_hash: str) -> bool:
        """检查本地是否已有指定哈希的模型"""
        hash_file_path = f"{self.model_cache_path}/{self.model_name}_hash.txt"
        if not os.path.exists(hash_file_path):
            return False
            
        try:
            with open(hash_file_path, 'r') as f:
                cached_hash = f.read().strip()
                return cached_hash == server_model_hash
        except Exception:
            return False
    
    def _download_model_from_server(self, model_name: str) -> bool:
        """从服务器下载模型"""
        import time
        last_print_time = 0  # 记录上次打印时间
        
        try:
            print(f"[CLIENT] 开始下载模型 {model_name}...")
            
            # 获取模型下载链接
            download_url = f"{self.server_url}/api/model/download"
            
            # 发送下载请求（流式下载）
            response = requests.get(download_url, stream=True, timeout=300)
            
            if response.status_code != 200:
                print(f"[CLIENT] 下载请求失败: {response.status_code}")
                return False
            
            # 获取文件总大小
            total_size = int(response.headers.get('content-length', 0))
            print(f"[CLIENT] 模型文件大小: {total_size / (1024*1024):.2f} MB")
            
            # 创建临时文件保存下载内容
            temp_dir = "./temp_downloads"
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_filename = os.path.join(temp_dir, f"{model_name}_download_{int(time.time())}.zip")
            
            # 保存下载内容到临时文件
            downloaded_size = 0
            start_time = time.time()
            
            with open(temp_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # 控制打印频率，每2秒打印一次
                        current_time = time.time()
                        if current_time - last_print_time >= 2.0:
                            progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                            print(f"[CLIENT] 下载进度: {progress:.1f}% ({downloaded_size}/{total_size} bytes)")
                            last_print_time = current_time
            
            # 创建模型缓存目录
            model_cache_dir = f"{self.model_cache_path}/{model_name}"
            os.makedirs(model_cache_dir, exist_ok=True)
            
            # 解压模型文件
            print(f"[CLIENT] 解压模型文件...")
            with zipfile.ZipFile(temp_filename, 'r') as zip_ref:
                zip_ref.extractall(model_cache_dir)
            
            # 保存模型哈希
            hash_file_path = f"{self.model_cache_path}/{model_name}_hash.txt"
            with open(hash_file_path, 'w') as f:
                f.write(self.server_model_hash)
            
            # 清理临时文件
            os.remove(temp_filename)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
            
            print(f"[CLIENT] 模型下载完成: {model_cache_dir}")
            return True
            
        except Exception as e:
            print(f"[CLIENT] 下载模型失败: {e}")
            return False
    
    def load_embedding_model(self, server_model_hash: str):
        """加载嵌入模型，如果本地没有则从服务器下载"""
        from sentence_transformers import SentenceTransformer
        
        model_path = f"{self.model_cache_path}/{self.model_name}"
        
        # 检查本地是否有匹配哈希的模型
        if self.check_local_model(server_model_hash):
            print(f"[CLIENT] 本地已存在匹配的模型，准备加载...")
            
            # 检查是否存在嵌套目录结构
            nested_path = f"{model_path}/{self.model_name}"
            if os.path.exists(nested_path):
                print(f"[CLIENT] 检测到嵌套模型目录结构，使用路径: {nested_path}")
                model_path = nested_path
            
            print(f"[CLIENT] 从本地缓存加载模型: {model_path}")
            self.embedding_model = SentenceTransformer(model_path)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        else:
            # 从服务器下载模型
            print(f"[CLIENT] 本地模型不存在或哈希不匹配，开始下载...")
            if self._download_model_from_server(self.model_name):
                # 下载成功，加载模型
                nested_path = f"{model_path}/{self.model_name}"
                if os.path.exists(nested_path):
                    print(f"[CLIENT] 检测到嵌套模型目录结构，使用路径: {nested_path}")
                    model_path = nested_path
                
                print(f"[CLIENT] 从本地缓存加载模型: {model_path}")
                self.embedding_model = SentenceTransformer(model_path)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            else:
                print(f"[CLIENT] 模型下载失败，无法继续执行")
                raise Exception("模型下载失败")
    
    def embed_behavior_strings(self, strings: List[str]) -> np.ndarray:
        """将行为字符串转换为嵌入向量"""
        if self.embedding_model is None:
            print("[CLIENT] 无法加载嵌入模型，使用随机向量作为嵌入（备用方案）")
            embeddings = np.random.rand(len(strings), self.embedding_dim).astype(np.float32)
            return embeddings
        
        print(f"[CLIENT] 嵌入 {len(strings)} 个行为字符串...")
        embeddings = self.embedding_model.encode(strings, show_progress_bar=False)
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
                timeout=30  # 增加超时时间
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"[CLIENT] 报到成功: {data}")
                
                # 检查服务器返回的模型哈希
                self.server_model_hash = data.get("model_hash")
                self.n_prototypes = data.get("n_prototypes", 5)
                
                # 加载嵌入模型
                self.load_embedding_model(self.server_model_hash)
                
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
            self.data_processor.behavior_embeddings = self.embed_behavior_strings(
                self.data_processor.behavior_strings
            )
        
        # 为每个本地向量找到最近的原型
        from sklearn.metrics.pairwise import pairwise_distances_argmin
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
    
    def check_readiness(self):
        """检查客户端就位状态"""
        print(f"[CLIENT] 等待服务器发起就位检查...")
        
        try:
            # 等待服务器发起就位检查（长轮询）
            response = requests.get(
                f"{self.server_url}/api/client/wait-readiness-check?client_id={self.client_id}",
                timeout=360  # 6分钟超时，给服务器足够时间
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"[CLIENT] 收到服务器就位检查指令: {data['message']}")
                
                # 确认就位状态
                confirm_response = requests.post(
                    f"{self.server_url}/api/client/confirm-readiness?client_id={self.client_id}",
                    timeout=10
                )
                
                if confirm_response.status_code == 200:
                    confirm_data = confirm_response.json()
                    print(f"[CLIENT] 客户端 {self.client_id} 就位确认成功")
                    
                    # 检查服务器返回的模型哈希
                    server_model_hash = data.get('model_hash')
                    if server_model_hash and self.server_model_hash != server_model_hash:
                        print(f"[CLIENT] 模型哈希不匹配，下载新模型...")
                        self.download_model_from_server()
                    
                    return True
                else:
                    print(f"[CLIENT] 就位确认失败: {confirm_response.status_code}")
                    return False
            else:
                print(f"[CLIENT] 等待就位检查指令失败: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[CLIENT] 等待就位检查时发生错误: {e}")
            return False
    
    def wait_for_training_start(self):
        """等待服务器训练开始指令"""
        print("[CLIENT] 等待服务器训练开始指令...")
        
        try:
            response = requests.get(
                f"{self.server_url}/api/training/wait",
                params={"client_id": self.client_id},
                timeout=60  # 1分钟超时
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "training_start":
                    print(f"[CLIENT] 收到服务器训练开始指令!")
                    return data
                else:
                    print(f"[CLIENT] 未收到训练开始指令: {data}")
                    return None
            else:
                print(f"[CLIENT] 等待训练开始失败: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[CLIENT] 等待训练开始时发生错误: {e}")
            return None
    
    def submit_model_updates(self, local_models: Dict[int, SimpleLSTM], current_round: int, epochs: int):
        """提交模型更新到服务器"""
        print("[CLIENT] 提交模型更新...")
        
        for proto_id, model in local_models.items():
            if proto_id not in self.prototype_mapping:
                print(f"[CLIENT] 原型 {proto_id} 没有分配数据，跳过提交")
                continue
                
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
                    "epochs": epochs,
                    "data_points": data_size,
                    "submitted_at": datetime.now().isoformat()
                }
            }
            
            try:
                response = requests.post(
                    f"{self.server_url}/api/model/update",
                    json=payload,
                    timeout=60  # 增加超时时间
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
    
    def cluster_local_data(self):
        """对本地数据进行聚类"""
        # 确保嵌入模型已加载
        if self.embedding_model is None:
            print("[CLIENT] 加载嵌入模型用于本地聚类...")
            self.load_embedding_model()
        
        # 使用本地行为字符串进行聚类
        behavior_strings = self.data_processor.behavior_strings
        if len(behavior_strings) == 0:
            print("[CLIENT] 错误：没有可用的行为数据进行聚类")
            return None
        
        # 生成嵌入向量
        print(f"[CLIENT] 为 {len(behavior_strings)} 个行为字符串生成嵌入向量...")
        embeddings = self.embedding_model.encode(behavior_strings)
        
        # 使用K-Means聚类
        from sklearn.cluster import KMeans
        n_clusters = min(len(behavior_strings), self.n_prototypes or 2)  # 使用服务器指定的原型数量
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 获取聚类中心
        cluster_centers = kmeans.cluster_centers_
        
        print(f"[CLIENT] 本地数据聚类完成，生成 {n_clusters} 个聚类中心")
        return cluster_centers.tolist()

    def send_local_prototypes(self):
        """发送本地聚类中心到服务器"""
        try:
            print(f"[CLIENT] 开始对本地数据进行聚类...")
            local_prototypes = self.cluster_local_data()
            
            if local_prototypes is None:
                print("[CLIENT] 本地聚类失败，无法发送聚类中心")
                return False
            
            # 发送聚类中心到服务器
            url = f"{self.server_url}/api/client/prototypes?client_id={self.client_id}"
            response = requests.post(url, json=local_prototypes, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"[CLIENT] 聚类中心发送成功: {result}")
                return True
            else:
                print(f"[CLIENT] 聚类中心发送失败: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            print(f"[CLIENT] 发送聚类中心时发生错误: {e}")
            return False
