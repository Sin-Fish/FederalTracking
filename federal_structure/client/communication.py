import requests
import torch
import torch.nn as nn
import numpy as np
import json
import os  # 添加缺失的os导入
import zipfile  # 添加缺失的zipfile导入
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import requests
import threading
import time
from collections import defaultdict
from models import SimpleLSTM
from data_processor import DataProcessor  # 添加缺失的导入


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
        
        # 服务器模型哈希
        self.server_model_hash = None  # 添加这个属性的初始化
    
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
                
                # 如果服务器返回了模型哈希，加载嵌入模型
                if "model_hash" in data:
                    server_model_hash = data["model_hash"]
                    try:
                        self.load_embedding_model(server_model_hash)
                        # 将加载的模型引用也赋给data_processor，以便在聚类中使用
                        self.data_processor.embedding_model = self.embedding_model
                    except Exception as e:
                        print(f"[CLIENT] 加载嵌入模型失败: {e}")
                        return False
                
                self.is_registered = True
                return True
            else:
                print(f"[CLIENT] 报到失败: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"[CLIENT] 报到异常: {e}")
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
        # 更新当前状态，以便心跳线程可以发送正确的状态
        self.current_status = status
        
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

    def send_local_prototypes(self) -> bool:
        """发送本地聚类中心到服务器"""
        print("[CLIENT] 发送本地聚类中心到服务器...")
        
        # 获取本地聚类中心
        state_prototypes_list, action_prototypes_list = self.data_processor.get_local_prototypes()
        
        # 合并状态和动作原型
        all_local_prototypes = state_prototypes_list + action_prototypes_list
        
        if not all_local_prototypes:
            print("[CLIENT] 没有本地聚类中心可以发送")
            return False
        
        try:
            # 发送原型数据 - 直接发送列表而不是字典
            response = requests.post(
                f"{self.server_url}/api/client/prototypes?client_id={self.client_id}",
                json=all_local_prototypes,  # 直接发送原型列表
                timeout=60  # 增加超时时间
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"[CLIENT] 聚类中心发送成功: {data}")
                return True
            else:
                print(f"[CLIENT] 聚类中心发送失败: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"[CLIENT] 发送聚类中心异常: {e}")
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
                        # 更新本地服务器模型哈希
                        self.server_model_hash = server_model_hash
                        self.load_embedding_model(server_model_hash)
                    
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
            
            # 获取模型状态 - 现在模型的get_state_dict方法已确保返回可序列化类型
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
            
            print(f"[CLIENT] 正在提交原型{proto_id}的模型更新，数据量: {data_size}")
            print(f"[CLIENT] 请求将发送到: {self.server_url}/api/model/update")
            
            try:
                # 在发送请求前对整个payload进行深度清洗
                cleaned_payload = self.deep_clean_for_json(payload)
                
                # 在发送请求前输出信息
                print(f"[CLIENT] 正在发送POST请求到服务器...")
                
                response = requests.post(
                    f"{self.server_url}/api/model/update",
                    json=cleaned_payload,
                    timeout=60  # 增加超时时间
                )
                
                print(f"[CLIENT] 服务器响应状态码: {response.status_code}")
                print(f"[CLIENT] 服务器响应内容: {response.text}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"[CLIENT] 原型{proto_id}的模型更新提交成功: {result}")
                else:
                    print(f"[CLIENT] 原型{proto_id}的模型更新提交失败: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"[CLIENT] 提交模型更新超时 (超过60秒)")
            except requests.exceptions.RequestException as e:
                print(f"[CLIENT] 提交模型更新时发生网络错误: {e}")
            except Exception as e:
                print(f"[CLIENT] 提交模型更新时发生错误: {e}")
        
        print("[CLIENT] 模型更新提交完成")
    
    def deep_clean_for_json(self, obj):
        """
        递归地遍历数据结构，将所有numpy/torch类型转换为Python原生类型。
        确保整个对象可以被json.dumps()序列化。
        """
        # 处理numpy标量 (int64, float32等)
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # 处理numpy数组：转换为列表并递归清理
        if isinstance(obj, np.ndarray):
            # 特别处理空数组或标量数组
            if obj.size == 0:
                return []
            if obj.ndim == 0:  # 标量数组，如 np.array(42)
                return self.deep_clean_for_json(obj.item())
            # 递归处理多维数组
            return [self.deep_clean_for_json(item) for item in obj]
        
        # 处理PyTorch张量：先转numpy，再递归清理
        if hasattr(torch, 'Tensor') and isinstance(obj, torch.Tensor):
            # 确保在CPU上并转为numpy
            cpu_obj = obj.cpu() if obj.is_cuda else obj
            return self.deep_clean_for_json(cpu_obj.detach().numpy())
        
        # 处理列表和元组：递归清理每个元素
        if isinstance(obj, (list, tuple)):
            return [self.deep_clean_for_json(item) for item in obj]
        
        # 处理字典：递归清理键和值
        if isinstance(obj, dict):
            cleaned_dict = {}
            for key, value in obj.items():
                # 键也必须是字符串（JSON要求）
                cleaned_key = str(key) if not isinstance(key, str) else key
                cleaned_dict[cleaned_key] = self.deep_clean_for_json(value)
            return cleaned_dict
        
        # 对于其他类型（str, int, float, bool, None），直接返回
        return obj

    def start_heartbeat(self, interval: int = 30):
        """启动心跳线程"""
        self.current_status = "idle"  # 添加一个状态变量来跟踪当前状态
        
        def heartbeat_loop():
            while True:
                time.sleep(interval)
                # 发送当前状态而不是固定的idle状态
                self.send_status_update(self.current_status)
        
        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        print(f"[CLIENT] 心跳线程已启动，间隔: {interval}秒")