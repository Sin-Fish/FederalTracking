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
            
            print(f"[CLIENT] 文件下载完成")
            
            # 验证文件完整性
            downloaded_hash = self._calculate_file_hash(temp_filename)
            server_hash_response = requests.get(f"{self.server_url}/api/model/hash")
            if server_hash_response.status_code == 200:
                server_hash = server_hash_response.json().get("hash", "")
                if downloaded_hash != server_hash:
                    print(f"[CLIENT] 文件哈希验证失败")
                    return False
                else:
                    print(f"[CLIENT] 文件哈希验证通过: {server_hash[:16]}...")
            else:
                print(f"[CLIENT] 无法获取服务器文件哈希，跳过验证")
            
            # 解压文件
            print(f"[CLIENT] 解压模型文件...")
            model_cache_dir = "./model_cache"
            os.makedirs(model_cache_dir, exist_ok=True)
            
            extract_path = os.path.join(model_cache_dir, model_name)
            with zipfile.ZipFile(temp_filename, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # 验证解压后的模型
            print(f"[CLIENT] 验证解压后的模型...")
            extracted_hash = self._calculate_directory_hash(extract_path)
            print(f"[CLIENT] 模型目录哈希: {extracted_hash[:16]}...")
            
            # 清理临时文件
            os.remove(temp_filename)
            
            return True
            
        except Exception as e:
            print(f"[CLIENT] 下载模型时发生错误: {e}")
            return False
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的哈希值"""
        import hashlib
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"[CLIENT] 计算文件哈希失败: {e}")
            return ""
    
    def _calculate_directory_hash(self, dir_path: str) -> str:
        """计算目录的哈希值"""
        import hashlib
        sha256_hash = hashlib.sha256()
        
        try:
            for root, dirs, files in os.walk(dir_path):
                for file in sorted(files):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        # 添加文件名到哈希
                        rel_path = os.path.relpath(file_path, dir_path)
                        sha256_hash.update(rel_path.encode('utf-8'))
                        # 添加文件内容
                        while chunk := f.read(8192):
                            sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"[CLIENT] 计算目录哈希失败: {e}")
            return ""

    def _load_embedding_model(self, model_name: str):
        """从本地缓存加载嵌入模型"""
        cache_path = f"./model_cache/{model_name}"
        
        # 检查模型目录是否存在，如果不存在则检查嵌套目录
        actual_model_path = cache_path
        nested_path = os.path.join(cache_path, model_name)
        if os.path.exists(nested_path):
            actual_model_path = nested_path
            print(f"[CLIENT] 检测到嵌套模型目录结构，使用路径: {actual_model_path}")
        
        if not os.path.exists(actual_model_path):
            print(f"[CLIENT] 错误：模型路径不存在: {actual_model_path}")
            return None
        
        try:
            print(f"[CLIENT] 从本地缓存加载模型: {actual_model_path}")
            from sentence_transformers import SentenceTransformer
            
            # 检查模型目录是否包含必要的文件
            required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
            missing_files = []
            for file in required_files:
                if not os.path.exists(os.path.join(actual_model_path, file)):
                    missing_files.append(file)
            
            if missing_files:
                print(f"[CLIENT] 错误：模型目录缺少必要文件: {missing_files}")
                return None
            
            # 尝试加载模型
            self.embedding_model = SentenceTransformer(actual_model_path)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"[CLIENT] 模型加载成功，嵌入维度: {self.embedding_dim}")
            return self.embedding_model
        except Exception as e:
            print(f"[CLIENT] 加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None

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
                    # 加载本地缓存模型
                    model = self._load_embedding_model(self.model_name)
                    if model is not None:
                        print("[CLIENT] 嵌入模型加载成功")
                        return True
                    else:
                        print("[CLIENT] 本地模型加载失败")
                
                print(f"[CLIENT] 本地无匹配模型，从服务器下载 (Hash: {model_hash[:16]}...)")
                
                # 从服务器下载模型
                if self._download_model_from_server(self.model_name):
                    # 下载成功后，加载模型
                    model = self._load_embedding_model(self.model_name)
                    if model is not None:
                        # 保存哈希值用于后续验证
                        hash_file_path = f"{self.model_cache_path}/{self.model_name}_hash.txt"
                        with open(hash_file_path, 'w') as f:
                            f.write(model_hash)
                        print("[CLIENT] 从服务器下载的模型加载成功")
                        return True
                    else:
                        print("[CLIENT] 从服务器下载的模型加载失败")
                        return False
                else:
                    print("[CLIENT] 从服务器下载模型失败")
                    return False
            else:
                print(f"[CLIENT] 获取模型信息失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"[CLIENT] 加载嵌入模型时发生错误: {e}")
            return False
    
    def embed_behavior_strings(self, strings: List[str]) -> np.ndarray:
        """将行为字符串转换为嵌入向量"""
        success = self.load_embedding_model()
        if not success:
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
    
    def check_readiness(self) -> Optional[Dict]:
        """等待服务器发起就位检查，检查模型是否需要更新"""
        print(f"[CLIENT] 等待服务器发起就位检查...")
        
        try:
            # 发送请求到服务器的就位等待端点
            response = requests.get(
                f"{self.server_url}/api/client/wait-readiness-check",
                params={"client_id": self.client_id},
                timeout=30  # 35秒超时
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ready_check":
                    print(f"[CLIENT] 收到服务器就位检查指令: {data['message']}")
                    
                    # 检查是否需要更新模型
                    server_model_hash = data.get("model_hash")
                    if server_model_hash and not self.check_local_model(server_model_hash):
                        print("[CLIENT] 检测到模型版本不匹配，开始下载新模型...")
                        if self._download_model_from_server(self.model_name):
                            # 重新加载模型
                            self._load_embedding_model(self.model_name)
                            print("[CLIENT] 模型更新完成")
                        else:
                            print("[CLIENT] 模型更新失败")
                    
                    # 向服务器发送就位确认
                    confirm_response = requests.post(
                        f"{self.server_url}/api/client/confirm-readiness",
                        params={"client_id": self.client_id},
                        timeout=self.request_timeout
                    )
                    
                    if confirm_response.status_code == 200:
                        confirm_data = confirm_response.json()
                        print(f"[CLIENT] {confirm_data['message']}")
                    else:
                        print(f"[CLIENT] 就位确认失败: {confirm_response.status_code}")
                    
                    return data
                elif data.get("status") == "timeout":
                    print(f"[CLIENT] 等待就位检查超时")
                    return None
                else:
                    print(f"[CLIENT] 就位检查失败: {data}")
                    return None
            else:
                print(f"[CLIENT] 就位检查请求失败: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"[CLIENT] 等待就位检查超时")
            return None
        except Exception as e:
            print(f"[CLIENT] 就位检查时发生错误: {e}")
            return None
    
    def wait_for_training_start(self) -> Optional[Dict]:
        """
        长轮询等待服务器训练开始指令
        服务器将保持连接直到有训练指令
        """
        print(f"[CLIENT] 等待服务器训练开始指令...")
        
        try:
            # 发送请求，服务器将保持连接直到有指令
            response = requests.get(
                f"{self.server_url}/api/training/wait",
                params={"client_id": self.client_id},
                timeout=3600  # 设置为1小时超时，允许手动中断
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "training_start":
                    print("[CLIENT] 收到服务器训练开始指令!")
                    # 更新原型信息（如果服务器返回了新的原型）
                    if "training_info" in data and data["training_info"]:
                        training_info = data["training_info"]
                        if training_info.get("prototypes"):
                            self.global_prototypes = np.array(training_info["prototypes"])
                        if training_info.get("prototype_labels"):
                            self.prototype_labels = training_info["prototype_labels"]
                    return data
                elif data.get("status") == "timeout":
                    print("[CLIENT] 等待训练指令超时")
                    return None
                else:
                    print("[CLIENT] 未收到训练开始指令")
                    return None
            else:
                print(f"[CLIENT] 等待训练指令失败: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"[CLIENT] 等待训练指令超时")
            return None
        except KeyboardInterrupt:
            print(f"[CLIENT] 用户中断等待训练指令")
            return None
        except Exception as e:
            print(f"[CLIENT] 等待训练指令时发生错误: {e}")
            return None
    
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