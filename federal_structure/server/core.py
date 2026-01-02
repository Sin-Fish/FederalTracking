import os
# 设置环境变量以禁用tokenizers并行处理警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio
import json
import hashlib
import time
import pickle
import os
import zipfile
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque

import numpy as np
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse, FileResponse

# 机器学习相关导入（延迟导入）
from ml_models import SimpleLSTM
from models import ModelUpdate, ClientStatus


class FederatedServerCore:
    def __init__(self, n_prototypes=5):
        self.n_prototypes = n_prototypes
        
        # 状态管理
        self.client_registry: Dict[str, Dict] = {}  # 客户端注册表
        self.training_queue: deque = deque()  # 训练队列
        self.status_check_interval = 60  # 状态检查间隔(秒)
        
        # 联邦学习状态
        self.global_prototypes: Optional[np.ndarray] = None  # 全局原型向量 [n_prototypes, embedding_dim]
        self.prototype_labels: List[str] = []  # 原型标签
        self.global_models: Dict[int, SimpleLSTM] = {}  # 全局模型 {prototype_id: model}
        self.model_updates: Dict[int, List[Dict]] = defaultdict(list)  # 模型更新池
        
        # 嵌入模型
        self.embedding_model = None
        self.embedding_model_hash = None
        self.embedding_dim = 384
        
        # 训练状态跟踪
        self.training_start_events: Dict[str, asyncio.Event] = {}  # 用于跟踪客户端训练开始事件
        self.readiness_check_events: Dict[str, asyncio.Event] = {}  # 用于跟踪客户端就位检查事件
        
        # 存储客户端聚类中心
        self.client_prototypes: Dict[str, np.ndarray] = {}  # 存储客户端发回的聚类中心
        self.pending_clients_for_aggregation: set = set()  # 等待聚合的客户端
        
        # 跟踪客户端训练完成状态
        self.client_finished_prototypes: Dict[str, set] = defaultdict(set)  # 跟踪每个客户端已完成的原型
        self.client_training_status: Dict[str, str] = {}  # 跟踪客户端当前状态（training/finished等）
        self.client_expected_prototypes_count: Dict[str, int] = {}  # 记录每个客户端应该完成的原型数量
        
        # 跟踪已标记为离线的客户端，避免重复打印
        self.offline_clients: set = set()
        
        # 跟踪客户端超时次数，多次超时后移除客户端
        self.client_timeout_counts: Dict[str, int] = {}
        
        # 初始化时尝试加载本地模型
        self._initialize_embedding_model()
        
        # 状态监控任务 - 将在 start() 中启动
        self.monitor_task: Optional[asyncio.Task] = None
    
    def _initialize_embedding_model(self):
        """初始化嵌入模型，优先从本地加载"""
        model_name = "all-MiniLM-L6-v2"
        model_dir = f"./models/{model_name}"
        
        print(f"[SERVER] 初始化嵌入模型111...")
        
        # 检查是否可以导入SentenceTransformer
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            print(f"[SERVER] 无法导入SentenceTransformer: {e}")
            print("[SERVER] 请运行 'pip install -r requirements.txt' 安装依赖")
            raise HTTPException(
                status_code=500,
                detail=f"无法导入SentenceTransformer: {e}"
            )
        
        if os.path.exists(model_dir):
            try:
                print(f"[SERVER] 从本地加载模型: {model_dir}")
                self.embedding_model = SentenceTransformer(model_dir)
                
                # 计算并存储模型哈希
                self.embedding_model_hash = self._compute_model_hash(model_dir)
                print(f"[SERVER] 本地模型加载成功，哈希: {self.embedding_model_hash[:16]}")
            except Exception as e:
                print(f"[SERVER] 从本地加载模型失败: {e}，将重新下载")
                self._download_model_sync(model_name)
        else:
            print(f"[SERVER] 本地模型不存在: {model_dir}，开始下载")
            self._download_model_sync(model_name)

    def _compute_model_hash(self, model_path: str) -> str:
        """计算模型目录的哈希值"""
        import hashlib
        import os
        
        hash_obj = hashlib.sha256()
        
        for root, dirs, files in os.walk(model_path):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    while chunk := f.read(8192):
                        hash_obj.update(chunk)
        
        return hash_obj.hexdigest()

    def _download_model_sync(self, model_name: str):
        """同步下载模型，优先尝试从本地加载"""
        try:
            print(f"[SERVER] 正在加载嵌入模型 {model_name}...")
            
            models_dir = "./models"
            os.makedirs(models_dir, exist_ok=True)
            
            # 检查是否可以导入SentenceTransformer
            from sentence_transformers import SentenceTransformer
            
            model_path = f"{models_dir}/{model_name}"
            
            # 首先尝试从本地路径加载
            if os.path.exists(model_path):
                print(f"[SERVER] 从本地路径加载模型: {model_path}")
                model = SentenceTransformer(model_path)
            else:
                print(f"[SERVER] 本地模型不存在: {model_path}，尝试在线下载...")
                
                # 下载模型
                model = SentenceTransformer(model_name)
                model.save(f"{models_dir}/{model_name}")
                print(f"[SERVER] 模型下载完成，保存至 {models_dir}/{model_name}")
            
            # 加载模型
            self.embedding_model = model
            
            # 计算并存储模型哈希
            self.embedding_model_hash = self._compute_model_hash(f"{models_dir}/{model_name}")
            print(f"[SERVER] 模型哈希: {self.embedding_model_hash[:16]}")
            
        except Exception as e:
            print(f"[SERVER] 加载模型失败: {e}")
            print(f"[SERVER] 尝试使用备用方案...")
            
            # 如果下载失败，使用随机嵌入作为备用方案
            try:
                import numpy as np
                
                class MockSentenceTransformer:
                    def __init__(self):
                        self.embedding_dim = 384  # all-MiniLM-L6-v2的维度
                        print(f"[SERVER] 已创建随机嵌入生成器，服务器将继续运行")
                    
                    def encode(self, sentences, **kwargs):
                        import numpy as np
                        # 返回随机向量作为嵌入
                        embedding_dim = 384  # all-MiniLM-L6-v2的维度
                        embeddings = np.random.rand(len(sentences), embedding_dim).astype(np.float32)
                        return embeddings
                
                self.embedding_model = MockSentenceTransformer()
                self.embedding_model_hash = "fallback_model"
                
                print("[SERVER] 已创建随机嵌入生成器，服务器将继续运行")
                
            except Exception as e2:
                print(f"[SERVER] 备用方案也失败了: {e2}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"无法加载模型: {e}，备用方案也失败: {e2}"
                )

    # ==================== 核心业务逻辑 ====================
    async def handle_client_register(self, req):
        """处理客户端报到"""
        client_id = req.client_id
        
        # 检查是否已注册
        if client_id in self.client_registry:
            # 更新指纹和报到时间
            self.client_registry[client_id].update({
                "last_heartbeat": time.time(),
                "data_fingerprint": req.data_fingerprint,
                "status": "idle",
                "last_seen": datetime.now().isoformat()
            })
            
            # 验证嵌入模型一致性
            if hasattr(req, 'embedding_hash') and req.embedding_hash != self.embedding_model_hash:
                print(f"[SERVER] 客户端 {client_id} 嵌入模型哈希不匹配")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "status": "error",
                        "message": "嵌入模型版本不匹配，请更新模型",
                        "server_model_hash": self.embedding_model_hash,
                        "client_model_hash": req.embedding_hash
                    }
                )
            
            # 接收并存储客户端聚类中心（如果提供）
            if hasattr(req, 'prototypes') and req.prototypes is not None:
                self.client_prototypes[client_id] = np.array(req.prototypes)
                print(f"[SERVER] 已接收客户端 {client_id} 的聚类中心: {len(req.prototypes)} 个")
            
            return {"status": "updated", "client_id": client_id}
        
        # 新客户端注册
        client_info = {
            "id": client_id,
            "data_fingerprint": req.data_fingerprint,
            "embedding_version": req.embedding_version,
            "sample_count": req.sample_count,
            "status": "idle",
            "registered_at": datetime.now().isoformat(),
            "last_heartbeat": time.time(),
            "last_seen": datetime.now().isoformat(),
            "training_round": 0,
            "in_queue": False
        }
        
        self.client_registry[client_id] = client_info
        
        # 为客户端创建训练开始事件和就位检查事件
        self.training_start_events[client_id] = asyncio.Event()
        self.readiness_check_events[client_id] = asyncio.Event()
        
        # 验证嵌入模型一致性
        if hasattr(req, 'embedding_hash') and req.embedding_hash != self.embedding_model_hash:
            print(f"[SERVER] 客户端 {client_id} 嵌入模型哈希不匹配")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "message": "嵌入模型版本不匹配，请更新模型",
                    "server_model_hash": self.embedding_model_hash,
                    "client_model_hash": req.embedding_hash
                }
            )
        
        # 验证数据格式（这里可以添加更复杂的验证逻辑）
        if req.sample_count < 100:
            return JSONResponse(
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                content={
                    "status": "warning",
                    "message": "数据量较少，可能影响训练效果",
                    "client_id": client_id,
                    "model_hash": self.embedding_model_hash,
                    "n_prototypes": self.n_prototypes
                }
            )
        
        # 接收并存储客户端聚类中心（如果提供）
        if hasattr(req, 'prototypes') and req.prototypes is not None:
            self.client_prototypes[client_id] = np.array(req.prototypes)
            print(f"[SERVER] 已接收客户端 {client_id} 的聚类中心: {len(req.prototypes)} 个")
        
        # 自动加入训练队列（根据你的策略可调整）
        if not client_info["in_queue"]:
            self.training_queue.append(client_id)
            client_info["in_queue"] = True
        
        print(f"[SERVER] 客户端 {client_id} 报到成功，样本数: {req.sample_count}")
        return {
            "status": "registered", 
            "client_id": client_id, 
            "queue_position": len(self.training_queue),
            "model_hash": self.embedding_model_hash,
            "n_prototypes": self.n_prototypes
        }
    
    async def handle_status_update(self, status_update: ClientStatus):
        """处理客户端状态更新"""
        client_id = status_update.client_id
        
        if client_id not in self.client_registry:
            raise HTTPException(status_code=404, detail="客户端未注册")
        
        client = self.client_registry[client_id]
        client["status"] = status_update.status
        client["last_heartbeat"] = time.time()
        client["last_seen"] = datetime.now().isoformat()
        
        if status_update.progress is not None:
            client["progress"] = status_update.progress
        
        # 如果训练完成或中断，从队列中移除
        if status_update.status in ["finished", "interrupted", "offline"]:
            client["in_queue"] = False
            if client_id in self.training_queue:
                self.training_queue.remove(client_id)
            
            # 检查是否所有客户端都已完成训练和模型上传
            if status_update.status == "finished":
                await self.check_all_clients_finished()
        
        print(f"[SERVER] 客户端 {client_id} 状态更新: {status_update.status}")
        return {"status": "updated"}
    
    async def check_all_clients_finished(self):
        """检查是否所有客户端都已完成训练和模型上传，如果是则触发聚合"""
        # 获取所有曾经上传过原型的客户端
        clients_with_prototypes = set(self.client_expected_prototypes_count.keys())
        
        # 检查这些客户端是否都完成了相应数量的模型上传
        all_clients_finished = True
        unfinished_clients = []
        
        for client_id in clients_with_prototypes:
            expected_count = self.client_expected_prototypes_count[client_id]
            actual_uploaded = len(self.client_finished_prototypes[client_id])
            
            if actual_uploaded < expected_count:
                all_clients_finished = False
                unfinished_clients.append(f"{client_id} ({actual_uploaded}/{expected_count})")
        
        if all_clients_finished and len(clients_with_prototypes) > 0:
            print(f"[SERVER] 所有 {len(clients_with_prototypes)} 个客户端已完成训练和模型上传，准备聚合")
            # 检查是否每个原型都有至少一个更新
            all_prototypes_have_updates = True
            for pid in range(self.n_prototypes):
                if len(self.model_updates[pid]) == 0:
                    print(f"[SERVER] 原型 {pid} 还没有收到任何更新，无法聚合")
                    all_prototypes_have_updates = False
                    break
            
            if all_prototypes_have_updates:
                print("[SERVER] 所有原型都有更新，开始聚合")
                await self.perform_federated_averaging()
            else:
                print("[SERVER] 某些原型缺少更新，无法聚合")
        else:
            # 检查是否所有注册的客户端都处于完成或离线状态
            all_registered_clients = set(self.client_registry.keys())
            active_clients = set()  # 仍在训练或活动中的客户端
            
            for client_id in all_registered_clients:
                client_status = self.client_registry[client_id].get("status", "unknown")
                # 如果客户端状态不是finished或offline，则认为仍在活动
                if client_status not in ["finished", "offline", "interrupted"]:
                    active_clients.add(client_id)
            
            # 如果没有活动的客户端，开始聚合
            if len(active_clients) == 0 and len(self.model_updates) > 0:
                print(f"[SERVER] 所有注册客户端都已完成或离线，开始聚合，共 {len(clients_with_prototypes)} 个客户端参与")
                
                # 检查是否每个原型都有至少一个更新
                all_prototypes_have_updates = True
                for pid in range(self.n_prototypes):
                    if len(self.model_updates[pid]) == 0:
                        print(f"[SERVER] 原型 {pid} 还没有收到任何更新")
                    
                if any(len(self.model_updates[pid]) > 0 for pid in range(self.n_prototypes)):
                    print("[SERVER] 部分原型有更新，开始聚合")
                    await self.perform_federated_averaging()
                else:
                    print("[SERVER] 没有任何原型有更新，无法聚合")
            elif not all_clients_finished:
                print(f"[SERVER] 尚有客户端未完成模型上传: {', '.join(unfinished_clients)}")
                print(f"[SERVER] 活跃客户端: {list(active_clients)}")
                print(f"[SERVER] 继续等待...")
    
    async def handle_data_collection(self, submission):
        """处理数据征收请求"""
        client_id = submission.client_id
        
        if client_id not in self.client_registry:
            raise HTTPException(status_code=404, detail="客户端未注册")
        
        # 存储高频行为数据
        self.client_registry[client_id]["high_freq_data"] = submission.high_freq_actions
        self.client_registry[client_id]["data_submitted"] = True
        self.client_registry[client_id]["submitted_at"] = datetime.now().isoformat()
        
        print(f"[SERVER] 收到客户端 {client_id} 的高频数据，数量: {len(submission.high_freq_actions)}")
        
        return {"status": "collected", "action_count": len(submission.high_freq_actions)}
    
    async def generate_global_prototypes(self):
        """生成全局行为原型 - 旧方法，现在只在没有客户端聚类中心时使用"""
        print("[SERVER] 开始生成全局行为原型...")
        
        # 收集所有高频行为数据
        all_samples = []
        for client_id in self.training_queue:
            if "high_freq_data" in self.client_registry[client_id]:
                all_samples.extend(self.client_registry[client_id]["high_freq_data"])
        
        if len(all_samples) < self.n_prototypes:
            print(f"[SERVER] 警告: 样本数({len(all_samples)})少于原型数({self.n_prototypes})")
            return {"status": "insufficient_data"}
        
        # 确保嵌入模型已加载
        if self.embedding_model is None:
            print("[SERVER] 错误：嵌入模型未加载")
            return {"status": "model_not_loaded"}
        
        # 转换为向量
        print(f"[SERVER] 正在嵌入 {len(all_samples)} 个行为样本...")
        embeddings = self.embedding_model.encode(all_samples, show_progress_bar=False)
        
        # K-Means聚类生成原型
        print(f"[SERVER] 正在聚类生成 {self.n_prototypes} 个全局原型...")
        from sklearn.cluster import KMeans
        from collections import Counter
        kmeans = KMeans(n_clusters=self.n_prototypes, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 保存原型
        self.global_prototypes = kmeans.cluster_centers_
        
        # 为每个原型生成描述性标签
        self.prototype_labels = []
        for i in range(self.n_prototypes):
            # 找到该簇的样本
            cluster_samples = [all_samples[j] for j in range(len(all_samples)) 
                             if cluster_labels[j] == i]
            # 取前3个最常见的词汇作为标签
            if cluster_samples:
                words = " ".join(cluster_samples[:5]).replace("_", " ").split()
                common_words = Counter(words).most_common(3)
                label = f"原型{i}: " + ", ".join([word for word, _ in common_words])
            else:
                label = f"原型{i}"
            self.prototype_labels.append(label)
        
        print(f"[SERVER] 全局原型生成完成!")
        
        # 初始化全局模型
        self.initialize_global_models()
        
        return {
            "status": "prototypes_generated",
            "prototype_count": self.n_prototypes,
            "labels": self.prototype_labels
        }
    
    def initialize_global_models(self):
        """为每个原型初始化一个全局LSTM模型"""
        for proto_id in range(self.n_prototypes):
            model = SimpleLSTM()
            self.global_models[proto_id] = model
        print(f"[SERVER] 已初始化 {self.n_prototypes} 个全局模型")
    
    async def initiate_training(self):
        """启动联邦训练"""
        if len(self.training_queue) == 0:
            raise HTTPException(status_code=400, detail="训练队列为空")
        
        print(f"[SERVER] 启动联邦训练，参与客户端: {len(self.training_queue)}")
        
        # 首先使用客户端发回的聚类中心进行第二次聚类，生成全局原型
        await self.aggregate_client_prototypes()
        
        if self.global_prototypes is None:
            raise HTTPException(status_code=400, detail="请先生成全局原型")
        
        # 然后检查所有客户端的就位状态
        ready_clients = await self.check_client_readiness()
        
        if len(ready_clients) == 0:
            return {
                "status": "error",
                "message": "没有就位的客户端，无法开始训练"
            }
        
        # 重置模型更新池
        self.model_updates.clear()
        
        # 准备训练指令
        training_info = {
            "prototypes": self.global_prototypes.tolist() if self.global_prototypes is not None else None,
            "prototype_labels": self.prototype_labels,
            "model_arch": "SimpleLSTM",
            "model_config": {
                "input_size": 384,
                "hidden_size": 128,
                "output_size": 100
            },
            "clients": ready_clients,
            "model_hash": self.embedding_model_hash
        }
        
        # 更新客户端状态
        for client_id in ready_clients:
            if client_id in self.client_registry:
                self.client_registry[client_id]["status"] = "training"
                self.client_registry[client_id]["progress"] = 0.0
                self.client_registry[client_id]["training_round"] += 1
        
        # 主动向所有就位的客户端发送训练开始指令
        await self.notify_clients_training_start(training_info)
        
        return {
            "status": "training_started",
            "client_count": len(ready_clients),
            "training_info": training_info
        }
    
    async def aggregate_client_prototypes(self):
        """聚合客户端发回的聚类中心，重新进行聚类"""
        print("[SERVER] 开始聚合客户端聚类中心...")
        
        # 收集所有在训练队列中的客户端的聚类中心
        all_client_prototypes = []
        for client_id, prototypes in self.client_prototypes.items():
            if client_id in self.training_queue:
                all_client_prototypes.extend(prototypes)
        
        if len(all_client_prototypes) == 0:
            print("[SERVER] 没有收集到任何客户端聚类中心")
            return
        
        print(f"[SERVER] 收集到 {len(all_client_prototypes)} 个客户端聚类中心")
        
        # 使用K-Means重新聚类，得到新的全局原型
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_prototypes, random_state=42, n_init=10)
        kmeans.fit(all_client_prototypes)
        
        # 更新全局原型为中心点
        self.global_prototypes = kmeans.cluster_centers_
        print(f"[SERVER] 重新聚类完成，更新了 {self.n_prototypes} 个全局原型")
        
        # 为每个原型生成描述性标签
        self.prototype_labels = [f"原型{i}: 聚类中心" for i in range(self.n_prototypes)]
    
    async def check_client_readiness(self) -> List[str]:
        """检查客户端就位状态"""
        print(f"[SERVER] 检查 {len(self.training_queue)} 个客户端的就位状态...")
        
        ready_clients = []
        unresponsive_clients = []
        
        # 清除之前的所有就位确认状态
        for client_id in self.client_registry:
            if "readiness_confirmed" in self.client_registry[client_id]:
                del self.client_registry[client_id]["readiness_confirmed"]
            if "readiness_confirmed_at" in self.client_registry[client_id]:
                del self.client_registry[client_id]["readiness_confirmed_at"]
        
        # 为每个客户端重置就位检查事件
        for client_id in list(self.training_queue):
            if client_id in self.readiness_check_events:
                self.readiness_check_events[client_id].clear()
        
        # 向所有客户端发送就位检查请求（通过设置事件，触发客户端的长轮询响应）
        for client_id in list(self.training_queue):
            print(f"[SERVER] 检查客户端 {client_id} 就位状态...")
            
            # 设置事件，通知客户端进行就位检查
            if client_id in self.readiness_check_events:
                self.readiness_check_events[client_id].set()
                print(f"[SERVER] 已向客户端 {client_id} 发送就位检查信号")
        
        # 等待客户端响应（最多等待30秒）
        print("[SERVER] 等待客户端响应就位检查...")
        await asyncio.sleep(30.0)
        
        # 检查哪些客户端已经确认就位
        for client_id in list(self.training_queue):
            client_info = self.client_registry.get(client_id, {})
            
            # 检查客户端是否确认了就位
            if client_info.get("readiness_confirmed", False):
                ready_clients.append(client_id)
                print(f"[SERVER] 客户端 {client_id} 已确认就位")
                
                # 更新客户端状态为就绪
                if client_id in self.client_registry:
                    self.client_registry[client_id]["status"] = "ready"
            else:
                print(f"[SERVER] 客户端 {client_id} 未响应就位检查，从队列中移除")
                unresponsive_clients.append(client_id)
                if client_id in self.training_queue:
                    self.training_queue.remove(client_id)
                if client_id in self.client_registry:
                    self.client_registry[client_id]["in_queue"] = False
                    self.client_registry[client_id]["status"] = "offline"
        
        print(f"[SERVER] 就位检查完成: {len(ready_clients)} 个客户端就位, {len(unresponsive_clients)} 个客户端无响应")
        return ready_clients
    
    async def notify_clients_training_start(self, training_info: Dict):
        """主动向所有客户端发送训练开始指令"""
        print(f"[SERVER] 向 {len(training_info['clients'])} 个客户端发送训练开始指令...")
        
        # 为每个客户端设置训练开始事件
        for client_id in training_info['clients']:
            if client_id in self.client_registry:
                if client_id in self.training_start_events:
                    # 设置事件，通知客户端训练已开始
                    self.training_start_events[client_id].set()
                    print(f"[SERVER] 已通知客户端 {client_id} 开始训练")
        
        print(f"[SERVER] 训练开始指令已发送给所有客户端")

    async def handle_model_update(self, update: ModelUpdate):
        """处理模型更新"""
        client_id = update.client_id
        prototype_id = update.prototype_id
        
        if client_id not in self.client_registry:
            raise HTTPException(status_code=404, detail="客户端未注册")
        
        # 验证原型ID是否在有效范围内
        if prototype_id >= self.n_prototypes:
            raise HTTPException(status_code=400, detail="原型ID超出范围")
        
        # 保存模型更新
        self.model_updates[prototype_id].append({
            "client_id": client_id,
            "model_state_dict": update.model_state_dict,
            "data_size": update.data_size,
            "metadata": update.metadata
        })
        
        # 记录客户端已完成的原型
        self.client_finished_prototypes[client_id].add(prototype_id)
        
        print(f"[SERVER] 收到客户端 {client_id} 原型 {prototype_id} 的模型更新，数据量: {update.data_size}")
        
        # 打印当前模型更新状态
        print("[SERVER] 当前模型更新状态:")
        for pid in range(self.n_prototypes):
            count = len(self.model_updates[pid])
            print(f"  原型 {pid}: {count} 个更新")
        
        return {"status": "received", "update_id": f"{client_id}_{prototype_id}"}
    
    async def receive_client_prototypes(self, client_id: str, prototypes: List[List[float]]):
        """接收客户端发回的聚类中心"""
        print(f"[SERVER] 收到客户端 {client_id} 的聚类中心，数量: {len(prototypes)}")
        
        # 验证客户端是否已注册
        if client_id not in self.client_registry:
            return {"status": "error", "message": "客户端未注册"}
        
        # 将聚类中心存储到字典中
        self.client_prototypes[client_id] = np.array(prototypes)
        
        # 记录该客户端应该完成的原型数量
        self.client_expected_prototypes_count[client_id] = len(prototypes)
        
        # 确保客户端在训练队列中
        if client_id not in self.training_queue:
            self.training_queue.append(client_id)
            self.client_registry[client_id]["in_queue"] = True
        
        return {
            "status": "received",
            "message": f"成功接收客户端 {client_id} 的 {len(prototypes)} 个聚类中心",
            "client_id": client_id
        }
    
    async def perform_federated_averaging(self):
        """执行联邦平均"""
        print("[SERVER] 执行联邦平均...")
        
        if not self.model_updates:
            print("[SERVER] 没有模型更新可用于聚合")
            return {"status": "no_updates"}
        
        # 修复：遍历实际存在的原型ID，而不是range(self.n_prototypes)
        for prototype_id in list(self.model_updates.keys()):
            updates = self.model_updates[prototype_id]
            if not updates:
                continue
            
            # 获取第一个更新的模型状态作为基准
            base_state = updates[0]["model_state_dict"]
            
            # 创建平均状态字典
            avg_state = {}
            for key in base_state:
                # 计算加权平均
                weighted_sum = None
                total_weight = 0
                
                for update in updates:
                    weight = update["data_size"]  # 使用数据量作为权重
                    value = np.array(update["model_state_dict"][key])
                    
                    if weighted_sum is None:
                        weighted_sum = value * weight
                    else:
                        weighted_sum += value * weight
                    
                    total_weight += weight
                
                avg_state[key] = weighted_sum / total_weight
            
            # 确保全局模型存在，如果不存在则根据客户端模型的参数创建
            if prototype_id not in self.global_models:
                print(f"[SERVER] 为原型 {prototype_id} 初始化全局模型")
                
                # 从第一个模型更新中推断正确的模型参数
                first_update_state = updates[0]["model_state_dict"]
                
                # 从状态字典中推断模型参数（现在是列表格式）
                # 转换为numpy数组以获取形状信息
                fc_weight_array = np.array(first_update_state['fc.weight'])
                output_size = fc_weight_array.shape[0]  # 输出维度
                hidden_size = fc_weight_array.shape[1]  # 隐藏层维度
                
                lstm_ih_weight_array = np.array(first_update_state['lstm.weight_ih_l0'])
                input_size = lstm_ih_weight_array.shape[1]  # 输入维度
                
                # 创建具有正确参数的模型
                self.global_models[prototype_id] = SimpleLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size
                )
            
            # 更新全局模型
            self.global_models[prototype_id].load_state_dict(avg_state)
        
        # 清空模型更新
        self.model_updates.clear()
        print(f"[SERVER] 联邦平均完成，聚合了 {len(self.global_models)} 个原型的模型")
        
        # 保存聚合后的模型
        self.save_aggregated_models()
        
        # 重置客户端完成状态，为下一轮训练做准备
        self.client_finished_prototypes.clear()
        
        # 测试聚合模型
        await self.test_aggregated_models()
        
        return {
            "status": "aggregated",
            "aggregated_prototypes": len(self.global_models),
            "model_updates_count": sum(len(updates) for updates in self.model_updates.values())
        }
    
    def save_aggregated_models(self):
        """保存聚合后的模型"""
        import os
        import pickle
        from datetime import datetime
        
        # 创建模型保存目录
        models_dir = "./aggregated_models"
        os.makedirs(models_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{models_dir}/aggregated_models_{timestamp}.pkl"
        
        # 准备保存的数据
        models_state_dict = {}
        for proto_id, model in self.global_models.items():
            models_state_dict[proto_id] = model.state_dict()
        
        save_data = {
            "models": models_state_dict,
            "prototypes": self.global_prototypes.tolist() if self.global_prototypes is not None else None,
            "prototype_labels": self.prototype_labels,
            "timestamp": timestamp,
            "n_prototypes": self.n_prototypes
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"[SERVER] 聚合模型已保存到: {filename}")
    
    async def test_aggregated_models(self):
        """使用测试数据测试聚合后的模型"""
        print("[SERVER] 开始测试聚合模型...")
        
        # 查找最新的测试数据文件
        test_data_dir = "./test_data"
        if not os.path.exists(test_data_dir):
            print(f"[SERVER] 测试数据目录不存在: {test_data_dir}")
            return
        
        # 获取所有jsonl格式的测试数据文件
        test_files = [f for f in os.listdir(test_data_dir) if f.endswith('.jsonl')]
        if not test_files:
            print(f"[SERVER] 在 {test_data_dir} 中未找到测试数据文件")
            return
        
        # 选择最新的测试数据文件
        latest_test_file = max(
            [os.path.join(test_data_dir, f) for f in test_files],
            key=os.path.getctime
        )
        
        # 查找最新保存的聚合模型文件
        aggregated_models_dir = "./aggregated_models"
        if not os.path.exists(aggregated_models_dir):
            print(f"[SERVER] 聚合模型目录不存在: {aggregated_models_dir}")
            return
        
        model_files = [f for f in os.listdir(aggregated_models_dir) if f.endswith('.pkl')]
        if not model_files:
            print(f"[SERVER] 在 {aggregated_models_dir} 中未找到聚合模型文件")
            return
        
        # 选择最新的模型文件
        latest_model_file = max(
            [os.path.join(aggregated_models_dir, f) for f in model_files],
            key=os.path.getctime
        )
        
        print(f"[SERVER] 使用测试数据: {latest_test_file}")
        print(f"[SERVER] 使用聚合模型: {latest_model_file}")
        
        # 导入测试模块并运行测试
        try:
            from test_model import test_aggregated_model
            loop = asyncio.get_event_loop()
            # 在线程池中运行测试，避免阻塞服务器
            await loop.run_in_executor(None, test_aggregated_model, latest_test_file, latest_model_file)
            
            print("[SERVER] 模型测试完成")
        except ImportError:
            print("[SERVER] 无法导入测试模块 test_model.py")
        except Exception as e:
            print(f"[SERVER] 模型测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    
    async def get_model_info(self):
        """获取模型信息"""
        return {
            "model_hash": self.embedding_model_hash,
            "embedding_dim": self.embedding_dim,
            "model_available": self.embedding_model is not None,
            "prototypes_count": len(self.prototype_labels) if self.prototype_labels else 0,
            "client_count": len(self.client_registry),
            "training_queue_size": len(self.training_queue)
        }
    
    async def start(self):
        """启动服务器异步服务"""
        if self.monitor_task is None or self.monitor_task.done():
            self.monitor_task = asyncio.create_task(self.status_monitor())
            print("[SERVER] 状态监控任务已启动")

    async def stop(self):
        """停止服务器异步服务"""
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            finally:
                self.monitor_task = None
        print("[SERVER] 状态监控任务已停止")

    async def status_monitor(self):
        """监控客户端状态"""
        while True:
            try:
                current_time = time.time()
                
                # 创建客户端副本以避免运行时修改错误
                clients_copy = dict(self.client_registry)
                
                # 检查超时的客户端
                for client_id, client_info in clients_copy.items():
                    last_seen = client_info.get("last_heartbeat", 0)
                    
                    # 如果客户端超过3个心跳间隔没有响应，则标记为离线
                    if current_time - last_seen > self.status_check_interval * 3:
                        # 检查是否已经记录过此客户端的超时
                        if client_id not in self.offline_clients:
                            # 增加超时计数
                            self.client_timeout_counts[client_id] = self.client_timeout_counts.get(client_id, 0) + 1
                            
                            # 如果超时次数超过3次，则从注册表和队列中移除客户端
                            if self.client_timeout_counts[client_id] >= 3:
                                print(f"[SERVER] 客户端 {client_id} 连续超时，取消注册并从队列中移除")
                                
                                # 从训练队列中移除
                                if client_id in self.training_queue:
                                    try:
                                        self.training_queue.remove(client_id)
                                    except ValueError:
                                        pass  # 已经被移除
                                
                                # 从注册表中移除
                                if client_id in self.client_registry:
                                    del self.client_registry[client_id]
                                
                                # 从其他跟踪集合中移除
                                if client_id in self.client_finished_prototypes:
                                    del self.client_finished_prototypes[client_id]
                                if client_id in self.client_expected_prototypes_count:
                                    del self.client_expected_prototypes_count[client_id]
                                if client_id in self.client_prototypes:
                                    del self.client_prototypes[client_id]
                                
                                # 重置超时计数
                                if client_id in self.client_timeout_counts:
                                    del self.client_timeout_counts[client_id]
                                
                                # 添加到已处理的离线客户端集合
                                self.offline_clients.add(client_id)
                            else:
                                print(f"[SERVER] 客户端 {client_id} 超时，标记为离线 ({self.client_timeout_counts[client_id]}/3)")
                                
                                # 更新原始记录
                                original_info = self.client_registry.get(client_id)
                                if original_info:
                                    original_info["status"] = "offline"
                                    
                                    # 从训练队列中移除
                                    if client_id in self.training_queue:
                                        try:
                                            self.training_queue.remove(client_id)
                                        except ValueError:
                                            pass  # 已经被移除
                                    
                                    # 标记为不在队列中
                                    original_info["in_queue"] = False
                    else:
                        # 如果客户端响应了，重置超时计数
                        if client_id in self.client_timeout_counts:
                            del self.client_timeout_counts[client_id]
                
                # 定期清理已离线的客户端集合，避免无限增长
                active_client_ids = set(self.client_registry.keys())
                self.offline_clients = {cid for cid in self.offline_clients if cid in active_client_ids}
                
                # 等待下一个检查周期
                await asyncio.sleep(self.status_check_interval)
                
            except Exception as e:
                print(f"[SERVER] 状态监控出错: {e}")
                await asyncio.sleep(self.status_check_interval)
    
    async def download_model(self):
        """下载嵌入模型"""
        model_name = "all-MiniLM-L6-v2"  # 硬编码模型名称
        model_dir = f"./models/{model_name}"
        
        if not os.path.exists(model_dir):
            raise HTTPException(status_code=404, detail="模型文件不存在")
        
        # 创建临时ZIP文件
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, f"{model_name}.zip")
        
        # 压缩模型目录
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_path = os.path.relpath(file_path, model_dir)
                    zipf.write(file_path, arc_path)
        
        return FileResponse(
            zip_path,
            media_type='application/zip',
            headers={"Content-Disposition": f"attachment; filename={model_name}.zip"}
        )