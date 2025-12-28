import asyncio
import json
import hashlib
import time
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque

import numpy as np
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

# 机器学习相关导入
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter

from .ml_models import SimpleLSTM
from .models import ModelUpdate


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
        self.embedding_dim = 384

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
                "status": "idle"
            })
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
            "training_round": 0,
            "in_queue": False
        }
        
        self.client_registry[client_id] = client_info
        
        # 验证数据格式（这里可以添加更复杂的验证逻辑）
        if req.sample_count < 100:
            return JSONResponse(
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                content={
                    "status": "warning",
                    "message": "数据量较少，可能影响训练效果",
                    "client_id": client_id
                }
            )
        
        # 自动加入训练队列（根据你的策略可调整）
        if not client_info["in_queue"]:
            self.training_queue.append(client_id)
            client_info["in_queue"] = True
        
        print(f"[SERVER] 客户端 {client_id} 报到成功，样本数: {req.sample_count}")
        return {"status": "registered", "client_id": client_id, "queue_position": len(self.training_queue)}
    
    async def handle_status_update(self, status_update):
        """处理客户端状态更新"""
        client_id = status_update.client_id
        
        if client_id not in self.client_registry:
            raise HTTPException(status_code=404, detail="客户端未注册")
        
        client = self.client_registry[client_id]
        client["status"] = status_update.status
        client["last_heartbeat"] = time.time()
        
        if status_update.progress is not None:
            client["progress"] = status_update.progress
        
        # 如果训练完成或中断，从队列中移除
        if status_update.status in ["finished", "interrupted", "offline"]:
            client["in_queue"] = False
            if client_id in self.training_queue:
                self.training_queue.remove(client_id)
        
        print(f"[SERVER] 客户端 {client_id} 状态更新: {status_update.status}")
        return {"status": "updated"}
    
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
        
        # 检查是否所有队列中的客户端都已提交数据
        all_submitted = all(
            self.client_registry.get(cid, {}).get("data_submitted", False)
            for cid in self.training_queue
        )
        
        if all_submitted and len(self.training_queue) > 0:
            # 自动触发原型生成
            asyncio.create_task(self.generate_global_prototypes())
        
        return {"status": "collected", "action_count": len(submission.high_freq_actions)}
    
    async def generate_global_prototypes(self):
        """生成全局行为原型"""
        print("[SERVER] 开始生成全局行为原型...")
        
        # 收集所有高频行为数据
        all_samples = []
        for client_id in self.training_queue:
            if "high_freq_data" in self.client_registry[client_id]:
                all_samples.extend(self.client_registry[client_id]["high_freq_data"])
        
        if len(all_samples) < self.n_prototypes:
            print(f"[SERVER] 警告: 样本数({len(all_samples)})少于原型数({self.n_prototypes})")
            return {"status": "insufficient_data"}
        
        # 加载嵌入模型
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 转换为向量
        print(f"[SERVER] 正在嵌入 {len(all_samples)} 个行为样本...")
        embeddings = self.embedding_model.encode(all_samples, show_progress_bar=False)
        
        # K-Means聚类生成原型
        print(f"[SERVER] 正在聚类生成 {self.n_prototypes} 个全局原型...")
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
        if self.global_prototypes is None:
            raise HTTPException(status_code=400, detail="请先生成全局原型")
        
        if len(self.training_queue) == 0:
            raise HTTPException(status_code=400, detail="训练队列为空")
        
        print(f"[SERVER] 启动联邦训练，参与客户端: {len(self.training_queue)}")
        
        # 重置模型更新池
        self.model_updates.clear()
        
        # 准备训练指令
        training_info = {
            "prototypes": self.global_prototypes.tolist(),
            "prototype_labels": self.prototype_labels,
            "model_arch": "SimpleLSTM",
            "model_config": {
                "input_size": 384,
                "hidden_size": 128,
                "output_size": 100
            },
            "clients": list(self.training_queue)
        }
        
        # 更新客户端状态
        for client_id in self.training_queue:
            if client_id in self.client_registry:
                self.client_registry[client_id]["status"] = "training"
                self.client_registry[client_id]["progress"] = 0.0
                self.client_registry[client_id]["training_round"] += 1
        
        return {
            "status": "training_started",
            "client_count": len(self.training_queue),
            "training_info": training_info
        }
    
    async def handle_model_update(self, update: ModelUpdate):
        """处理模型更新"""
        client_id = update.client_id
        proto_id = update.prototype_id
        
        if client_id not in self.client_registry:
            raise HTTPException(status_code=404, detail="客户端未注册")
        
        if proto_id not in self.global_models:
            raise HTTPException(status_code=400, detail="无效的原型ID")
        
        # 存储更新
        update_record = {
            "client_id": client_id,
            "model_state": update.model_state_dict,
            "data_size": update.data_size,
            "timestamp": datetime.now().isoformat(),
            "metadata": update.metadata
        }
        
        self.model_updates[proto_id].append(update_record)
        
        # 更新客户端进度
        client = self.client_registry[client_id]
        if "updates_submitted" not in client:
            client["updates_submitted"] = 0
        client["updates_submitted"] += 1
        
        print(f"[SERVER] 收到客户端 {client_id} 对原型{proto_id}的更新，数据量: {update.data_size}")
        
        # 检查是否所有客户端都已完成更新
        all_clients_done = await self.check_training_completion()
        
        if all_clients_done:
            print("[SERVER] 所有客户端更新完成，准备聚合...")
            asyncio.create_task(self.perform_federated_averaging())
        
        return {"status": "update_received", "prototype_id": proto_id}
    
    async def check_training_completion(self):
        """检查训练是否完成"""
        # 简化检查：至少每个原型都有一些更新
        total_updates = sum(len(updates) for updates in self.model_updates.values())
        expected_min = len(self.training_queue) * self.n_prototypes // 2  # 至少一半
        
        return total_updates >= expected_min
    
    async def perform_federated_averaging(self):
        """执行联邦平均"""
        print("[SERVER] 开始执行联邦平均...")
        
        aggregated_models = {}
        
        for proto_id, updates in self.model_updates.items():
            if len(updates) == 0:
                continue
            
            # 计算总数据量
            total_data_size = sum(update["data_size"] for update in updates)
            
            # 初始化平均权重
            avg_state_dict = {}
            model_keys = None
            
            # 加权平均
            for update in updates:
                state_dict = update["model_state"]
                weight = update["data_size"] / total_data_size
                
                if model_keys is None:
                    model_keys = state_dict.keys()
                    # 初始化平均字典
                    for key in model_keys:
                        avg_state_dict[key] = np.array(state_dict[key]) * weight
                else:
                    for key in model_keys:
                        avg_state_dict[key] += np.array(state_dict[key]) * weight
            
            # 转换为列表格式
            final_state_dict = {k: v.tolist() for k, v in avg_state_dict.items()}
            
            # 更新全局模型
            if proto_id in self.global_models:
                self.global_models[proto_id].load_state_dict(final_state_dict)
                aggregated_models[proto_id] = final_state_dict
            
            print(f"[SERVER] 原型{proto_id}聚合完成，使用{len(updates)}个客户端更新")
        
        # 保存聚合结果
        self.save_aggregated_models(aggregated_models)
        
        return {
            "status": "aggregation_complete",
            "aggregated_models_count": len(aggregated_models),
            "timestamp": datetime.now().isoformat()
        }
    
    def save_aggregated_models(self, aggregated_models):
        """保存聚合后的模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aggregated_models_{timestamp}.pkl"
        
        save_data = {
            "models": aggregated_models,
            "prototypes": self.global_prototypes.tolist() if self.global_prototypes is not None else None,
            "prototype_labels": self.prototype_labels,
            "timestamp": timestamp
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"[SERVER] 聚合模型已保存到: {filename}")
    
    # ==================== 后台任务 ====================
    async def status_monitor(self):
        """后台状态监控任务"""
        while True:
            await asyncio.sleep(self.status_check_interval)
            
            current_time = time.time()
            offline_clients = []
            
            for client_id, client_info in list(self.client_registry.items()):
                # 检查心跳超时
                if current_time - client_info.get("last_heartbeat", 0) > self.status_check_interval * 3:
                    if client_info["status"] != "offline":
                        print(f"[MONITOR] 客户端 {client_id} 心跳超时，标记为离线")
                        client_info["status"] = "offline"
                        offline_clients.append(client_id)
            
            # 从队列中移除离线客户端
            for client_id in offline_clients:
                if client_id in self.training_queue:
                    self.training_queue.remove(client_id)