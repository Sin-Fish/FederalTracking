from datetime import datetime, timedelta
import os
import json
import asyncio
import hashlib
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import requests
from starlette.responses import FileResponse

from models import ClientRegister, ClientStatus, DataSubmission, ModelUpdate
from core import FederatedServerCore


class FederatedServerAPI:
    def __init__(self, n_prototypes=5):
        self.app = FastAPI(title="联邦学习服务器", version="1.0")
        self.core = FederatedServerCore(n_prototypes=n_prototypes)
        
        # 添加启动和关闭事件处理器
        @self.app.on_event("startup")
        async def startup_event():
            # 启动状态监控任务
            asyncio.create_task(self.core.status_monitor())
        
        # 初始化API路由
        self.setup_routes()
    
    def setup_routes(self):
        """设置所有API路由"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "联邦学习服务器",
                "status": "运行中",
                "client_count": len(self.core.client_registry),
                "training_queue_size": len(self.core.training_queue),
                "prototypes_ready": self.core.global_prototypes is not None
            }
        
        @self.app.post("/api/client/register")
        async def client_register(req: ClientRegister):
            """客户端报到接口"""
            return await self.core.handle_client_register(req)
        
        @self.app.post("/api/client/status")
        async def update_status(status: ClientStatus):
            """客户端状态上报接口"""
            return await self.core.handle_status_update(status)
        
        @self.app.post("/api/data/collect")
        async def collect_data(submission: DataSubmission):
            """数据征收接口"""
            return await self.core.handle_data_collection(submission)
        
        @self.app.get("/api/training/start")
        async def start_training():
            """启动训练（管理员接口）"""
            return await self.core.initiate_training()
        
        @self.app.get("/api/training/wait")
        async def wait_for_training_start(client_id: str = Query(..., description="客户端ID")):
            """客户端等待训练开始指令（长轮询）"""
            # 检查客户端是否已注册
            if client_id not in self.core.client_registry:
                raise HTTPException(status_code=404, detail="客户端未注册")
            
            # 检查客户端是否在训练队列中
            if client_id not in self.core.training_queue:
                raise HTTPException(status_code=400, detail="客户端不在训练队列中")
            
            # 获取客户端的训练开始事件
            event = self.core.training_start_events.get(client_id)
            if not event:
                raise HTTPException(status_code=500, detail="客户端事件未初始化")
            
            # 等待事件被设置（最多等待30秒）
            try:
                await asyncio.wait_for(event.wait(), timeout=30.0)
                
                # 返回训练信息
                return {
                    "status": "training_start",
                    "message": "训练已开始，请开始本地训练",
                    "training_info": {
                        "prototypes": self.core.global_prototypes.tolist() if self.core.global_prototypes is not None else None,
                        "prototype_labels": self.core.prototype_labels,
                        "model_hash": self.core.embedding_model_hash
                    }
                }
            except asyncio.TimeoutError:
                # 超时返回
                return {
                    "status": "timeout",
                    "message": "等待训练指令超时"
                }
        
        @self.app.get("/api/client/readiness-check")
        async def readiness_check(client_id: str = Query(..., description="客户端ID")):
            """服务器检查客户端就位状态的端点"""
            # 检查客户端是否已注册
            if client_id not in self.core.client_registry:
                raise HTTPException(status_code=404, detail="客户端未注册")
            
            # 检查客户端是否在训练队列中
            if client_id not in self.core.training_queue:
                raise HTTPException(status_code=400, detail="客户端不在训练队列中")
            
            # 获取客户端的就位检查事件
            event = self.core.readiness_check_events.get(client_id)
            if not event:
                raise HTTPException(status_code=500, detail="客户端就位检查事件未初始化")
            
            # 设置事件，表示服务器发起就位检查
            event.set()
            
            return {
                "status": "ready_check_initiated",
                "message": f"已向客户端 {client_id} 发送就位检查信号",
                "model_hash": self.core.embedding_model_hash,
                "model_available": self.core.embedding_model is not None
            }
        
        @self.app.get("/api/client/wait-readiness-check")
        async def wait_for_readiness_check(client_id: str = Query(..., description="客户端ID")):
            """客户端等待服务器发起就位检查（长轮询）"""
            # 检查客户端是否已注册
            if client_id not in self.core.client_registry:
                raise HTTPException(status_code=404, detail="客户端未注册")
            
            # 检查客户端是否在训练队列中
            if client_id not in self.core.training_queue:
                raise HTTPException(status_code=400, detail="客户端不在训练队列中")
            
            # 获取客户端的就位检查事件
            event = self.core.readiness_check_events.get(client_id)
            if not event:
                raise HTTPException(status_code=500, detail="客户端就位检查事件未初始化")
            
            # 等待事件被设置（最多等待60秒，给服务器足够时间发起训练）
            try:
                await asyncio.wait_for(event.wait(), timeout=360.0)
                
                # 返回就位检查信息
                return {
                    "status": "ready_check",
                    "message": f"客户端 {client_id} 进行就位检查",
                    "model_hash": self.core.embedding_model_hash,
                    "model_available": self.core.embedding_model is not None
                }
            except asyncio.TimeoutError:
                # 超时返回
                return {
                    "status": "timeout",
                    "message": "等待就位检查指令超时"
                }
        
        @self.app.post("/api/client/confirm-readiness")
        async def confirm_readiness(client_id: str = Query(..., description="客户端ID")):
            """客户端确认就位状态"""
            # 检查客户端是否已注册
            if client_id not in self.core.client_registry:
                raise HTTPException(status_code=404, detail="客户端未注册")
            
            # 检查客户端是否在训练队列中
            if client_id not in self.core.training_queue:
                raise HTTPException(status_code=400, detail="客户端不在训练队列中")
            
            # 标记客户端已确认就位
            self.core.client_registry[client_id]["readiness_confirmed"] = True
            self.core.client_registry[client_id]["readiness_confirmed_at"] = datetime.now().isoformat()
            
            return {
                "status": "confirmed",
                "message": f"客户端 {client_id} 就位确认成功"
            }
        
        @self.app.post("/api/model/update")
        async def submit_update(update: ModelUpdate):
            """接收模型更新"""
            return await self.core.handle_model_update(update)
        
        @self.app.post("/api/client/prototypes")
        async def receive_client_prototypes(client_id: str = Query(..., description="客户端ID"), prototypes: List[List[float]] = None):
            """接收客户端发回的聚类中心"""
            if prototypes is None:
                raise HTTPException(status_code=400, detail="聚类中心数据不能为空")
            return await self.core.receive_client_prototypes(client_id, prototypes)
        
        @self.app.get("/api/system/prototypes")
        async def get_prototypes():
            """获取当前全局原型"""
            if self.core.global_prototypes is None:
                raise HTTPException(status_code=404, detail="原型尚未生成")
            return {
                "prototypes": self.core.global_prototypes.tolist(),
                "labels": self.core.prototype_labels,
                "count": len(self.core.prototype_labels)
            }
        
        @self.app.get("/api/system/clients")
        async def get_clients():
            """获取所有客户端状态"""
            return self.core.client_registry
        
        @self.app.post("/api/federated/aggregate")
        async def aggregate_models():
            """手动触发模型聚合"""
            return await self.core.perform_federated_averaging()
        
        @self.app.get("/api/system/model_info")
        async def get_model_info():
            """获取嵌入模型信息"""
            return await self.core.get_model_info()
        
        @self.app.get("/api/model/download")
        async def download_model():
            """下载嵌入模型文件"""
            return await self.core.download_model()