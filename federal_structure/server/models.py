from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

# ==================== 数据模型定义 ====================

class ClientRegister(BaseModel):
    """客户端报到请求模型"""
    client_id: str
    data_fingerprint: str  # 数据哈希指纹
    embedding_version: str = "all-MiniLM-L6-v2"
    sample_count: int = Field(ge=1, le=1000000)  # 数据样本量


class ClientStatus(BaseModel):
    """客户端状态上报模型"""
    client_id: str
    status: str  # training, idle, finished, interrupted, offline
    progress: Optional[float] = Field(None, ge=0, le=1)  # 训练进度 0-1


class DataSubmission(BaseModel):
    """数据征收提交模型"""
    client_id: str
    high_freq_actions: List[str]  # 高频行为字符串列表
    sample_count: int  # 样本总数


class ModelUpdate(BaseModel):
    """模型更新提交模型"""
    client_id: str
    prototype_id: int  # 原型ID
    model_state_dict: Dict[str, Any]  # 模型参数
    data_size: int  # 用于该原型训练的数据量
    metadata: Dict[str, Any] = {}  # 训练元数据