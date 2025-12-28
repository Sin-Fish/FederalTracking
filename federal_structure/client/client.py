import asyncio
import json
import hashlib
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
import socket

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from .data_processor import DataProcessor
from .communication import CommunicationModule
from .training import TrainingModule
from .models import SimpleLSTM


# ==================== 客户端核心类 ====================
class FederatedClient:
    """联邦学习客户端"""
    
    def __init__(self, 
                 server_url: str = "http://localhost:8000",
                 client_id: Optional[str] = None,
                 data_path: Optional[str] = None,
                 privacy_level: str = "medium"):
        
        self.server_url = server_url.rstrip("/")
        self.client_id = client_id or self._generate_client_id()
        self.data_path = Path(data_path) if data_path else None
        self.privacy_level = privacy_level
        
        # 初始化组件
        self.data_processor = DataProcessor(privacy_level)
        self.communication = CommunicationModule(self.server_url, self.client_id, self.data_processor)
        self.training = TrainingModule()
        
        # 嵌入模型
        self.embedding_model = None
        self.embedding_dim = 384
        
        # 训练配置
        self.sequence_length = 10
        self.batch_size = 32
        self.epochs_per_round = 3
        self.learning_rate = 0.001
        
        print(f"""
        ╔══════════════════════════════════════╗
        ║     联邦学习客户端启动              ║
        ╠══════════════════════════════════════╣
        ║ 客户端ID: {self.client_id:<25} ║
        ║ 服务器: {self.server_url:<25} ║
        ║ 隐私级别: {privacy_level:<25} ║
        ╚══════════════════════════════════════╝
        """)
    
    def _generate_client_id(self) -> str:
        """生成客户端ID"""
        hostname = socket.gethostname()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"client_{hostname}_{timestamp}"
    
    def load_data(self, data_path: Optional[str] = None):
        """加载数据"""
        self.data_processor.load_data(data_path)
    
    def process_raw_data(self) -> List[str]:
        """处理原始数据，生成行为字符串"""
        return self.data_processor.process_raw_data()
    
    def get_data_fingerprint(self) -> str:
        """生成数据指纹（用于验证数据一致性）"""
        return self.data_processor.get_data_fingerprint()
    
    def get_high_frequency_actions(self, top_k: int = 50) -> List[str]:
        """获取最高频的行为字符串"""
        return self.data_processor.get_high_frequency_actions(top_k)
    
    # ==================== 嵌入与原型映射 ====================
    def load_embedding_model(self):
        """加载嵌入模型"""
        if self.embedding_model is None:
            print("[CLIENT] 加载嵌入模型...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_behavior_strings(self, strings: List[str]) -> np.ndarray:
        """将行为字符串转换为嵌入向量"""
        self.load_embedding_model()
        
        print(f"[CLIENT] 嵌入 {len(strings)} 个行为字符串...")
        embeddings = self.embedding_model.encode(strings, show_progress_bar=False)
        return embeddings
    
    # ==================== 联邦通信接口 ====================
    def register_to_server(self) -> bool:
        """向服务器报到"""
        return self.communication.register_to_server()
    
    def submit_high_freq_data(self, top_k: int = 50) -> bool:
        """提交高频数据用于联邦聚类"""
        return self.communication.submit_high_freq_data(top_k)
    
    def fetch_prototypes(self) -> bool:
        """从服务器获取全局原型"""
        # 需要将嵌入模型设置到communication模块
        # 这里需要先加载数据和嵌入模型
        if not self.data_processor.behavior_strings:
            self.process_raw_data()
        
        if self.data_processor.behavior_embeddings is None:
            self.data_processor.behavior_embeddings = self.embed_behavior_strings(self.data_processor.behavior_strings)
        
        success = self.communication.fetch_prototypes()
        return success
    
    def send_status_update(self, status: str, progress: Optional[float] = None) -> bool:
        """向服务器发送状态更新"""
        return self.communication.send_status_update(status, progress)
    
    def submit_model_updates(self):
        """提交模型更新到服务器"""
        local_models = self.training.local_models
        self.communication.submit_model_updates(local_models, self.training.current_round, self.training.epochs_per_round)
    
    def start_heartbeat(self, interval: int = 30):
        """启动心跳线程"""
        self.communication.start_heartbeat(interval)
    
    # ==================== 主流程控制 ====================
    def run_full_pipeline(self, data_path: Optional[str] = None):
        """运行完整的客户端流程"""
        print("\n" + "="*60)
        print("开始运行联邦学习客户端完整流程")
        print("="*60)
        
        # 1. 加载数据
        self.load_data(data_path)
        
        # 2. 处理数据
        self.process_raw_data()
        
        # 3. 向服务器报到
        if not self.register_to_server():
            print("[CLIENT] 报到失败，退出流程")
            return
        
        # 4. 启动心跳
        self.start_heartbeat()
        
        # 5. 提交高频数据
        if not self.submit_high_freq_data():
            print("[CLIENT] 高频数据提交失败，继续流程...")
        
        # 6. 等待并获取原型（这里需要等待服务器完成聚类）
        print("[CLIENT] 等待服务器生成全局原型...")
        time.sleep(10)  # 等待10秒
        
        if not self.fetch_prototypes():
            print("[CLIENT] 获取原型失败，退出流程")
            return
        
        # 7. 等待训练开始信号（在实际系统中，这里应该监听服务器指令）
        print("[CLIENT] 等待训练开始...")
        # 这里可以添加轮询服务器状态的逻辑
        
        # 8. 开始本地训练
        local_models = self.training.train_local_models(
            self.data_processor.behavior_embeddings, 
            self.communication.prototype_mapping
        )
        
        # 9. 提交模型更新
        self.submit_model_updates()
        
        print("\n" + "="*60)
        print("客户端流程完成！")
        print("="*60)


# ==================== 命令行接口 ====================
def main():
    """主函数，处理命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="联邦学习客户端")
    parser.add_argument("--server", default="http://localhost:8000", help="服务器地址")
    parser.add_argument("--client-id", help="客户端ID（默认自动生成）")
    parser.add_argument("--data", help="数据文件路径（JSON或JSONL格式）")
    parser.add_argument("--privacy", choices=["low", "medium", "high"], default="medium", 
                       help="隐私保护级别")
    parser.add_argument("--simulate", action="store_true", help="使用模拟数据")
    parser.add_argument("--register-only", action="store_true", help="仅注册不训练")
    
    args = parser.parse_args()
    
    # 创建客户端实例
    client = FederatedClient(
        server_url=args.server,
        client_id=args.client_id,
        data_path=args.data,
        privacy_level=args.privacy
    )
    
    # 使用模拟数据（如果指定）
    if args.simulate:
        client.data_processor._generate_simulated_data()
    
    # 运行流程
    if args.register_only:
        # 仅注册
        client.load_data(args.data)
        client.process_raw_data()
        client.register_to_server()
        client.start_heartbeat()
        client.submit_high_freq_data()
        print("\n[CLIENT] 仅注册模式完成，客户端保持运行...")
        
        # 保持运行
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n[CLIENT] 客户端停止")
    else:
        # 完整流程
        client.run_full_pipeline(args.data)


if __name__ == "__main__":
    main()