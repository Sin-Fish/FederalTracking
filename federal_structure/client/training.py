import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .models import SimpleLSTM


class TrainingModule:
    """训练模块，处理本地模型训练"""
    
    def __init__(self, sequence_length: int = 10, batch_size: int = 32, 
                 epochs_per_round: int = 3, learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs_per_round = epochs_per_round
        self.learning_rate = learning_rate
        
        # 训练状态
        self.is_training = False
        self.current_round = 0
        self.local_models: Dict[int, SimpleLSTM] = {}
        self.optimizers: Dict[int, optim.Optimizer] = {}
    
    def prepare_training_data(self, prototype_id: int, behavior_embeddings: np.ndarray, 
                           prototype_mapping: Dict[int, List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """为指定原型准备训练数据"""
        if prototype_id not in prototype_mapping or len(prototype_mapping[prototype_id]) < self.sequence_length + 1:
            return None, None
        
        # 获取该原型的所有数据索引
        indices = prototype_mapping[prototype_id]
        
        # 获取对应的嵌入向量
        embeddings = behavior_embeddings[indices]
        
        # 构建序列
        sequences = []
        targets = []
        
        for i in range(len(embeddings) - self.sequence_length):
            seq = embeddings[i:i + self.sequence_length]
            target = embeddings[i + self.sequence_length]
            
            sequences.append(seq)
            targets.append(target)
        
        if len(sequences) == 0:
            return None, None
        
        X = torch.tensor(np.array(sequences), dtype=torch.float32)
        y = torch.tensor(np.array(targets), dtype=torch.float32)
        
        return X, y
    
    def train_local_models(self, behavior_embeddings: np.ndarray, prototype_mapping: Dict[int, List[int]]):
        """训练所有本地模型"""
        print("[CLIENT] 开始本地训练...")
        
        self.is_training = True
        self.current_round += 1
        
        # 初始化本地模型
        self.local_models.clear()
        self.optimizers.clear()
        
        # 为每个有数据的原型创建模型
        for proto_id in prototype_mapping.keys():
            if len(prototype_mapping[proto_id]) > self.sequence_length * 2:  # 有足够数据
                model = SimpleLSTM()
                self.local_models[proto_id] = model
                self.optimizers[proto_id] = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        print(f"[CLIENT] 将为 {len(self.local_models)} 个原型训练模型")
        
        # 训练每个模型
        for epoch in range(self.epochs_per_round):
            total_loss = 0
            model_count = 0
            
            for proto_id, model in self.local_models.items():
                X, y = self.prepare_training_data(proto_id, behavior_embeddings, prototype_mapping)
                if X is None or y is None:
                    continue
                
                optimizer = self.optimizers[proto_id]
                criterion = nn.MSELoss()
                
                # 小批量训练
                for batch_start in range(0, len(X), self.batch_size):
                    batch_end = min(batch_start + self.batch_size, len(X))
                    X_batch = X[batch_start:batch_end]
                    y_batch = y[batch_start:batch_end]
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                model_count += 1
            
            # 更新进度
            progress = (epoch + 1) / self.epochs_per_round
            
            if model_count > 0:
                avg_loss = total_loss / model_count
                print(f"[CLIENT] 轮次 {self.current_round}, 周期 {epoch+1}/{self.epochs_per_round}, 平均损失: {avg_loss:.4f}")
        
        print("[CLIENT] 本地训练完成")
        self.is_training = False
        
        return self.local_models