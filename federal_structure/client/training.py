import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from models import SimpleLSTM


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
        """为每个原型训练本地模型"""
        print(f"[CLIENT] 开始本地训练...")
        
        # 获取嵌入维度
        embedding_dim = behavior_embeddings.shape[1] if len(behavior_embeddings.shape) > 1 else behavior_embeddings.shape[0]
        print(f"[CLIENT] 嵌入维度: {embedding_dim}")
        
        # 获取原型数量
        num_prototypes = len(prototype_mapping)
        print(f"[CLIENT] 将为 {num_prototypes} 个原型训练模型")
        
        if num_prototypes == 0:
            print("[CLIENT] 没有原型需要训练")
            return {}
        
        local_models = {}
        
        for prototype_id in prototype_mapping:
            print(f"[CLIENT] 正在训练原型 {prototype_id} 的模型...")
            
            # 获取属于当前原型的行为索引
            prototype_indices = prototype_mapping[prototype_id]
            
            if len(prototype_indices) == 0:
                print(f"[CLIENT] 原型 {prototype_id} 没有分配的行为数据，跳过训练")
                continue
            
            # 获取当前原型的行为数据
            prototype_behaviors = behavior_embeddings[prototype_indices]
            
            if len(prototype_behaviors) == 0:
                print(f"[CLIENT] 原型 {prototype_id} 行为数据为空，跳过训练")
                continue
            
            # 使用LSTM模型
            model = SimpleLSTM(input_size=embedding_dim, hidden_size=100, output_size=embedding_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.MSELoss()
            
            # 准备训练数据
            # 将行为序列转换为输入-目标对 (x, y) 其中 y 是 x 的下一项
            if len(prototype_behaviors) < 2:
                print(f"[CLIENT] 原型 {prototype_id} 行为数据太少，无法训练")
                continue
            
            # 转换为tensor
            data_tensor = torch.FloatTensor(prototype_behaviors)
            
            # 创建序列数据
            seq_length = min(10, len(data_tensor) - 1)  # 确保序列长度不会超过数据长度
            sequences = []
            for i in range(len(data_tensor) - seq_length):
                seq = data_tensor[i:i+seq_length]
                target = data_tensor[i+seq_length]
                sequences.append((seq, target))
            
            if not sequences:
                print(f"[CLIENT] 原型 {prototype_id} 无法创建训练序列")
                continue
            
            # 训练模型
            model.train()
            for epoch in range(self.epochs_per_round):
                total_loss = 0
                for x_batch, y_batch in sequences:
                    # 增加批次维度
                    x_batch = x_batch.unsqueeze(0)  # (1, seq_len, embedding_dim)
                    y_batch = y_batch.unsqueeze(0)  # (1, embedding_dim)
                    
                    optimizer.zero_grad()
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if epoch % 5 == 0:  # 每5轮打印一次
                    print(f"[CLIENT] 原型 {prototype_id} - 轮次 {epoch}, 平均损失: {total_loss/len(sequences):.6f}")
            
            # 保存模型
            local_models[prototype_id] = model
            print(f"[CLIENT] 原型 {prototype_id} 模型训练完成")
        
        print("[CLIENT] 本地训练完成")
        return local_models
