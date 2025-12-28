import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
from utils.numpy_utils import prepare_model_update_for_upload

# ==================== LSTM模型定义（与服务器一致） ====================
class SimpleLSTM(nn.Module):
    """简化的LSTM模型，与服务器保持一致"""
    def __init__(self, input_size=384, hidden_size=128, output_size=100):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
    
    def get_state_dict(self):
        """获取可序列化的状态字典"""
        # 先获取原始状态字典
        raw_state_dict = {}
        for k, v in self.state_dict().items():
            numpy_val = v.cpu().numpy()
            raw_state_dict[k] = numpy_val
        
        # 使用工具函数安全地转换为可JSON序列化的格式
        return prepare_model_update_for_upload(raw_state_dict)
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        new_dict = {}
        for k, v in state_dict.items():
            new_dict[k] = torch.tensor(v)
        super().load_state_dict(new_dict)