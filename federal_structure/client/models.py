import torch
import torch.nn as nn
from typing import Dict, Any

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
        import numpy as np
        
        def convert_to_python_type(obj):
            """将numpy类型转换为Python原生类型"""
            if isinstance(obj, np.ndarray):
                # 将numpy数组转换为Python列表，并确保所有元素都是原生类型
                return [convert_to_python_type(item) for item in obj.tolist()]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, list):
                return [convert_to_python_type(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                # 如果是其他类型，尝试转换为numpy然后tolist
                try:
                    return convert_to_python_type(np.asarray(obj).tolist())
                except:
                    return obj
        
        state_dict = {}
        for k, v in self.state_dict().items():
            numpy_val = v.cpu().numpy()
            state_dict[k] = convert_to_python_type(numpy_val)
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        new_dict = {}
        for k, v in state_dict.items():
            new_dict[k] = torch.tensor(v)
        super().load_state_dict(new_dict)