import torch
import torch.nn as nn
from typing import Dict, Any

# ==================== LSTM模型定义（简化示例） ====================

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=384, hidden_size=128, output_size=100):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
    
    def get_state_dict(self):
        return {k: v.cpu().numpy().tolist() for k, v in self.state_dict().items()}
    
    def load_state_dict(self, state_dict):
        new_dict = {}
        for k, v in state_dict.items():
            new_dict[k] = torch.tensor(v)
        super().load_state_dict(new_dict)