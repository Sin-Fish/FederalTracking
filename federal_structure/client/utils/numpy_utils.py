import json
import numpy as np
import torch

class NumpyJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，安全地处理numpy和torch类型"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            # 递归转换数组中的所有元素
            return self._convert_ndarray(obj)
        elif isinstance(obj, torch.Tensor):
            # 将Tensor先转为numpy数组，再递归处理
            return self.default(obj.cpu().numpy())
        else:
            return super().default(obj)
    
    def _convert_ndarray(self, arr):
        """安全地将numpy数组转换为Python列表"""
        # 这是关键：.tolist() 对于非标量数组可能仍保留numpy类型
        # 我们递归地应用转换
        if arr.ndim == 0:
            # 标量
            return self.default(arr.item())
        elif arr.ndim == 1:
            # 一维数组
            return [self.default(x) for x in arr]
        else:
            # 多维数组：递归处理
            return [self._convert_ndarray(sub_arr) for sub_arr in arr]

def prepare_model_update_for_upload(model_state_dict):
    """
    将模型状态字典安全地转换为可JSON序列化的格式
    
    参数:
        model_state_dict: 从 model.state_dict() 获取的字典
    
    返回:
        完全由Python内置类型 (list, dict, int, float) 组成的字典
    """
    # 第一步：创建一个新的字典，避免修改原始状态
    safe_dict = {}
    
    for key, value in model_state_dict.items():
        # 使用自定义编码器进行深度转换
        # 先将对象序列化为JSON字符串，再解析回来，确保类型纯净
        json_str = json.dumps(value, cls=NumpyJSONEncoder)
        safe_dict[key] = json.loads(json_str)
    
    return safe_dict