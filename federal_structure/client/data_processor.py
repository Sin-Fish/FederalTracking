import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from collections import Counter
import numpy as np

from privacy_filter import PrivacyFilter


class DataProcessor:
    """数据处理模块"""
    
    def __init__(self, privacy_level: str = "medium"):
        self.privacy_filter = PrivacyFilter(privacy_level)
        self.raw_data: List[Dict] = []
        self.behavior_strings: List[str] = []
        self.behavior_embeddings: np.ndarray = None
        
    def load_data(self, data_path: Optional[str] = None):
        """加载数据"""
        print(f"[CLIENT] 调试: 传入的数据路径为：{data_path}")
        
        if not data_path or not Path(data_path).exists():
            print(f"[CLIENT] 数据文件不存在，生成模拟数据...")
            self._generate_simulated_data()
            return
        
        print(f"[CLIENT] 从 {data_path} 加载数据...")
        
        try:
            path = Path(data_path)
            if path.suffix == '.jsonl':
                # JSON Lines格式
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.raw_data.append(json.loads(line.strip()))
            elif path.suffix == '.json':
                # JSON数组格式
                with open(path, 'r', encoding='utf-8') as f:
                    self.raw_data = json.load(f)
            else:
                print(f"[CLIENT] 不支持的文件格式: {path.suffix}")
                self._generate_simulated_data()
                return
                
            print(f"[CLIENT] 成功加载 {len(self.raw_data)} 条记录")
            
        except Exception as e:
            print(f"[CLIENT] 加载数据失败: {e}")
            self._generate_simulated_data()

    def _generate_simulated_data(self):
        """生成模拟数据用于测试"""
        print("[CLIENT] 生成模拟用户行为数据...")
        
        # 模拟不同的用户行为模式
        behavior_patterns = [
            # 工作模式
            {
                "process": "code.exe",
                "window": "project_main.py - VS Code",
                "actions": ["key_s", "key_ctrl", "key_tab", "click_left", "key_enter"]
            },
            # 学习模式
            {
                "process": "chrome.exe",
                "window": "Python Tutorial - Stack Overflow - Chrome",
                "actions": ["scroll_down", "click_link", "key_ctrl_f", "key_space"]
            },
            # 娱乐模式
            {
                "process": "steam.exe",
                "window": "Game Name - Steam",
                "actions": ["key_w", "key_a", "key_s", "key_d", "click_right", "key_shift"]
            }
        ]
        
        # 生成1000条模拟记录
        for i in range(1000):
            pattern = np.random.choice(behavior_patterns, p=[0.4, 0.4, 0.2])
            action = np.random.choice(pattern["actions"])
            
            record = {
                "timestamp": datetime.now().isoformat(),
                "environment": {
                    "process_name": pattern["process"],
                    "window_title": pattern["window"]
                },
                "event_type": "key_press" if action.startswith("key") else "mouse_click",
                "keyboard_event": {"key": action.replace("key_", "")} if action.startswith("key") else None,
                "mouse_event": {"button": "left", "x": np.random.randint(100, 1000), "y": np.random.randint(100, 800)} 
                if action.startswith("click") else None
            }
            
            self.raw_data.append(record)
        
        print(f"[CLIENT] 生成 {len(self.raw_data)} 条模拟记录")
    
    def process_raw_data(self) -> List[str]:
        """处理原始数据，生成行为字符串"""
        print("[CLIENT] 处理原始数据，生成行为字符串...")
        
        self.behavior_strings = []
        
        for record in self.raw_data:
            try:
                # 检查记录是否为None
                if record is None:
                    print("[CLIENT] 跳过None记录")
                    continue
                
                # 提取基本信息，添加对environment为None的检查
                environment = record.get("environment", {})
                if environment is None:
                    environment = {}
                process = environment.get("process_name", "unknown")
                window = environment.get("window_title", "unknown")
                
                # 应用隐私过滤
                safe_window = self.privacy_filter.filter_window_title(process, window)
                
                # 提取操作信息
                event_type = record.get("event_type", "unknown")
                
                if event_type == "key_press":
                    key_event = record.get("keyboard_event", {})
                    if key_event is None:  # 添加对key_event为None的检查
                        key_event = {}
                    key = key_event.get("key", "unknown")
                    # 简化特殊键名
                    if key.startswith("Key."):
                        key = key.replace("Key.", "")
                    action = f"key_{key.lower()}"
                    
                elif event_type == "mouse_click":
                    mouse_event = record.get("mouse_event", {})
                    if mouse_event is None:  # 添加对mouse_event为None的检查
                        mouse_event = {}
                    button = mouse_event.get("button", "unknown")
                    # 对于鼠标点击，我们可以选择是否包含坐标
                    # 为了隐私，这里只记录点击动作，不记录具体坐标
                    action = f"click_{button.lower()}"
                    
                elif event_type == "mouse_scroll":
                    scroll_event = record.get("scroll_event", {})
                    if scroll_event is None:  # 添加对scroll_event为None的检查
                        scroll_event = {}
                    direction = scroll_event.get("direction", "unknown")
                    action = f"scroll_{direction.lower()}"
                    
                else:
                    action = "unknown"
                
                # 构建行为字符串
                behavior_str = f"{process}_{safe_window}_{action}"
                # 进一步清理字符串
                behavior_str = behavior_str.replace(" ", "_").replace(".exe", "")[:100]
                
                self.behavior_strings.append(behavior_str)
                
            except Exception as e:
                print(f"[CLIENT] 处理记录时出错: {e}")
                continue
        
        print(f"[CLIENT] 生成 {len(self.behavior_strings)} 个行为字符串")
        return self.behavior_strings
    
    def get_data_fingerprint(self) -> str:
        """生成数据指纹（用于验证数据一致性）"""
        if not self.behavior_strings:
            self.process_raw_data()
        
        # 使用最高频的100个行为字符串生成指纹
        top_100 = Counter(self.behavior_strings).most_common(100)
        fingerprint_str = "|".join([f"{action}:{count}" for action, count in top_100])
        
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:32]
    
    def get_high_frequency_actions(self, top_k: int = 50) -> List[str]:
        """获取最高频的行为字符串"""
        if not self.behavior_strings:
            self.process_raw_data()
        
        counter = Counter(self.behavior_strings)
        return [action for action, _ in counter.most_common(top_k)]