import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
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
        self.state_embeddings: np.ndarray = None  # 状态路径嵌入
        self.action_embeddings: np.ndarray = None  # 动作路径嵌入
        self.state_prototypes: np.ndarray = None  # 状态聚类中心
        self.action_prototypes: np.ndarray = None  # 动作聚类中心
        self.browser_processes = ['chrome.exe', 'firefox.exe', 'msedge.exe', 'opera.exe', 'safari.exe']  # 浏览器进程列表

    def get_unique_process_count(self) -> int:
        """获取数据中唯一进程的数量"""
        if not self.raw_data:
            self.process_raw_data()
            
        processes = set()
        for record in self.raw_data:
            try:
                environment = record.get("environment", {})
                if environment is None:
                    environment = {}
                process = environment.get("process_name", "unknown").lower()
                # 只保留进程名，去掉扩展名
                process_name = process.replace('.exe', '').replace('.EXE', '')
                processes.add(process_name)
            except Exception as e:
                print(f"[CLIENT] 提取进程名时出错: {e}")
                continue

        # 返回唯一进程数量
        return len(processes)

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

    def extract_state_action_paths(self) -> Tuple[List[str], List[str]]:
        """提取状态路径和动作路径数据"""
        if not self.raw_data:
            self.process_raw_data()
            
        state_paths = []  # 状态路径：进程-一级窗口
        action_paths = []  # 动作路径：根据进程类型区分处理

        for record in self.raw_data:
            try:
                # 提取基本信息
                environment = record.get("environment", {})
                if environment is None:
                    environment = {}
                process = environment.get("process_name", "unknown").lower()
                window = environment.get("window_title", "unknown")
                
                # 应用隐私过滤
                safe_window = self.privacy_filter.filter_window_title(process, window)
                
                # 提取操作信息
                event_type = record.get("event_type", "unknown")
                
                if event_type == "key_press":
                    key_event = record.get("keyboard_event", {})
                    if key_event is None:
                        key_event = {}
                    key = key_event.get("key", "unknown")
                    if key.startswith("Key."):
                        key = key.replace("Key.", "")
                    action = f"key_{key.lower()}"
                elif event_type == "mouse_click":
                    mouse_event = record.get("mouse_event", {})
                    if mouse_event is None:
                        mouse_event = {}
                    button = mouse_event.get("button", "unknown")
                    action = f"click_{button.lower()}"
                elif event_type == "mouse_scroll":
                    scroll_event = record.get("scroll_event", {})
                    if scroll_event is None:
                        scroll_event = {}
                    direction = scroll_event.get("direction", "unknown")
                    action = f"scroll_{direction.lower()}"
                else:
                    action = "unknown"
                
                # 状态路径：进程-一级窗口
                state_path = f"{process}_{safe_window.split(' - ')[0]}"  # 取窗口标题的第一部分作为一级窗口
                state_paths.append(state_path)
                
                # 动作路径：根据进程类型区分处理
                if process in self.browser_processes:
                    # 浏览器进程：进程-一级窗口-操作
                    action_path = f"{process}_{safe_window.split(' - ')[0]}_{action}"
                else:
                    # 非浏览器进程：进程-操作
                    action_path = f"{process}_{action}"
                
                action_paths.append(action_path)
                
            except Exception as e:
                print(f"[CLIENT] 提取状态/动作路径时出错: {e}")
                continue

        print(f"[CLIENT] 提取了 {len(state_paths)} 个状态路径，{len(action_paths)} 个动作路径")
        return state_paths, action_paths

    def perform_local_clustering(self, state_paths: List[str], action_paths: List[str], 
                                n_state_prototypes: int = 5, n_action_prototypes: int = 10, 
                                embedding_model=None) -> Tuple[np.ndarray, np.ndarray]:
        """执行本地双路聚类"""
        print(f"[CLIENT] 开始本地双路聚类...")
        print(f"[CLIENT] 状态路径数量: {len(state_paths)}, 动作路径数量: {len(action_paths)}")
        print(f"[CLIENT] 生成 {n_state_prototypes} 个状态原型和 {n_action_prototypes} 个动作原型")
        
        if embedding_model is None:
            print("[CLIENT] 错误：未提供嵌入模型，无法执行聚类")
            print("[CLIENT] 使用随机嵌入向量作为备用方案...")
            
            # 为状态路径生成随机嵌入
            print(f"[CLIENT] 生成 {len(state_paths)} 个状态路径的随机嵌入...")
            state_embeddings = np.random.rand(len(state_paths), 384).astype(np.float32)
            
            # 为动作路径生成随机嵌入
            print(f"[CLIENT] 生成 {len(action_paths)} 个动作路径的随机嵌入...")
            action_embeddings = np.random.rand(len(action_paths), 384).astype(np.float32)
        else:
            print(f"[CLIENT] 嵌入 {len(state_paths)} 个状态路径...")
            state_embeddings = embedding_model.encode(state_paths, show_progress_bar=False)
            print(f"[CLIENT] 嵌入 {len(action_paths)} 个动作路径...")
            action_embeddings = embedding_model.encode(action_paths, show_progress_bar=False)

        # 保存嵌入向量
        self.state_embeddings = state_embeddings
        self.action_embeddings = action_embeddings

        # 对状态路径进行聚类
        if len(state_paths) >= n_state_prototypes:
            from sklearn.cluster import KMeans
            print(f"[CLIENT] 对状态路径进行K-Means聚类，k={n_state_prototypes}...")
            state_kmeans = KMeans(n_clusters=n_state_prototypes, random_state=42, n_init=10)
            state_kmeans.fit(state_embeddings)
            self.state_prototypes = state_kmeans.cluster_centers_
        else:
            print(f"[CLIENT] 状态路径数量({len(state_paths)})少于原型数({n_state_prototypes})，使用随机原型")
            self.state_prototypes = np.random.rand(n_state_prototypes, 384).astype(np.float32)

        # 对动作路径进行聚类
        if len(action_paths) >= n_action_prototypes:
            from sklearn.cluster import KMeans
            print(f"[CLIENT] 对动作路径进行K-Means聚类，k={n_action_prototypes}...")
            action_kmeans = KMeans(n_clusters=n_action_prototypes, random_state=42, n_init=10)
            action_kmeans.fit(action_embeddings)
            self.action_prototypes = action_kmeans.cluster_centers_
        else:
            print(f"[CLIENT] 动作路径数量({len(action_paths)})少于原型数({n_action_prototypes})，使用随机原型")
            self.action_prototypes = np.random.rand(n_action_prototypes, 384).astype(np.float32)

        print(f"[CLIENT] 本地双路聚类完成！状态原型形状: {self.state_prototypes.shape}, 动作原型形状: {self.action_prototypes.shape}")
        
        return self.state_prototypes, self.action_prototypes

    def get_local_prototypes(self) -> Tuple[List[List[float]], List[List[float]]]:
        """获取本地聚类中心，用于发送到服务器"""
        if self.state_prototypes is None or self.action_prototypes is None:
            print("[CLIENT] 错误：尚未生成本地聚类中心，请先执行本地聚类")
            return [], []
            
        # 转换为列表格式，便于JSON序列化
        state_prototypes_list = [proto.tolist() for proto in self.state_prototypes]
        action_prototypes_list = [proto.tolist() for proto in self.action_prototypes]
        
        print(f"[CLIENT] 获取本地聚类中心：{len(state_prototypes_list)} 个状态原型，{len(action_prototypes_list)} 个动作原型")
        return state_prototypes_list, action_prototypes_list

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