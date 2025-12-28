# federated_client.py
import asyncio
import json
import hashlib
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity

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
        return {k: v.cpu().numpy().tolist() for k, v in self.state_dict().items()}
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        new_dict = {}
        for k, v in state_dict.items():
            new_dict[k] = torch.tensor(v)
        super().load_state_dict(new_dict)

# ==================== 隐私过滤器 ====================
class PrivacyFilter:
    """隐私过滤器，处理窗口标题中的敏感信息"""
    
    def __init__(self, privacy_level="medium"):
        """
        privacy_level: low/medium/high
        - low: 保留最多信息
        - medium: 平衡隐私与效用（推荐）
        - high: 最大程度隐私保护
        """
        self.privacy_level = privacy_level
        self.browser_keywords = ["chrome", "firefox", "edge", "safari", "浏览器"]
        
    def filter_window_title(self, process_name: str, window_title: str) -> str:
        """过滤窗口标题中的隐私信息"""
        process_lower = process_name.lower()
        
        # 浏览器特殊处理（按你的"后门"设计）
        if any(browser in process_lower for browser in self.browser_keywords):
            return self._filter_browser_title(window_title)
        
        # 聊天软件处理
        elif any(app in process_lower for app in ["qq", "wechat", "微信", "telegram"]):
            return self._filter_chat_title(window_title)
        
        # 文档处理软件
        elif any(app in process_lower for app in ["word", "wps", "libreoffice", "记事本"]):
            return self._filter_document_title(window_title)
        
        # 代码编辑器
        elif any(app in process_lower for app in ["vscode", "pycharm", "idea", "sublime"]):
            return self._filter_code_editor_title(window_title)
        
        # 默认处理
        else:
            return self._filter_general_title(window_title)
    
    def _filter_browser_title(self, title: str) -> str:
        """浏览器标题过滤 - 保留站点信息"""
        if self.privacy_level == "high":
            return "网页浏览"
        elif self.privacy_level == "medium":
            # 尝试提取站点名
            for sep in [" - ", " – ", " | "]:
                if sep in title:
                    parts = title.split(sep)
                    if len(parts) >= 2:
                        site_part = parts[-2]  # 通常是站点名
                        # 简单清理
                        site_part = site_part.replace("https://", "").replace("http://", "")
                        site_part = site_part.split("/")[0].split("?")[0]
                        return f"浏览_{site_part[:20]}"
            return "网页浏览"
        else:  # low
            return title[:50]  # 仅截断，保留大部分信息
    
    def _filter_chat_title(self, title: str) -> str:
        """聊天软件标题过滤"""
        if self.privacy_level == "high":
            return "聊天窗口"
        elif self.privacy_level == "medium":
            # 移除具体联系人，保留类型
            if " - " in title:
                return "聊天窗口"
            return title[:30]
        else:  # low
            return title[:40]
    
    def _filter_document_title(self, title: str) -> str:
        """文档标题过滤"""
        if self.privacy_level == "high":
            return "文档编辑"
        elif self.privacy_level == "medium":
            # 提取文件类型
            if "." in title:
                ext = title.split(".")[-1].lower()
                if ext in ["doc", "docx", "pdf", "txt"]:
                    return f"{ext}文档"
            return "文档编辑"
        else:  # low
            return title[:40]
    
    def _filter_code_editor_title(self, title: str) -> str:
        """代码编辑器标题过滤"""
        if self.privacy_level == "high":
            return "代码编辑"
        elif self.privacy_level == "medium":
            # 提取项目或文件类型
            if " - " in title:
                main_part = title.split(" - ")[0]
                if "." in main_part:
                    ext = main_part.split(".")[-1].lower()
                    if ext in ["py", "js", "java", "cpp"]:
                        return f"{ext}代码"
                return "代码项目"
            return "代码编辑"
        else:  # low
            return title[:50]
    
    def _filter_general_title(self, title: str) -> str:
        """通用标题过滤"""
        if self.privacy_level == "high":
            return "应用程序"
        elif self.privacy_level == "medium":
            # 取第一个分隔符前的部分
            for sep in [" - ", " – ", " | ", " : "]:
                if sep in title:
                    return title.split(sep)[0][:30]
            return title[:30]
        else:  # low
            return title[:50]

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
        self.privacy_filter = PrivacyFilter(privacy_level)
        self.embedding_model = None
        self.embedding_dim = 384
        
        # 本地数据
        self.raw_data: List[Dict] = []
        self.behavior_strings: List[str] = []
        self.behavior_embeddings: np.ndarray = None
        
        # 联邦学习状态
        self.global_prototypes: Optional[np.ndarray] = None
        self.prototype_labels: List[str] = []
        self.prototype_mapping: Dict[int, List[int]] = defaultdict(list)  # 原型->数据索引映射
        self.local_models: Dict[int, SimpleLSTM] = {}  # 原型->本地模型
        self.optimizers: Dict[int, optim.Optimizer] = {}
        
        # 训练配置
        self.sequence_length = 10
        self.batch_size = 32
        self.epochs_per_round = 3
        self.learning_rate = 0.001
        
        # 通信状态
        self.is_registered = False
        self.is_training = False
        self.current_round = 0
        self.last_heartbeat = time.time()
        
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
        import socket
        hostname = socket.gethostname()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"client_{hostname}_{timestamp}"
    
    # ==================== 数据加载与处理 ====================
    def load_data(self, data_path: Optional[str] = None):
        """加载数据"""
        if data_path:
            self.data_path = Path(data_path)
        
        if not self.data_path or not self.data_path.exists():
            print(f"[CLIENT] 数据文件不存在，生成模拟数据...")
            self._generate_simulated_data()
            return
        
        print(f"[CLIENT] 从 {self.data_path} 加载数据...")
        
        try:
            if self.data_path.suffix == '.jsonl':
                # JSON Lines格式
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.raw_data.append(json.loads(line.strip()))
            elif self.data_path.suffix == '.json':
                # JSON数组格式
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    self.raw_data = json.load(f)
            else:
                print(f"[CLIENT] 不支持的文件格式: {self.data_path.suffix}")
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
                # 提取基本信息
                process = record.get("environment", {}).get("process_name", "unknown")
                window = record.get("environment", {}).get("window_title", "unknown")
                
                # 应用隐私过滤
                safe_window = self.privacy_filter.filter_window_title(process, window)
                
                # 提取操作信息
                event_type = record.get("event_type", "unknown")
                
                if event_type == "key_press":
                    key_event = record.get("keyboard_event", {})
                    key = key_event.get("key", "unknown")
                    # 简化特殊键名
                    if key.startswith("Key."):
                        key = key.replace("Key.", "")
                    action = f"key_{key.lower()}"
                    
                elif event_type == "mouse_click":
                    mouse_event = record.get("mouse_event", {})
                    button = mouse_event.get("button", "unknown")
                    # 对于鼠标点击，我们可以选择是否包含坐标
                    # 为了隐私，这里只记录点击动作，不记录具体坐标
                    action = f"click_{button.lower()}"
                    
                elif event_type == "mouse_scroll":
                    scroll_event = record.get("scroll_event", {})
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
    
    def map_to_prototypes(self, prototypes: np.ndarray):
        """将本地数据映射到全局原型"""
        print("[CLIENT] 将本地数据映射到全局原型...")
        
        # 计算本地数据的嵌入
        if self.behavior_embeddings is None:
            self.behavior_embeddings = self.embed_behavior_strings(self.behavior_strings)
        
        # 为每个本地向量找到最近的原型
        from sklearn.metrics.pairwise import pairwise_distances_argmin
        prototype_indices = pairwise_distances_argmin(self.behavior_embeddings, prototypes)
        
        # 构建映射
        self.prototype_mapping.clear()
        for idx, proto_idx in enumerate(prototype_indices):
            self.prototype_mapping[proto_idx].append(idx)
        
        # 打印统计信息
        print("[CLIENT] 数据分布统计:")
        for proto_idx in sorted(self.prototype_mapping.keys()):
            count = len(self.prototype_mapping[proto_idx])
            percentage = count / len(self.behavior_strings) * 100
            label = self.prototype_labels[proto_idx] if proto_idx < len(self.prototype_labels) else f"原型{proto_idx}"
            print(f"  {label}: {count} 条数据 ({percentage:.1f}%)")
    
    # ==================== 联邦通信接口 ====================
    def register_to_server(self) -> bool:
        """向服务器报到"""
        print("[CLIENT] 向服务器报到...")
        
        # 确保数据已处理
        if not self.behavior_strings:
            self.process_raw_data()
        
        fingerprint = self.get_data_fingerprint()
        
        payload = {
            "client_id": self.client_id,
            "data_fingerprint": fingerprint,
            "embedding_version": "all-MiniLM-L6-v2",
            "sample_count": len(self.behavior_strings)
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/api/client/register",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"[CLIENT] 报到成功: {data}")
                self.is_registered = True
                return True
            else:
                print(f"[CLIENT] 报到失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"[CLIENT] 报到时发生错误: {e}")
            return False
    
    def submit_high_freq_data(self, top_k: int = 50) -> bool:
        """提交高频数据用于联邦聚类"""
        print("[CLIENT] 提交高频数据...")
        
        high_freq = self.get_high_frequency_actions(top_k)
        
        payload = {
            "client_id": self.client_id,
            "high_freq_actions": high_freq,
            "sample_count": len(self.behavior_strings)
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/api/data/collect",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"[CLIENT] 高频数据提交成功")
                return True
            else:
                print(f"[CLIENT] 高频数据提交失败: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[CLIENT] 提交高频数据时发生错误: {e}")
            return False
    
    def fetch_prototypes(self) -> bool:
        """从服务器获取全局原型"""
        print("[CLIENT] 获取全局原型...")
        
        try:
            response = requests.get(
                f"{self.server_url}/api/system/prototypes",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.global_prototypes = np.array(data["prototypes"])
                self.prototype_labels = data["labels"]
                
                print(f"[CLIENT] 成功获取 {len(self.prototype_labels)} 个全局原型")
                
                # 立即进行数据映射
                self.map_to_prototypes(self.global_prototypes)
                return True
            else:
                print(f"[CLIENT] 获取原型失败: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[CLIENT] 获取原型时发生错误: {e}")
            return False
    
    def send_status_update(self, status: str, progress: Optional[float] = None) -> bool:
        """向服务器发送状态更新"""
        payload = {
            "client_id": self.client_id,
            "status": status,
            "progress": progress
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/api/client/status",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                self.last_heartbeat = time.time()
                return True
            return False
            
        except Exception as e:
            print(f"[CLIENT] 状态更新失败: {e}")
            return False
    
    # ==================== 本地模型训练 ====================
    def prepare_training_data(self, prototype_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """为指定原型准备训练数据"""
        if prototype_id not in self.prototype_mapping or len(self.prototype_mapping[prototype_id]) < self.sequence_length + 1:
            return None, None
        
        # 获取该原型的所有数据索引
        indices = self.prototype_mapping[prototype_id]
        
        # 获取对应的嵌入向量
        embeddings = self.behavior_embeddings[indices]
        
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
    
    def train_model(self, model: SimpleLSTM, data: List[str], proto_id: int):
        """训练单个模型"""
        X, y = self.prepare_training_data(proto_id)
        if X is None or y is None:
            return
        
        optimizer = self.optimizers[proto_id]
        criterion = nn.MSELoss()
        
        # 小批量训练
        for epoch in range(self.epochs_per_round):
            total_loss = 0
            
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
            
            avg_loss = total_loss / len(X)
            print(f"[CLIENT] 原型 {proto_id} 轮次 {epoch+1}/{self.epochs_per_round}, 平均损失: {avg_loss:.4f}")
    
    def train_local_models(self):
        """训练所有本地模型"""
        print("[CLIENT] 开始本地训练...")
        self.is_training = True
        
        for proto_id, model in self.local_models.items():
            print(f"[CLIENT] 正在训练原型 {proto_id} 的模型...")
            
            # 获取该原型的数据
            proto_data_indices = self.prototype_mapping[proto_id]
            proto_data = [self.processed_data[i] for i in proto_data_indices]
            
            # 训练模型
            self.train_model(model, proto_data, proto_id)
            print(f"[CLIENT] 原型 {proto_id} 模型训练完成")
    
    def submit_model_updates(self):
        """提交模型更新到服务器"""
        print("[CLIENT] 提交模型更新...")
        
        for proto_id, model in self.local_models.items():
            data_size = len(self.prototype_mapping[proto_id])
            
            # 获取模型状态
            state_dict = model.get_state_dict()
            
            payload = {
                "client_id": self.client_id,
                "prototype_id": proto_id,
                "model_state_dict": state_dict,
                "data_size": data_size,
                "metadata": {
                    "round": self.current_round,
                    "epochs": self.epochs_per_round,
                    "data_points": data_size,
                    "submitted_at": datetime.now().isoformat()
                }
            }
            
            try:
                response = requests.post(
                    f"{self.server_url}/api/model/update",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    print(f"[CLIENT] 原型{proto_id}的模型更新提交成功")
                else:
                    print(f"[CLIENT] 原型{proto_id}的模型更新提交失败: {response.status_code}")
                    
            except Exception as e:
                print(f"[CLIENT] 提交模型更新时发生错误: {e}")
        
        print("[CLIENT] 模型更新提交完成")
    
    # ==================== 心跳与维护 ====================
    def start_heartbeat(self, interval: int = 30):
        """启动心跳线程"""
        import threading
        
        def heartbeat_loop():
            while True:
                time.sleep(interval)
                current_status = "training" if self.is_training else "idle"
                self.send_status_update(current_status)
        
        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        print(f"[CLIENT] 心跳线程已启动，间隔: {interval}秒")
    
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
        self.train_local_models()
        
        # 9. 提交模型更新
        self.submit_model_updates()
        
        # 10. 发送完成状态（之前在train_local_models内部发送）
        self.is_training = False
        self.send_status_update("finished", 1.0)
        
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
        client._generate_simulated_data()
    
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