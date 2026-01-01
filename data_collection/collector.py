#!/usr/bin/env python3
"""
电脑操作数据采集器
记录：1. 环境（当前窗口/应用） 2. 事件（鼠标点击、键盘按键）
"""

import time
import json
import hashlib
import platform
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Any
import threading
import sys

# ---------- 平台相关的导入 ----------
PLATFORM = platform.system().lower()

if PLATFORM == "windows":
    try:
        # 鼠标键盘监听 (跨平台)
        from pynput import mouse, keyboard
        from pynput.keyboard import Key, KeyCode
        # 窗口信息获取 (Windows)
        import win32gui
        import win32process
        import psutil
    except ImportError as e:
        print(f"导入警告: {e}. 将使用基础监听模式和模拟窗口信息。")
        try:
            from pynput import mouse, keyboard
            from pynput.keyboard import Key, KeyCode
        except ImportError:
            print("错误: 必须安装 'pynput' 库。请运行: pip install pynput")
            sys.exit(1)
elif PLATFORM == "linux":
    try:
        # 在Linux上，我们可能需要evdev或其他库
        from pynput import mouse, keyboard
        from pynput.keyboard import Key, KeyCode
        import psutil
    except ImportError:
        print("在Linux上可能需要安装 'python3-dev libevdev-dev' 等系统依赖。")
        # 可以尝试使用evdev
        try:
            import evdev
            from pynput import mouse, keyboard
            from pynput.keyboard import Key, KeyCode
            import psutil
        except ImportError:
            print("警告: 缺少必要的库，将使用模拟数据模式。")
            mouse = None
            keyboard = None
            Key = None
            KeyCode = None
            psutil = None
else:  # macOS或其他
    try:
        from pynput import mouse, keyboard
        from pynput.keyboard import Key, KeyCode
        import psutil
    except ImportError:
        print("警告: 缺少必要的库，将使用模拟数据模式。")
        mouse = None
        keyboard = None
        Key = None
        KeyCode = None
        psutil = None

# ---------- 数据结构定义 (使用dataclass) ----------
class EventType(Enum):
    MOUSE_CLICK = "mouse_click"
    MOUSE_SCROLL = "mouse_scroll"
    KEY_PRESS = "key_press"
    APP_SWITCH = "app_switch"  # 用于记录主动的窗口切换

@dataclass
class EnvironmentData:
    """环境层数据：描述操作发生时的上下文"""
    window_title: str
    process_name: str
    # 以下字段在非Windows平台可能为模拟值
    executable_path: Optional[str] = None
    timestamp: float = 0.0  # 环境捕获时间戳

    def to_dict(self):
        d = asdict(self)
        d['timestamp'] = datetime.fromtimestamp(self.timestamp).isoformat() if self.timestamp else None
        return d

@dataclass
class MouseEventData:
    """鼠标点击事件数据"""
    button: str = None  # 'left', 'right', 'middle'，滚动事件为None
    x: int = None
    y: int = None
    dx: int = None  # 水平滚动距离，仅滚动事件使用
    dy: int = None  # 垂直滚动距离，仅滚动事件使用
    pressed: bool = None  # True按下, False释放 (我们通常只记录按下)，滚动事件为None

@dataclass
class KeyboardEventData:
    """键盘按键事件数据"""
    key: str  # 键名，如 'a', 'Enter', 'shift'
    # 我们通常只记录按键按下，不记录释放

@dataclass
class OperationRecord:
    """单条完整操作记录 (环境 + 事件)"""
    # 元信息
    record_id: str  # 唯一ID，基于时间哈希生成
    user_id: str    # 匿名用户ID (在初始化时设置)
    timestamp: str  # ISO格式时间
    # 核心数据
    environment: EnvironmentData
    event_type: EventType
    mouse_data: Optional[MouseEventData] = None
    keyboard_data: Optional[KeyboardEventData] = None

    def to_dict(self) -> dict:
        """转换为字典，便于JSON序列化"""
        result = {
            "record_id": self.record_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "environment": self.environment.to_dict(),
            "event_type": self.event_type.value
        }
        if self.mouse_data:
            result["mouse_event"] = asdict(self.mouse_data)
        if self.keyboard_data:
            result["keyboard_event"] = asdict(self.keyboard_data)
        return result

# ---------- 核心采集器类 ----------
class ActivityCollector:
    def __init__(self, user_id: Optional[str] = None, log_file_prefix: str = "activity_log", data_dir: str = "./data"):
        """
        初始化采集器
        :param user_id: 用户标识，若为空则自动生成匿名哈希
        :param log_file_prefix: 日志文件前缀
        :param data_dir: 数据保存目录，默认为 ./data
        """
        # 用户标识 (匿名化处理)
        if user_id:
            # 对输入的user_id进行哈希，确保不可逆
            self.user_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        else:
            # 生成基于时间的匿名ID
            self.user_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

        # 确保数据目录存在
        import os
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 日志文件设置 (按日期自动分割)
        self.log_file = os.path.join(self.data_dir, f"{log_file_prefix}_{datetime.now().strftime('%Y%m%d')}.jsonl")
        self._init_log_file()

        # 状态变量
        self.current_environment = None
        self.last_window_title = None
        self.is_collecting = False
        self.listener_lock = threading.Lock()

        # 滚动事件聚合相关
        self.scroll_buffer = None
        self.scroll_buffer_env = None
        self.scroll_aggregation_timer = None
        self.SCROLL_AGGREGATION_TIMEOUT = 3  # 3秒内连续滚动视为一次滚动

        # Ctrl键状态
        self.ctrl_pressed = False

        # 监听器对象
        self.mouse_listener = None
        self.keyboard_listener = None

        # 统计信息
        self.stats = {"mouse_clicks": 0, "mouse_scroll": 0, "key_presses": 0, "records_saved": 0}

        print(f"[*] 数据采集器初始化完成")
        print(f"    User ID (匿名): {self.user_id}")
        print(f"    日志文件: {self.log_file}")
        print(f"    平台: {PLATFORM}")

    def _init_log_file(self):
        """初始化日志文件，写入元数据头"""
        meta = {
            "collector_start_time": datetime.now().isoformat(),
            "user_id_anonymous": self.user_id,
            "platform": PLATFORM,
            "data_schema": "1.0"
        }
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"type": "metadata", "content": meta}, ensure_ascii=False) + '\n')
        print(f"[+] 日志文件已创建: {self.log_file}")

    # ---------- 环境信息获取 ----------
    def _get_active_window_info_windows(self) -> Optional[EnvironmentData]:
        """Windows平台获取活动窗口信息"""
        try:
            hwnd = win32gui.GetForegroundWindow()
            window_title = win32gui.GetWindowText(hwnd).strip()

            # 过滤无效窗口
            if not window_title or window_title == "开始" or window_title == "Desktop":
                return None

            # 获取进程信息
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(pid)
            process_name = process.name()
            exe_path = process.exe()

            return EnvironmentData(
                window_title=window_title[:200],  # 限制长度
                process_name=process_name,
                executable_path=exe_path,
                timestamp=time.time()
            )
        except Exception as e:
            # 权限不足或窗口切换过快可能导致异常
            return None

    def _get_active_window_info_generic(self) -> Optional[EnvironmentData]:
        """非Windows平台或后备方案：返回模拟/基础信息"""
        # 这里可以扩展macOS或Linux的获取方式
        # 目前返回一个模拟环境
        return EnvironmentData(
            window_title="Simulated_Window",
            process_name="simulated_process",
            executable_path=None,
            timestamp=time.time()
        )

    def get_current_environment(self) -> Optional[EnvironmentData]:
        """获取当前环境信息 (平台适配)"""
        if PLATFORM == "windows":
            env = self._get_active_window_info_windows()
        else:
            env = self._get_active_window_info_generic()

        if env and env.window_title != self.last_window_title:
            # 环境发生变化时，刷新滚动缓冲区
            self._flush_scroll_buffer_if_exists()
            
            self.last_window_title = env.window_title
            # 可选：在这里记录一次APP_SWITCH事件
            # self._record_app_switch(env)
        return env

    # ---------- 事件监听回调 ----------
    def _on_mouse_click(self, x, y, button, pressed):
        """鼠标点击回调"""
        # 刷新滚动缓冲区（如果有的话）
        self._flush_scroll_buffer_if_exists()
        
        # 只记录按下事件，忽略拖动和释放
        if not pressed or not self.is_collecting:
            return

        with self.listener_lock:
            env = self.get_current_environment()
            if not env:
                return

            # 创建事件数据
            mouse_data = MouseEventData(
                button=str(button).split('.')[-1].lower(),
                x=int(x),
                y=int(y),
                pressed=pressed
            )

            # 创建完整记录
            record = OperationRecord(
                record_id=self._generate_record_id(),
                user_id=self.user_id,
                timestamp=datetime.now().isoformat(),
                environment=env,
                event_type=EventType.MOUSE_CLICK,
                mouse_data=mouse_data
            )

            # 保存记录
            self._save_record(record)
            self.stats["mouse_clicks"] += 1

            # 实时输出到控制台 (简洁版)
            if self.stats["mouse_clicks"] % 10 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 采集统计: 鼠标{self.stats['mouse_clicks']} | 键盘{self.stats['key_presses']} | 记录{self.stats['records_saved']}")

    def _on_mouse_scroll(self, x, y, dx, dy):
        """鼠标滚动回调 - 实现滚动事件聚合"""
        if not self.is_collecting:
            return

        with self.listener_lock:
            env = self.get_current_environment()
            if not env:
                return

            # 如果当前没有滚动缓冲，创建一个新的
            if self.scroll_buffer is None:
                self.scroll_buffer = {
                    'x': int(x),
                    'y': int(y),
                    'dx': int(dx),
                    'dy': int(dy),
                    'start_time': time.time(),
                    'end_time': time.time(),
                    'scroll_count': 1
                }
                self.scroll_buffer_env = env
            else:
                # 如果在缓冲中，更新累计值
                self.scroll_buffer['dx'] += int(dx)
                self.scroll_buffer['dy'] += int(dy)
                self.scroll_buffer['end_time'] = time.time()
                self.scroll_buffer['scroll_count'] += 1

            # 重置或启动聚合计时器
            if self.scroll_aggregation_timer:
                self.scroll_aggregation_timer.cancel()
            self.scroll_aggregation_timer = threading.Timer(
                self.SCROLL_AGGREGATION_TIMEOUT, 
                self._flush_scroll_buffer
            )
            self.scroll_aggregation_timer.start()

    def _flush_scroll_buffer_if_exists(self):
        """如果滚动缓冲区存在内容，则刷新它"""
        if self.scroll_buffer is not None:
            if self.scroll_aggregation_timer:
                self.scroll_aggregation_timer.cancel()
                self.scroll_aggregation_timer = None
            self._flush_scroll_buffer()

    def _flush_scroll_buffer(self):
        """将滚动缓冲区的内容写入日志文件"""
        with self.listener_lock:
            if self.scroll_buffer is not None:
                # 创建滚动事件数据
                mouse_data = MouseEventData(
                    x=self.scroll_buffer['x'],
                    y=self.scroll_buffer['y'],
                    dx=self.scroll_buffer['dx'],
                    dy=self.scroll_buffer['dy']
                )

                # 创建完整记录
                record = OperationRecord(
                    record_id=self._generate_record_id(),
                    user_id=self.user_id,
                    timestamp=datetime.fromtimestamp(self.scroll_buffer['end_time']).isoformat(),
                    environment=self.scroll_buffer_env,
                    event_type=EventType.MOUSE_SCROLL,
                    mouse_data=mouse_data
                )

                # 保存记录
                self._save_record(record)
                self.stats["mouse_scroll"] += 1

                # 清空缓冲区
                self.scroll_buffer = None
                self.scroll_buffer_env = None

    def _on_key_press(self, key):
        """键盘按键回调"""
        # 刷新滚动缓冲区（如果有的话）
        self._flush_scroll_buffer_if_exists()
        
        if not self.is_collecting:
            return

        with self.listener_lock:
            env = self.get_current_environment()
            if not env:
                return

            # 处理按键名
            try:
                # 普通字符键
                key_str = key.char if hasattr(key, 'char') and key.char else str(key)
                # 清理格式，如 'Key.enter' -> 'enter'
                if key_str.startswith('Key.'):
                    key_str = key_str.split('.')[-1]
            except AttributeError:
                key_str = str(key)

            # 过滤掉某些系统键（可选）
            if key_str in ['esc', 'cmd', 'cmd_r', 'win', 'alt']:
                return

            keyboard_data = KeyboardEventData(key=key_str)

            record = OperationRecord(
                record_id=self._generate_record_id(),
                user_id=self.user_id,
                timestamp=datetime.now().isoformat(),
                environment=env,
                event_type=EventType.KEY_PRESS,
                keyboard_data=keyboard_data
            )

            self._save_record(record)
            self.stats["key_presses"] += 1

    def _generate_record_id(self) -> str:
        """生成唯一记录ID"""
        return hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:12]

    def _save_record(self, record: OperationRecord):
        """保存单条记录到文件"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')
            self.stats["records_saved"] += 1
        except Exception as e:
            print(f"[!] 保存记录失败: {e}")

    # ---------- 控制接口 ----------
    def start_collection(self):
        """开始采集"""
        if self.is_collecting:
            print("[!] 采集已在运行中")
            return

        print("\n" + "="*50)
        print("开始采集电脑操作数据...")
        print("提示: 采集期间请正常使用电脑")
        print("      按 'Esc+Ctrl' 键可停止采集")
        print("="*50 + "\n")

        self.is_collecting = True

        # 重置滚动缓冲区
        self.scroll_buffer = None
        self.scroll_buffer_env = None

        # 启动鼠标监听器 (抑制事件传递，避免干扰正常使用)
        self.mouse_listener = mouse.Listener(
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll,
            suppress=False  # 设为True会阻止鼠标点击事件，请谨慎
        )

        # 启动键盘监听器
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=None,  # 我们不记录释放事件
            suppress=False    # 设为True会阻止键盘输入，请务必设为False！
        )

        self.mouse_listener.start()
        self.keyboard_listener.start()

        print("[+] 监听器已启动。正在采集...\n")

        # 设置一个全局键盘钩子来监听停止键（这里用Ctrl+Esc）
        self.ctrl_pressed = False  # 标记Ctrl键是否按下

        def on_stop_key(key):
            if key == Key.ctrl_l or key == Key.ctrl_r:
                self.ctrl_pressed = True
            elif key == Key.esc and self.ctrl_pressed:
                print("\n[!] 检测到 Ctrl+Esc 组合键，正在停止采集...")
                self.stop_collection()
                return False  # 停止监听
            return True

        def on_release(key):
            if key == Key.ctrl_l or key == Key.ctrl_r:
                self.ctrl_pressed = False
            return True

        # 启动单独的停止键监听器
        with keyboard.Listener(on_press=on_stop_key, on_release=on_release) as stop_listener:
            stop_listener.join()

    def stop_collection(self):
        """停止采集"""
        if not self.is_collecting:
            return

        self.is_collecting = False

        # 确保滚动缓冲区内容被写入
        if self.scroll_aggregation_timer:
            self.scroll_aggregation_timer.cancel()
        self._flush_scroll_buffer()

        # 停止监听器
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()

        # 等待线程结束
        time.sleep(0.5)

        print("\n" + "="*50)
        print("数据采集已停止")
        print("="*50)
        self._print_statistics()
        self._print_sample_record()

    def _print_statistics(self):
        """打印采集统计"""
        print(f"\n📊 采集统计:")
        print(f"   鼠标点击事件: {self.stats['mouse_clicks']}")
        print(f"   鼠标滚动事件: {self.stats['mouse_scroll']}")
        print(f"   键盘按键事件: {self.stats['key_presses']}")
        print(f"   有效记录总数: {self.stats['records_saved']}")
        print(f"   日志文件位置: {self.log_file}")

    def _print_sample_record(self):
        """打印一条样本记录"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in reversed(lines[-10:]):  # 查看最后10条
                    if line.strip() and '"type"' not in line:  # 跳过元数据行
                        record = json.loads(line.strip())
                        print(f"\n📄 最后一条记录示例:")
                        print(json.dumps(record, indent=2, ensure_ascii=False))
                        break
        except Exception as e:
            print(f"\n[!] 无法读取样本记录: {e}")

# ---------- 主程序入口 ----------
if __name__ == "__main__":
    print("="*60)
    print("电脑操作数据采集器 v1.0")
    print("="*60)

    # 简单配置
    user_input = input("请输入一个用户名（用于匿名标识，回车将自动生成）: ").strip()
    log_prefix = input("请输入日志文件前缀（默认为 'activity_log'）: ").strip()
    if not log_prefix:
        log_prefix = "activity_log"
    
    data_path = input("请输入数据保存路径（默认为 './data'）: ").strip()
    if not data_path:
        data_path = "./data"

    # 创建并启动采集器
    collector = ActivityCollector(
        user_id=user_input if user_input else None,
        log_file_prefix=log_prefix,
        data_dir=data_path
    )

    try:
        collector.start_collection()
    except KeyboardInterrupt:
        print("\n[!] 用户中断 (Ctrl+C)")
        collector.stop_collection()
    except Exception as e:
        print(f"\n[!] 发生未预期错误: {e}")
        import traceback
        traceback.print_exc()
        collector.stop_collection()
