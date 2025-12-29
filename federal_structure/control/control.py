import os
import subprocess
import sys
import threading
import time
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional


class FederatedControl:
    def __init__(self):
        self.server_process = None
        self.client_processes = []
        self.server_url = "http://localhost:8000"
        self.is_server_running = False

    def display_menu(self):
        """显示主菜单"""
        print("\n" + "="*50)
        print("        联邦学习系统控制台")
        print("="*50)
        print("1. 启动服务器")
        print("2. 启动客户端")
        print("3. 开始训练")
        print("4. 查看已连接客户端")
        print("5. 停止服务器")
        print("6. 停止所有客户端")
        print("7. 查看服务器状态")
        print("8. 退出")
        print("="*50)

    def start_server(self):
        """启动服务器"""
        if self.is_server_running:
            print("服务器已在运行中！")
            return

        print("启动服务器配置:")
        
        # 获取原型数量
        prototypes = input("请输入原型数量 (默认4): ").strip()
        if not prototypes:
            prototypes = "4"
        
        # 获取端口号
        port = input("请输入端口号 (默认8000): ").strip()
        if not port:
            port = "8000"
        
        # 获取主机地址
        host = input("请输入主机地址 (默认0.0.0.0): ").strip()
        if not host:
            host = "0.0.0.0"

        print(f"正在启动服务器 (原型数: {prototypes}, 端口: {port}, 主机: {host})...")
        
        try:
            # 启动服务器进程
            cmd = [
                sys.executable, 
                "server.py", 
                f"--prototypes", prototypes,
                f"--port", port,
                f"--host", host
            ]
            
            # 保存当前工作目录
            original_dir = os.getcwd()
            server_dir = os.path.join(os.path.dirname(__file__), "server")
            os.chdir(server_dir)
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 恢复工作目录
            os.chdir(original_dir)
            
            # 等待一段时间以确保服务器启动
            time.sleep(3)
            
            # 检查服务器是否成功启动
            if self.server_process.poll() is None:
                self.is_server_running = True
                print(f"服务器已启动，监听 {host}:{port}")
                print(f"服务器进程ID: {self.server_process.pid}")
                
                # 启动一个线程来监控服务器输出
                server_thread = threading.Thread(
                    target=self.monitor_server_output,
                    daemon=True
                )
                server_thread.start()
            else:
                print("服务器启动失败！")
                stderr_output = self.server_process.stderr.read()
                print(f"错误信息: {stderr_output}")
                
        except Exception as e:
            print(f"启动服务器时发生错误: {e}")

    def monitor_server_output(self):
        """监控服务器输出"""
        while self.is_server_running and self.server_process:
            if self.server_process.poll() is not None:
                break
            time.sleep(0.1)

    def start_client(self):
        """启动客户端"""
        print("启动客户端配置:")
        
        # 获取服务器地址
        server_url = input(f"请输入服务器地址 (默认{self.server_url}): ").strip()
        if not server_url:
            server_url = self.server_url
        
        # 获取数据路径
        data_path = input("请输入数据文件路径 (默认./data/activity_log.jsonl): ").strip()
        if not data_path:
            data_path = "./data/activity_log.jsonl"
        
        # 检查数据文件是否存在
        if not os.path.exists(data_path):
            print(f"警告: 数据文件 {data_path} 不存在")
            create_sample = input("是否创建示例数据文件? (y/n): ").strip().lower()
            if create_sample == 'y':
                self.create_sample_data(data_path)
        
        # 获取隐私级别
        privacy_level = input("请选择隐私保护级别 (low/medium/high，默认medium): ").strip()
        if not privacy_level or privacy_level not in ['low', 'medium', 'high']:
            privacy_level = 'medium'
        
        print(f"正在启动客户端 (服务器: {server_url}, 数据: {data_path})...")
        
        try:
            # 启动客户端进程
            cmd = [
                sys.executable,
                "client.py",
                f"--server", server_url,
                f"--data", data_path,
                f"--privacy", privacy_level
            ]
            
            # 保存当前工作目录
            original_dir = os.getcwd()
            client_dir = os.path.join(os.path.dirname(__file__), "client")
            os.chdir(client_dir)
            
            client_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 恢复工作目录
            os.chdir(original_dir)
            
            self.client_processes.append(client_process)
            print(f"客户端已启动，进程ID: {client_process.pid}")
            
            # 启动一个线程来监控客户端输出
            client_thread = threading.Thread(
                target=self.monitor_client_output,
                args=(client_process,),
                daemon=True
            )
            client_thread.start()
            
        except Exception as e:
            print(f"启动客户端时发生错误: {e}")

    def monitor_client_output(self, process):
        """监控客户端输出"""
        while process.poll() is None:
            time.sleep(0.1)

    def create_sample_data(self, path):
        """创建示例数据文件"""
        print(f"正在创建示例数据文件: {path}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        sample_data = [
            {
                "activity": {
                    "window_title": "Visual Studio Code - main.py",
                    "process_name": "Code.exe",
                    "executable_path": "/path/to/vscode",
                    "timestamp": datetime.now().isoformat()
                },
                "event_type": "keypress",
                "keyboard_event": {
                    "key": "a",
                    "scan_code": 30,
                    "is_keypad": False
                }
            },
            {
                "activity": {
                    "window_title": "File Explorer - Documents",
                    "process_name": "explorer.exe",
                    "executable_path": "/windows/explorer",
                    "timestamp": datetime.now().isoformat()
                },
                "event_type": "window_focus"
            }
        ]
        
        with open(path, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"示例数据已创建: {path}")

    def start_training(self):
        """开始训练"""
        if not self.is_server_running:
            print("错误: 服务器未运行，请先启动服务器！")
            return
        
        try:
            response = requests.get(f"{self.server_url}/api/training/start", timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"训练已启动: {result.get('message', '未知')}")
            else:
                print(f"启动训练失败: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"与服务器通信时发生错误: {e}")

    def view_connected_clients(self):
        """查看已连接客户端"""
        if not self.is_server_running:
            print("错误: 服务器未运行，请先启动服务器！")
            return
        
        try:
            response = requests.get(f"{self.server_url}/api/system/clients", timeout=10)
            if response.status_code == 200:
                clients = response.json()
                if clients:
                    print("\n已连接客户端列表:")
                    print("-" * 60)
                    for client_id, info in clients.items():
                        status = info.get('status', 'unknown')
                        last_seen = info.get('last_seen', 'unknown')
                        sample_count = info.get('sample_count', 'unknown')
                        print(f"ID: {client_id}")
                        print(f"  状态: {status}")
                        print(f"  最后活动: {last_seen}")
                        print(f"  样本数: {sample_count}")
                        print("-" * 60)
                else:
                    print("当前没有连接的客户端")
            else:
                print(f"获取客户端列表失败: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"与服务器通信时发生错误: {e}")

    def stop_server(self):
        """停止服务器"""
        if not self.is_server_running or not self.server_process:
            print("服务器未运行！")
            return
        
        try:
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
            print("服务器已停止")
        except subprocess.TimeoutExpired:
            self.server_process.kill()
            print("服务器强制停止")
        except Exception as e:
            print(f"停止服务器时发生错误: {e}")
        finally:
            self.is_server_running = False
            self.server_process = None

    def stop_all_clients(self):
        """停止所有客户端"""
        if not self.client_processes:
            print("没有运行中的客户端")
            return
        
        for i, process in enumerate(self.client_processes):
            try:
                if process.poll() is None:  # 进程仍在运行
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                        print(f"客户端 {i+1} 已停止")
                    except subprocess.TimeoutExpired:
                        process.kill()
                        print(f"客户端 {i+1} 强制停止")
            except Exception as e:
                print(f"停止客户端 {i+1} 时发生错误: {e}")
        
        self.client_processes.clear()

    def view_server_status(self):
        """查看服务器状态"""
        if not self.is_server_running:
            print("服务器未运行")
            return
        
        try:
            response = requests.get(f"{self.server_url}/", timeout=10)
            if response.status_code == 200:
                status = response.json()
                print("\n服务器状态:")
                print("-" * 30)
                print(f"服务: {status.get('service', 'Unknown')}")
                print(f"状态: {status.get('status', 'Unknown')}")
                print(f"客户端数量: {status.get('client_count', 0)}")
                print(f"训练队列大小: {status.get('training_queue_size', 0)}")
                print(f"原型就绪: {status.get('prototypes_ready', False)}")
                print("-" * 30)
            else:
                print(f"获取服务器状态失败: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"与服务器通信时发生错误: {e}")

    def run(self):
        """运行控制台"""
        print("欢迎使用联邦学习系统控制台")
        
        while True:
            self.display_menu()
            choice = input("\n请选择操作 (1-8): ").strip()
            
            if choice == '1':
                self.start_server()
            elif choice == '2':
                self.start_client()
            elif choice == '3':
                self.start_training()
            elif choice == '4':
                self.view_connected_clients()
            elif choice == '5':
                self.stop_server()
            elif choice == '6':
                self.stop_all_clients()
            elif choice == '7':
                self.view_server_status()
            elif choice == '8':
                print("正在退出...")
                self.stop_all_clients()
                if self.is_server_running:
                    self.stop_server()
                break
            else:
                print("无效选择，请重新输入")
            
            input("\n按回车键继续...")


if __name__ == "__main__":
    control = FederatedControl()
    control.run()