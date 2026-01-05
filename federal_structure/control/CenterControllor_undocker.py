import os
import subprocess
import sys
import threading
import time
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional


class NewFederatedControl:
    def __init__(self):
        self.is_server_running = False
        # 计算项目根目录路径 - 从当前文件向上三级到项目根目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.server_process = None
        self.client_processes = []

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
        print("6. 停止单个客户端")
        print("7. 停止所有客户端")
        print("8. 退出")
        print("="*50)

    def start_server(self):
        """启动服务器进程"""
        if self.is_server_running:
            print("服务器已在运行中！")
            return

        print("启动服务器配置:")
        
        # 获取用户输入
        prototypes = input("请输入原型数量 (默认4): ").strip() or "4"
        port = input("请输入端口号 (默认8000): ").strip() or "8000"
        host = input("请输入主机地址 (默认0.0.0.0): ").strip() or "0.0.0.0"

        print(f"正在启动服务器 (原型数: {prototypes}, 端口: {port}, 主机: {host})...")

        try:
            # 构建启动命令
            server_script = os.path.join(self.project_root, "federal_structure", "server", "server.py")
            cmd = [
                sys.executable,
                server_script,
                "--prototypes", prototypes,
                "--port", port,
                "--host", host
            ]
            
            # 使用CREATE_NEW_CONSOLE标志在新窗口启动
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            # 启动服务器进程
            self.server_process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE  # 在新窗口创建
            )
            self.is_server_running = True
            print("服务器已启动，查看新窗口获取详细信息")
            time.sleep(5)
            return

        except Exception as e:
            print(f"启动服务器时发生错误: {e}")

    def start_client(self):
        """启动客户端进程"""
        print("启动客户端配置:")

        # 获取用户输入
        server_url = input("请输入服务器地址 (默认 http://localhost:8000): ").strip() or "http://localhost:8000"
        privacy_level = input("请输入隐私级别 (low/medium/high，默认medium): ").strip() or "medium"
        data_path = input("请输入数据文件路径 (默认 ./data/activity_log.jsonl): ").strip() or os.path.join(self.project_root, "data", "activity_log.jsonl")
        
        client_id = f"federated-client-{int(time.time())}"
        
        print(f"正在启动客户端 {client_id} (服务器: {server_url}, 隐私级别: {privacy_level}, 数据路径: {data_path})...")

        try:
            # 构建启动命令
            client_script = os.path.join(self.project_root, "federal_structure", "client", "client.py")
            cmd = [
                sys.executable,
                client_script,
                "--server", server_url,
                "--privacy", privacy_level,
                "--data", data_path,
                "--client-id", client_id
            ]
            
            # 使用CREATE_NEW_CONSOLE标志在新窗口启动
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            # 启动客户端进程
            client_process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE  # 在新窗口创建
            )
            self.client_processes.append(client_process)
            print(f"客户端 {client_id} 已启动，查看新窗口获取详细信息")
            return

        except Exception as e:
            print(f"启动客户端时发生错误: {e}")

    def stop_server(self):
        """停止服务器进程"""
        try:
            if self.server_process and self.server_process.poll() is None:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                print("服务器已停止")
                self.is_server_running = False
            else:
                print("服务器未运行")

        except Exception as e:
            print(f"停止服务器时发生错误: {e}")

    def stop_all_clients(self):
        """停止所有客户端进程"""
        try:
            for process in self.client_processes:
                if process and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                    print("客户端已停止")
            self.client_processes.clear()

        except Exception as e:
            print(f"停止客户端时发生错误: {e}")

    def check_server_status(self):
        """检查服务器状态"""
        try:
            esult = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "ps", "server"],
                capture_output=True, text=True
            )# 检查服务器容器状态
            r
            
            if "Up" in result.stdout:
                print("服务器正在运行")
                # 尝试连接服务器API
                try:
                    response = requests.get(f"http://localhost:8000/status", timeout=10)
                    if response.status_code == 200:
                        status_data = response.json()
                        print(f"服务器详细状态: {status_data}")
                    else:
                        print("无法获取服务器详细状态")
                except requests.exceptions.RequestException:
                    print("服务器容器运行中，但API连接失败")
            else:
                print("服务器未运行")
                
        except Exception as e:
            print(f"检查服务器状态时发生错误: {e}")

    def check_connected_clients(self):
        """查看已连接客户端"""
        try:
            # 使用默认服务器地址，或从环境变量获取
            server_address = os.getenv("SERVER_URL", "http://localhost:8000")
            response = requests.get(f"{server_address}/api/system/clients", timeout=60)
            if response.status_code == 200:
                clients = response.json()
                print(f"已连接客户端数量: {len(clients)}")
                for client in clients:
                    print(f"- {client}")
            else:
                print("无法获取客户端连接信息")
        except requests.exceptions.RequestException as e:
            print(f"获取客户端信息失败: {e}")

    def start_training(self):
        """开始训练"""
        try:
            # 使用默认服务器地址，或从环境变量获取
            server_address = os.getenv("SERVER_URL", "http://localhost:8000")
            
            # 添加进度提示，因为训练启动可能需要较长时间
            print("正在启动训练，请稍候...")
            # 修正API端点和HTTP方法
            response = requests.get(f"{server_address}/api/training/start", timeout=120)  # 使用GET方法和正确的API端点
            if response.status_code == 200:
                result = response.json()
                
                # 更友好的输出格式
                if result.get("status") == "training_started":
                    print(f"✓ 训练已成功启动!")
                    print(f"  参与客户端数量: {result.get('client_count', 0)}")
                    training_info = result.get("training_info", {})
                    if training_info:
                        print(f"  模型架构: {training_info.get('model_arch', 'N/A')}")
                        print(f"  原型数量: {len(training_info.get('prototype_labels', []))}")
                        print(f"  客户端列表: {training_info.get('clients', [])}")
                elif result.get("status") == "error":
                    print(f"✗ 训练启动失败: {result.get('message', '未知错误')}")
                else:
                    print(f"训练启动结果: {result}")
            else:
                print(f"启动训练失败: {response.status_code} - {response.text}")
        except requests.exceptions.Timeout:
            print("启动训练请求超时，请检查服务器状态")
        except requests.exceptions.RequestException as e:
            print(f"启动训练请求失败: {e}")

    def view_client_output(self):
        """查看客户端输出"""
        try:
            # 获取所有客户端容器（包括运行中和已停止的）
            result = subprocess.run(
                ["docker", "ps", "-a", "-f", "name=federated-client-", "--format", "{{.Names}}: {{.Status}}"],
                capture_output=True, text=True
            )
            
            if not result.stdout.strip():
                print("没有找到客户端容器")
                return
            
            client_containers = result.stdout.strip().split('\n')
            if len(client_containers) == 1 and client_containers[0] == '':
                print("没有找到客户端容器")
                return
            
            print("当前客户端容器 (名称: 状态):")
            for container in client_containers:
                print(container)
            
            container_name = input("请输入要查看日志的客户端容器名称: ").strip()
            
            # 验证输入的容器名称是否存在于客户端容器列表中
            container_names = [name.split(':')[0] for name in client_containers]
            if container_name not in container_names:
                print(f"错误: 找不到名为 {container_name} 的客户端容器")
                return

            # 查看指定客户端日志
            log_result = subprocess.run(
                ["docker", "logs", container_name],
                capture_output=True, text=True
            )
            
            if log_result.returncode == 0:
                print(f"客户端 {container_name} 日志:")
                print(log_result.stdout[-2000:])  # 显示最后2000个字符
            else:
                print(f"获取客户端日志失败: {log_result.stderr}")
                
        except Exception as e:
            print(f"获取客户端日志时发生错误: {e}")

    def run(self):
        """运行主循环"""
        while True:
            self.display_menu()
            choice = input("请选择操作 (1-10): ").strip()

            if choice == "1":
                self.start_server()
            elif choice == "2":
                self.start_client()
            elif choice == "3":
                self.start_training()
            elif choice == "4":
                self.check_connected_clients()
            elif choice == "5":
                self.stop_server()
            elif choice == "6":
                self.stop_client()
            elif choice == "7":
                self.stop_all_clients()
            elif choice == "8":
                print("正在退出...")
                break
            else:
                print("无效选择，请重试！")

            input("\n按回车键继续...")


if __name__ == "__main__":
    control = NewFederatedControl()
    control.run()