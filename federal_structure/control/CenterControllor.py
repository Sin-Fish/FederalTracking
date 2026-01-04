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
        self.compose_file = os.path.join(self.project_root, "docker-compose.yml")

    def display_menu(self):
        """显示主菜单"""
        print("\n" + "="*50)
        print("        联邦学习系统控制台 (容器版)")
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
        """启动服务器容器"""
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
            # 检查Docker服务是否运行
            result = subprocess.run(["docker", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print("错误: Docker服务未运行，请先启动Docker Desktop或Docker服务")
                return

            # 设置环境变量
            env = os.environ.copy()
            env["PROTOTYPES"] = prototypes
            env["PORT"] = port
            env["SERVER_HOST"] = host

            # 检查容器状态
            result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "ps", "server"],
                capture_output=True, text=True, env=env
            )
            
            # 检查是否已经运行或已停止
            if "Up" in result.stdout:
                print("服务器容器已在运行中！")
                self.is_server_running = True
                return
            elif "Exit" in result.stdout or "Exited" in result.stdout:
                # 容器已存在但已停止，启动现有容器
                print("发现已停止的服务器容器，正在启动...")
                start_result = subprocess.run(
                    ["docker-compose", "-f", self.compose_file, "start", "server"],
                    capture_output=True, text=True, env=env
                )
                if start_result.returncode == 0:
                    print("服务器容器已启动")
                    self.is_server_running = True
                    # 等待一段时间以确保服务器启动
                    time.sleep(5)
                    return
                else:
                    print(f"启动服务器容器失败: {start_result.stderr}")
                    return
            else:
                # 容器不存在，需要创建并启动
                print("正在创建并启动服务器容器...")
                
                # 启动服务器
                up_result = subprocess.run(
                    ["docker-compose", "-f", self.compose_file, "up", "-d", "server"],
                    capture_output=True, text=True, env=env
                )
                
                if up_result.returncode == 0:
                    print("服务器容器已启动")
                    self.is_server_running = True
                    # 等待一段时间以确保服务器启动
                    time.sleep(5)
                    return
                else:
                    print(f"创建服务器容器失败: {up_result.stderr}")
                    return
                    
        except Exception as e:
            print(f"启动服务器时发生错误: {e}")

    def start_client(self):
        """启动客户端容器"""
        print("启动客户端Docker容器配置:")

        # 获取服务器地址
        server_url = input("请输入服务器地址 (默认 http://server:8000): ").strip()
        if not server_url:
            server_url = "http://server:8000"

        # 获取隐私级别
        privacy_level = input("请输入隐私级别 (low/medium/high，默认medium): ").strip()
        if not privacy_level or privacy_level not in ["low", "medium", "high"]:
            privacy_level = "medium"

        # 获取数据文件路径
        data_path = input("请输入数据文件路径 (默认 /app/data/activity_log.jsonl): ").strip()
        if not data_path:
            data_path = "/app/data/activity_log.jsonl"
            
        # 为每个客户端生成唯一ID
        client_id = f"federated-client-{int(time.time())}"
        
        print(f"正在创建并启动客户端容器 {client_id} (服务器: {server_url}, 隐私级别: {privacy_level}, 数据路径: {data_path})...")

        try:
            # 检查Docker服务是否运行
            result = subprocess.run(["docker", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print("错误: Docker服务未运行，请先启动Docker Desktop或Docker服务")
                return

            # 检查服务器是否运行
            server_check = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "ps", "server"],
                capture_output=True, text=True
            )
            if "Up" not in server_check.stdout:
                print("警告: 服务器似乎没有运行，客户端可能无法连接到服务器")
                
            # 检查是否存在客户端镜像
            image_check = subprocess.run(
                ["docker", "images", "-q", "federated-client:latest"],
                capture_output=True, text=True
            )
            
            if not image_check.stdout.strip():
                print("未找到客户端镜像，正在构建...")
                # 构建客户端镜像
                build_result = subprocess.run(
                    ["docker-compose", "-f", self.compose_file, "build", "client"],
                    capture_output=True, text=True
                )
                
                if build_result.returncode != 0:
                    print(f"构建客户端镜像失败: {build_result.stderr}")
                    return
                else:
                    print("客户端镜像构建成功")
            else:
                print("检测到已存在的客户端镜像，直接使用...")
                
            # 运行客户端容器
            cmd = [
                "docker", "run", 
                "-d",  # 后台运行
                "--name", client_id,  # 使用唯一名称
                "--network", "federaltracking_federated-network",  # 连接到联邦网络
                "-v", f"{os.path.join(self.project_root, 'data')}:/app/data",
                "-v", f"{os.path.join(self.project_root, 'federal_structure/client')}:/app",
                "-e", f"SERVER_URL={server_url}",
                "-e", f"PRIVACY_LEVEL={privacy_level}",
                "-e", f"DATA_PATH={data_path}",
                "--restart", "unless-stopped",
                "federated-client:latest"  # 使用预构建镜像
            ]
            
            run_result = subprocess.run(cmd, capture_output=True, text=True)
            
            if run_result.returncode == 0:
                print(f"客户端容器 {client_id} 已启动")
                return
            else:
                print(f"启动客户端容器失败: {run_result.stderr}")
                return
                
        except Exception as e:
            print(f"启动客户端时发生错误: {e}")

    def stop_server(self):
        """停止服务器容器"""
        try:
            # 检查Docker服务是否运行
            result = subprocess.run(["docker", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print("错误: Docker服务未运行")
                return

            # 停止服务器容器
            result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "stop", "server"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                print("服务器已停止")
                self.is_server_running = False
            else:
                print(f"停止服务器失败: {result.stderr}")
                
        except Exception as e:
            print(f"停止服务器时发生错误: {e}")

    def stop_client(self):
        """停止单个客户端容器"""
        try:
            # 检查Docker服务是否运行
            result = subprocess.run(["docker", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print("错误: Docker服务未运行")
                return

            # 获取所有运行中的客户端容器
            list_result = subprocess.run(
                ["docker", "ps", "-f", "name=federated-client-", "--format", "{{.Names}}"],
                capture_output=True, text=True
            )
            
            if not list_result.stdout.strip():
                print("没有找到客户端容器")
                return
            
            client_containers = list_result.stdout.strip().split('\n')
            if len(client_containers) == 1 and client_containers[0] == '':
                print("没有找到客户端容器")
                return
            
            print("当前运行的客户端:")
            for i, container in enumerate(client_containers):
                print(f"{i+1}. {container}")
            
            if len(client_containers) == 1:
                selected_container = client_containers[0]
            else:
                choice = input(f"请选择要停止的客户端 (1-{len(client_containers)}): ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(client_containers):
                        selected_container = client_containers[idx]
                    else:
                        print("无效选择")
                        return
                except ValueError:
                    print("无效输入")
                    return

            # 停止选定的客户端容器
            result = subprocess.run(
                ["docker", "stop", selected_container],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                print(f"客户端 {selected_container} 已停止")
            else:
                print(f"停止客户端失败: {result.stderr}")
                
        except Exception as e:
            print(f"停止客户端时发生错误: {e}")

    def stop_all_clients(self):
        """停止所有客户端容器"""
        try:
            # 检查Docker服务是否运行
            result = subprocess.run(["docker", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print("错误: Docker服务未运行")
                return

            # 停止所有以 federated-client- 开头的容器
            result = subprocess.run(
                ["docker", "stop", "$(docker", "ps", "-q", "-f", "name=federated-client-)"],
                shell=True,  # 使用shell执行命令
                capture_output=True, text=True
            )
            
            # 如果上面的命令失败，尝试使用更简单的方式
            list_result = subprocess.run(
                ["docker", "ps", "-f", "name=federated-client-", "-q"],
                capture_output=True, text=True
            )
            
            if list_result.stdout.strip():
                container_ids = list_result.stdout.strip().split('\n')
                for container_id in container_ids:
                    if container_id:
                        stop_result = subprocess.run(
                            ["docker", "stop", container_id],
                            capture_output=True, text=True
                        )
                        if stop_result.returncode != 0:
                            print(f"停止客户端 {container_id} 失败: {stop_result.stderr}")
                        else:
                            print(f"客户端 {container_id} 已停止")
            else:
                print("没有找到客户端容器")
                
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