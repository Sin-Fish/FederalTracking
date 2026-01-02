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
        print("8. 查看服务器状态")
        print("9. 查看客户端输出")
        print("10. 退出")
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

        print(f"正在启动客户端 (服务器: {server_url}, 隐私级别: {privacy_level}, 数据路径: {data_path})...")

        try:
            # 检查Docker服务是否运行
            result = subprocess.run(["docker", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print("错误: Docker服务未运行，请先启动Docker Desktop或Docker服务")
                return

            # 设置环境变量
            env = os.environ.copy()
            env["SERVER_URL"] = server_url
            env["PRIVACY_LEVEL"] = privacy_level
            env["DATA_PATH"] = data_path

            # 启动客户端
            up_result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "up", "-d", "client"],
                capture_output=True, text=True, env=env
            )
            
            if up_result.returncode == 0:
                print("客户端容器已启动")
                return
            else:
                print(f"创建客户端容器失败: {up_result.stderr}")
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

            # 停止客户端容器
            result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "stop", "client"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                print("客户端已停止")
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

            # 停止所有客户端容器
            result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "stop", "client"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                print("所有客户端已停止")
            else:
                print(f"停止客户端失败: {result.stderr}")
                
        except Exception as e:
            print(f"停止客户端时发生错误: {e}")

    def check_server_status(self):
        """检查服务器状态"""
        try:
            # 检查服务器容器状态
            result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "ps", "server"],
                capture_output=True, text=True
            )
            
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
            response = requests.get(f"{self.server_url}/clients", timeout=10)
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
            response = requests.post(f"{self.server_url}/start_training", timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"训练启动结果: {result}")
            else:
                print(f"启动训练失败: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"启动训练请求失败: {e}")

    def view_client_output(self):
        """查看客户端输出"""
        try:
            # 查看客户端日志
            result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "logs", "client"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                print("客户端日志:")
                print(result.stdout[-2000:])  # 显示最后2000个字符
            else:
                print(f"获取客户端日志失败: {result.stderr}")
                
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
                self.check_server_status()
            elif choice == "9":
                self.view_client_output()
            elif choice == "10":
                print("正在退出...")
                break
            else:
                print("无效选择，请重试！")

            input("\n按回车键继续...")


if __name__ == "__main__":
    control = NewFederatedControl()
    control.run()