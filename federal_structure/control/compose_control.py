import os
import subprocess
import sys
import threading
import time
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional


class ComposeControl:
    def __init__(self):
        # 修正docker-compose.yml文件路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # 从control目录向上三级
        self.compose_file = os.path.join(project_root, "docker-compose.yml")
        self.server_url = "http://localhost:8000"  # 本地访问端口映射

    def display_menu(self):
        """显示主菜单"""
        print("\n" + "="*50)
        print("        联邦学习系统 Docker Compose 控制台")
        print("="*50)
        print("1. 启动完整系统 (服务器+客户端)")
        print("2. 仅启动服务器")
        print("3. 仅启动客户端")
        print("4. 扩展客户端数量")
        print("5. 开始训练")
        print("6. 查看已连接客户端")
        print("7. 停止所有服务")
        print("8. 停止客户端")
        print("9. 查看服务器状态")
        print("10. 查看系统日志")
        print("11. 退出")
        print("="*50)

    def start_full_system(self):
        """启动完整系统"""
        print("正在启动完整系统 (服务器和客户端)...")
        
        try:
            # 检查Docker是否可用
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("错误: Docker未安装或未正确配置")
                return
        except FileNotFoundError:
            print("错误: Docker命令未找到，请确保Docker已安装并添加到PATH")
            return
        
        # 检查Docker Compose是否可用
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("错误: Docker Compose未安装或未正确配置")
            return

        # 检查docker-compose.yml是否存在
        if not os.path.exists(self.compose_file):
            print(f"错误: 未找到docker-compose.yml文件: {self.compose_file}")
            return
        
        # 启动系统
        result = subprocess.run([
            'docker-compose', '-f', self.compose_file, 'up', '-d', '--build'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("系统已启动成功！")
            print("服务器地址: http://localhost:8000")
        else:
            print(f"启动失败: {result.stderr}")

    def start_server_only(self):
        """仅启动服务器"""
        print("正在启动服务器...")
        
        try:
            # 检查Docker是否可用
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("错误: Docker未安装或未正确配置")
                return
        except FileNotFoundError:
            print("错误: Docker命令未找到，请确保Docker已安装并添加到PATH")
            return
        
        # 检查Docker Compose是否可用
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("错误: Docker Compose未安装或未正确配置")
            return

        if not os.path.exists(self.compose_file):
            print(f"错误: 未找到docker-compose.yml文件: {self.compose_file}")
            return
        
        # 启动服务器
        result = subprocess.run([
            'docker-compose', '-f', self.compose_file, 'up', '-d', '--build', 'server'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("服务器已启动成功！")
            print("服务器地址: http://localhost:8000")
        else:
            print(f"启动服务器失败: {result.stderr}")

    def start_client_only(self):
        """仅启动客户端"""
        print("正在启动客户端...")
        
        try:
            # 检查Docker是否可用
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("错误: Docker未安装或未正确配置")
                return
        except FileNotFoundError:
            print("错误: Docker命令未找到，请确保Docker已安装并添加到PATH")
            return
        
        # 检查Docker Compose是否可用
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("错误: Docker Compose未安装或未正确配置")
            return

        if not os.path.exists(self.compose_file):
            print(f"错误: 未找到docker-compose.yml文件: {self.compose_file}")
            return
        
        # 启动客户端
        result = subprocess.run([
            'docker-compose', '-f', self.compose_file, 'up', '-d', '--build', 'client'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("客户端已启动成功！")
        else:
            print(f"启动客户端失败: {result.stderr}")

    def scale_clients(self):
        """扩展客户端数量"""
        try:
            num_clients_input = input("请输入要启动的客户端总数 (默认1): ").strip()
            if not num_clients_input:
                num_clients = 1
            else:
                num_clients = int(num_clients_input)
                
            if not os.path.exists(self.compose_file):
                print(f"错误: 未找到docker-compose.yml文件: {self.compose_file}")
                return
            
            # 扩展客户端数量
            result = subprocess.run([
                'docker-compose', '-f', self.compose_file, 'up', '-d', '--scale', f'client={num_clients}', '--build'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"客户端已扩展到 {num_clients} 个")
            else:
                print(f"扩展客户端失败: {result.stderr}")
        except ValueError:
            print("输入的不是有效数字")

    def start_training(self):
        """开始训练"""
        if not self.is_server_running():
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
        if not self.is_server_running():
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

    def stop_all_services(self):
        """停止所有服务"""
        if not os.path.exists(self.compose_file):
            print(f"错误: 未找到docker-compose.yml文件: {self.compose_file}")
            return
        
        result = subprocess.run([
            'docker-compose', '-f', self.compose_file, 'down'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("所有服务已停止")
        else:
            print(f"停止服务失败: {result.stderr}")

    def stop_clients(self):
        """停止客户端"""
        if not os.path.exists(self.compose_file):
            print(f"错误: 未找到docker-compose.yml文件: {self.compose_file}")
            return
        
        result = subprocess.run([
            'docker-compose', '-f', self.compose_file, 'rm', '-f', 'client'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("客户端已停止")
        else:
            print(f"停止客户端失败: {result.stderr}")

    def view_server_status(self):
        """查看服务器状态"""
        if not self.is_server_running():
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

    def view_logs(self):
        """查看系统日志"""
        try:
            if not os.path.exists(self.compose_file):
                print(f"错误: 未找到docker-compose.yml文件: {self.compose_file}")
                return
            
            print("正在显示系统日志 (按Ctrl+C退出)...")
            result = subprocess.run([
                'docker-compose', '-f', self.compose_file, 'logs', '-f'
            ], capture_output=True, text=True)
        except KeyboardInterrupt:
            print("\n已退出日志查看")

    def is_server_running(self):
        """检查服务器是否正在运行"""
        try:
            response = requests.get(f"{self.server_url}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def run(self):
        """运行控制台"""
        print("欢迎使用联邦学习系统 Docker Compose 控制台")
        
        while True:
            self.display_menu()
            choice = input("\n请选择操作 (1-11): ").strip()
            
            if choice == '1':
                self.start_full_system()
            elif choice == '2':
                self.start_server_only()
            elif choice == '3':
                self.start_client_only()
            elif choice == '4':
                self.scale_clients()
            elif choice == '5':
                self.start_training()
            elif choice == '6':
                self.view_connected_clients()
            elif choice == '7':
                self.stop_all_services()
            elif choice == '8':
                self.stop_clients()
            elif choice == '9':
                self.view_server_status()
            elif choice == '10':
                self.view_logs()
            elif choice == '11':
                print("正在退出...")
                break
            else:
                print("无效选择，请重新输入")
            
            input("\n按回车键继续...")


if __name__ == "__main__":
    control = ComposeControl()
    control.run()