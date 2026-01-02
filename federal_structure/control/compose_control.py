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
        print("9. 停止服务器")
        print("10. 查看服务器状态")
        print("11. 查看系统日志")
        print("12. 退出")
        print("="*50)

    def start_full_system_no_build(self):
        """启动完整系统（不构建）"""
        print("正在启动完整系统 (服务器和客户端)...")
        
        try:
            # 检查Docker是否可用
            print("检查Docker是否可用...")
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("错误: Docker未安装或未正确配置")
                return
            print("Docker检查通过")
        except FileNotFoundError:
            print("错误: Docker命令未找到，请确保Docker已安装并添加到PATH")
            return
        except subprocess.TimeoutExpired:
            print("错误: Docker命令执行超时")
            return
        
        # 检查Docker Compose是否可用
        print("检查Docker Compose是否可用...")
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("错误: Docker Compose未安装或未正确配置")
            return
        print("Docker Compose检查通过")

        # 检查docker-compose.yml是否存在
        print(f"检查docker-compose.yml文件是否存在: {self.compose_file}")
        if not os.path.exists(self.compose_file):
            print(f"错误: 未找到docker-compose.yml文件: {self.compose_file}")
            return
        print("docker-compose.yml文件存在")
        
        print("正在启动系统...")
        print("注意：这将尝试启动现有容器，保留容器状态")
        
        # 尝试启动现有容器，如果不存在则创建
        try:
            # 首先检查是否存在已停止的容器
            ps_result = subprocess.run([
                'docker-compose', '-f', self.compose_file, 'ps', '--status', 'exited'
            ], capture_output=True, text=True)
            
            # 如果存在已停止的容器，使用start命令启动
            if ps_result.stdout and ('server' in ps_result.stdout or 'client' in ps_result.stdout):
                print("检测到已停止的容器，正在启动现有容器...")
                start_result = subprocess.run([
                    'docker-compose', '-f', self.compose_file, 'start'
                ], capture_output=True, text=True)
                
                if start_result.returncode != 0:
                    print("启动现有容器失败，尝试使用up命令...")
                    # 如果start失败，使用up命令
                    result = subprocess.run([
                        'docker-compose', '-f', self.compose_file, 'up', '-d'
                    ], capture_output=True, text=True, timeout=300)  # 5分钟超时
                else:
                    print("现有容器已启动成功！")
                    result = start_result
            else:
                # 如果没有已停止的容器，使用up命令
                result = subprocess.run([
                    'docker-compose', '-f', self.compose_file, 'up', '-d'
                ], capture_output=True, text=True, timeout=300)  # 5分钟超时
            
            if result.returncode == 0:
                print("系统已启动成功！")
                print("服务器地址: http://localhost:8000")
                
                # 检查容器是否正在运行
                print("正在检查容器状态...")
                status_result = subprocess.run([
                    'docker-compose', '-f', self.compose_file, 'ps'
                ], capture_output=True, text=True)
                
                print("容器状态:")
                print(status_result.stdout)
            else:
                print(f"启动失败: {result.stderr}")
                print("请检查错误信息并重试")
        except subprocess.TimeoutExpired:
            print("启动超时，请检查手动启动系统")
            print("您也可以尝试单独启动服务器和客户端")

    def start_server_only_no_build(self):
        """仅启动服务器（不构建）"""
        print("正在启动服务器...")
        
        try:
            # 检查Docker是否可用
            print("检查Docker是否可用...")
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("错误: Docker未安装或未正确配置")
                return
            print("Docker检查通过")
        except FileNotFoundError:
            print("错误: Docker命令未找到，请确保Docker已安装并添加到PATH")
            return
        except subprocess.TimeoutExpired:
            print("错误: Docker命令执行超时")
            return
        
        # 检查Docker Compose是否可用
        print("检查Docker Compose是否可用...")
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("错误: Docker Compose未安装或未正确配置")
            return
        print("Docker Compose检查通过")

        if not os.path.exists(self.compose_file):
            print(f"错误: 未找到docker-compose.yml文件: {self.compose_file}")
            return
        
        print("正在启动服务器...")
        print("注意：这将尝试启动现有服务器容器，保留容器状态")
        
        try:
            # 检查服务器容器状态
            ps_result = subprocess.run([
                'docker-compose', '-f', self.compose_file, 'ps', 'server'
            ], capture_output=True, text=True)
            
            # 检查输出中是否包含容器信息
            if ps_result.stdout and 'server' in ps_result.stdout:
                # 检查容器是否已停止
                if 'Exit' in ps_result.stdout or 'Exited' in ps_result.stdout:
                    print("检测到已停止的服务器容器，正在启动...")
                    start_result = subprocess.run([
                        'docker-compose', '-f', self.compose_file, 'start', 'server'
                    ], capture_output=True, text=True)
                    
                    if start_result.returncode == 0:
                        print("服务器容器已启动成功！")
                    else:
                        print(f"启动现有服务器容器失败: {start_result.stderr}")
                elif 'Up' in ps_result.stdout:
                    print("服务器已在运行中")
                else:
                    print("服务器容器状态未知")
            else:
                print("未找到服务器容器，需要先创建")
            
            # 检查容器是否正在运行
            print("正在检查服务器容器状态...")
            status_result = subprocess.run([
                'docker-compose', '-f', self.compose_file, 'ps', 'server'
            ], capture_output=True, text=True)
            
            print("服务器容器状态:")
            print(status_result.stdout)
        except subprocess.TimeoutExpired:
            print("启动服务器超时，请检查手动启动服务器")

    def start_client_only_no_build(self):
        """仅启动客户端（不构建）"""
        print("正在启动客户端...")
        
        try:
            # 检查Docker是否可用
            print("检查Docker是否可用...")
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("错误: Docker未安装或未正确配置")
                return
            print("Docker检查通过")
        except FileNotFoundError:
            print("错误: Docker命令未找到，请确保Docker已安装并添加到PATH")
            return
        except subprocess.TimeoutExpired:
            print("错误: Docker命令执行超时")
            return
        
        # 检查Docker Compose是否可用
        print("检查Docker Compose是否可用...")
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("错误: Docker Compose未安装或未正确配置")
            return
        print("Docker Compose检查通过")

        if not os.path.exists(self.compose_file):
            print(f"错误: 未找到docker-compose.yml文件: {self.compose_file}")
            return
        
        print("正在启动客户端...")
        print("注意：这将尝试启动现有客户端容器，保留容器状态")
        
        try:
            # 检查客户端容器状态
            ps_result = subprocess.run([
                'docker-compose', '-f', self.compose_file, 'ps', 'client'
            ], capture_output=True, text=True)
            
            # 检查输出中是否包含容器信息
            if ps_result.stdout and 'client' in ps_result.stdout:
                # 检查容器是否已停止
                if 'Exit' in ps_result.stdout or 'Exited' in ps_result.stdout:
                    print("检测到已停止的客户端容器，正在启动...")
                    start_result = subprocess.run([
                        'docker-compose', '-f', self.compose_file, 'start', 'client'
                    ], capture_output=True, text=True)
                    
                    if start_result.returncode == 0:
                        print("客户端容器已启动成功！")
                    else:
                        print(f"启动现有客户端容器失败: {start_result.stderr}")
                elif 'Up' in ps_result.stdout:
                    print("客户端已在运行中")
                else:
                    print("客户端容器状态未知")
            else:
                print("未找到客户端容器，需要先创建")
            
            # 检查容器是否正在运行
            print("正在检查客户端容器状态...")
            status_result = subprocess.run([
                'docker-compose', '-f', self.compose_file, 'ps', 'client'
            ], capture_output=True, text=True)
            
            print("客户端容器状态:")
            print(status_result.stdout)
        except subprocess.TimeoutExpired:
            print("启动客户端超时，请检查手动启动客户端")

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
            
            print(f"正在将客户端扩展到 {num_clients} 个...")
            print("注意：这将保留现有的客户端容器，并根据需要创建或停止容器以达到指定数量")
            
            # 扩展客户端数量
            result = subprocess.run([
                'docker-compose', '-f', self.compose_file, 'up', '-d', '--scale', f'client={num_clients}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"客户端已扩展到 {num_clients} 个")
                
                # 显示当前容器状态
                status_result = subprocess.run([
                    'docker-compose', '-f', self.compose_file, 'ps'
                ], capture_output=True, text=True)
                
                print("当前容器状态:")
                print(status_result.stdout)
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
            print("正在开始训练...")
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
            print("正在获取客户端列表...")
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
        
        print("正在停止所有服务...")
        # 使用stop命令而不是down命令，stop只是停止容器，不删除
        result = subprocess.run([
            'docker-compose', '-f', self.compose_file, 'stop'
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
        
        print("正在停止客户端...")
        # 使用stop命令而不是rm命令，stop只是停止容器，不删除
        result = subprocess.run([
            'docker-compose', '-f', self.compose_file, 'stop', 'client'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("客户端已停止")
        else:
            print(f"停止客户端失败: {result.stderr}")

    def stop_server(self):
        """停止服务器"""
        if not os.path.exists(self.compose_file):
            print(f"错误: 未找到docker-compose.yml文件: {self.compose_file}")
            return
        
        print("正在停止服务器...")
        # 使用stop命令而不是rm命令，stop只是停止容器，不删除
        result = subprocess.run([
            'docker-compose', '-f', self.compose_file, 'stop', 'server'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("服务器已停止")
        else:
            print(f"停止服务器失败: {result.stderr}")

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
            choice = input("\n请选择操作 (1-12): ").strip()
            
            if choice == '1':
                self.start_full_system_no_build()
            elif choice == '2':
                self.start_server_only_no_build()
            elif choice == '3':
                self.start_client_only_no_build()
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
                self.stop_server()
            elif choice == '10':
                self.view_server_status()
            elif choice == '11':
                self.view_logs()
            elif choice == '12':
                print("正在退出...")
                break
            else:
                print("无效选择，请重新输入")
            
            input("\n按回车键继续...")


if __name__ == "__main__":
    control = ComposeControl()
    control.run()