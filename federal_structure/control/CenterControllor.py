import os
import subprocess
import sys
import threading
import time
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
import yaml


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
            result = subprocess.run(["docker", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
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
                capture_output=True, text=True, env=env, encoding='utf-8'
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
                    capture_output=True, text=True, env=env, encoding='utf-8'
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
                    capture_output=True, text=True, env=env, encoding='utf-8'
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
        
        # 1. 获取服务器地址
        server_url = input("请输入服务器地址 (默认 http://server:8000): ").strip()
        if not server_url:
            server_url = "http://server:8000"
        
        # 2. 获取隐私级别
        privacy_level = input("请输入隐私级别 (low/medium/high，默认medium): ").strip()
        if not privacy_level or privacy_level not in ["low", "medium", "high"]:
            privacy_level = "medium"
        
        # 3. 获取宿主机上的数据文件路径
        host_data_path = input("请输入数据文件在宿主机上的完整路径: ").strip()
        if not host_data_path:
            print("错误: 必须提供数据文件路径")
            return
        
        # 4. 验证文件存在
        if not os.path.exists(host_data_path):
            print(f"错误: 数据文件不存在: {host_data_path}")
            return
        
        # 5. 提取目录和文件名
        data_dir = os.path.dirname(host_data_path) or "."
        data_filename = os.path.basename(host_data_path)
        
        # 6. 生成唯一客户端ID
        client_id = f"federated-client-{int(time.time())}"
        
        print(f"正在创建并启动客户端容器 {client_id}")
        print(f"  - 服务器: {server_url}")
        print(f"  - 隐私级别: {privacy_level}")
        print(f"  - 数据文件: {host_data_path} → /app/data/input/{data_filename}")
        
        try:
            # 检查Docker服务是否运行
            result = subprocess.run(["docker", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            if result.returncode != 0:
                print("错误: Docker服务未运行，请先启动Docker Desktop或Docker服务")
                return

            # 检查服务器是否运行
            server_check = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "ps", "server"],
                capture_output=True, text=True, encoding='utf-8'
            )
            if "Up" not in server_check.stdout:
                print("警告: 服务器似乎没有运行，客户端可能无法连接到服务器")
            
            # 检查客户端镜像是否存在，不存在则构建
            image_check = subprocess.run(
                ["docker", "images", "-q", "federated-client:latest"],
                capture_output=True, text=True, encoding='utf-8'
            )
            
            if not image_check.stdout.strip():
                print("\n" + "="*60)
                print("未找到客户端镜像，正在构建...")
                print("="*60)
                # 构建客户端镜像，实时显示输出
                build_cmd = [
                    "docker", "build", 
                    "-t", "federated-client:latest",
                    "-f", os.path.join(self.project_root, "federal_structure/client/Dockerfile"),
                    os.path.join(self.project_root, "federal_structure/client")
                ]
                
                # 直接运行，实时显示输出（不使用 capture_output）
                try:
                    build_result = subprocess.run(
                        build_cmd,
                        encoding='utf-8'
                    )  # 不捕获输出，直接显示到控制台
                    
                    print("="*60)
                    if build_result.returncode != 0:
                        print(f"\n❌ 构建客户端镜像失败 (退出码: {build_result.returncode})")
                        return
                    else:
                        print("\n✓ 客户端镜像构建成功")
                except KeyboardInterrupt:
                    print("\n\n⚠️ 构建被用户中断")
                    return
                except Exception as e:
                    print(f"\n❌ 构建过程中发生错误: {e}")
                    return
            
            # 构建容器内的数据文件完整路径
            container_data_path = f"/app/data/input/{data_filename}"
            
            # 构建并运行容器
            cmd = [
                "docker", "run", 
                "-d",
                "--name", client_id,
                "--network", "federaltracking_federated-network",
                # 挂载代码目录（如果需要）
                "-v", f"{os.path.join(self.project_root, 'federal_structure/client')}:/app/code:ro",
                # 挂载数据目录
                "-v", f"{data_dir}:/app/data/input:ro",  # 只读挂载
                # 环境变量
                "-e", f"SERVER_URL={server_url}",
                "-e", f"PRIVACY_LEVEL={privacy_level}",
                "-e", f"DATA_FILE={data_filename}",
                "-e", f"DATA_DIR=/app/data/input",
                "-e", "PYTHONPATH=/app/code",  # 设置Python模块搜索路径
                # 其他可选参数
                "--restart", "unless-stopped",
                "federated-client:latest",
                "python", "/app/code/client.py", "--data", container_data_path  # 使用挂载的代码并传递数据路径参数
            ]
            
            # 运行容器
            print("正在创建并启动客户端容器...")
            run_result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if run_result.returncode != 0:
                print(f"✗ 创建客户端容器失败: {run_result.stderr}")
                return
            
            container_id = run_result.stdout.strip()
            print(f"  容器已创建: {container_id}")
            print("  等待容器启动并验证状态...")
            
            # 等待容器启动（最多等待30秒）
            max_wait_time = 30
            wait_interval = 2
            waited_time = 0
            startup_failed = False
            error_message = ""
            
            while waited_time < max_wait_time:
                time.sleep(wait_interval)
                waited_time += wait_interval
                
                # 检查容器状态
                status_result = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Status}}", client_id],
                    capture_output=True, text=True, encoding='utf-8'
                )
                
                if status_result.returncode != 0:
                    error_message = f"无法检查容器状态: {status_result.stderr}"
                    startup_failed = True
                    break
                
                container_status = status_result.stdout.strip()
                
                if container_status == "running":
                    # 容器在运行，检查日志看是否有启动错误
                    print("  容器正在运行，检查启动日志...")
                    log_result = subprocess.run(
                        ["docker", "logs", "--tail", "50", client_id],
                        capture_output=True, text=True, encoding='utf-8',
                        timeout=10
                    )
                    
                    if log_result.returncode == 0:
                        logs = log_result.stdout
                        # 检查是否有明显的错误或成功标志
                        error_keywords = [
                            "ImportError", "ModuleNotFoundError", "FileNotFoundError",
                            "启动失败", "启动异常", "❌"
                        ]
                        success_keywords = [
                            "环境检查完成", "开始运行联邦学习客户端完整流程",
                            "✓ PyTorch", "✓ transformers", "✓ sentence-transformers"
                        ]
                        
                        has_error = any(keyword in logs for keyword in error_keywords)
                        has_success = any(keyword in logs for keyword in success_keywords)
                        
                        if has_error and not has_success:
                            error_message = "容器启动时检测到错误"
                            startup_failed = True
                            print(f"\n  检测到启动错误，容器日志：")
                            print("  " + "="*56)
                            # 显示最后30行日志
                            for line in logs.split('\n')[-30:]:
                                if line.strip():
                                    print(f"  {line}")
                            print("  " + "="*56)
                            break
                        elif has_success:
                            # 启动成功
                            print(f"✓ 客户端容器 {client_id} 启动成功")
                            print(f"  容器ID: {container_id}")
                            print(f"  可以查看完整日志: docker logs {client_id}")
                            return
                    
                    # 如果日志检查不明确，再等待一下
                    if waited_time < max_wait_time:
                        continue
                    else:
                        # 超时但容器在运行，可能是启动较慢，给出警告
                        print(f"⚠️ 容器已运行但未能确认启动状态（已等待{waited_time}秒）")
                        print(f"  容器ID: {container_id}")
                        print(f"  请手动检查日志: docker logs {client_id}")
                        return
                        
                elif container_status == "exited":
                    # 容器已退出，说明启动失败
                    error_message = "容器启动后立即退出"
                    startup_failed = True
                    break
                elif container_status == "dead":
                    error_message = "容器状态为 dead"
                    startup_failed = True
                    break
                # 其他状态（creating, restarting等）继续等待
                
            # 处理启动失败
            if startup_failed or waited_time >= max_wait_time:
                if waited_time >= max_wait_time:
                    error_message = f"等待容器启动超时（{max_wait_time}秒）"
                
                print(f"\n✗ 客户端容器启动失败: {error_message}")
                
                # 获取容器日志
                print("\n  容器日志（最后50行）：")
                print("  " + "="*56)
                log_result = subprocess.run(
                    ["docker", "logs", "--tail", "50", client_id],
                    capture_output=True, text=True, encoding='utf-8',
                    timeout=10
                )
                if log_result.returncode == 0:
                    for line in log_result.stdout.split('\n'):
                        if line.strip():
                            print(f"  {line}")
                print("  " + "="*56)
                
                # 删除失败的容器
                print(f"\n  正在删除失败的容器 {client_id}...")
                delete_result = subprocess.run(
                    ["docker", "rm", "-f", client_id],
                    capture_output=True, text=True, encoding='utf-8'
                )
                if delete_result.returncode == 0:
                    print(f"  ✓ 容器已删除")
                else:
                    print(f"  ⚠️ 删除容器失败: {delete_result.stderr}")
                
                print(f"\n❌ 客户端启动失败，容器已删除")
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

            # 获取所有federated-client-开头的容器ID
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
            # 检查服务器容器状态
            result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "ps", "server"],
                capture_output=True, text=True
            )
            
            if "Up" in result.stdout:
                print("服务器正在运行")
                # 尝试连接服务器API
                try:
                    response = requests.get(f"http://localhost:8000/", timeout=10)  # 修正API端点
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
                for client_id, client_info in clients.items():
                    print(f"- {client_id}: {client_info.get('status', 'unknown')}")
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
            container_names = [name.split(':')[0] for name in client_containers if ':' in name]
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
            choice = input("请选择操作 (1-8): ").strip()  # 修正菜单选项数量

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