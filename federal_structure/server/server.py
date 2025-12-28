import asyncio
import argparse
from api import FederatedServerAPI

# ==================== 启动服务器 ====================
def start_server(host="0.0.0.0", port=8000, n_prototypes=5):
    """启动联邦学习服务器"""
    server = FederatedServerAPI(n_prototypes=n_prototypes)
    
    print(f"""
    ╔══════════════════════════════════════╗
    ║    联邦学习服务器启动成功!           ║
    ╠══════════════════════════════════════╣
    ║ 地址: http://{host}:{port}           ║
    ║ 原型数: {n_prototypes}               ║
    ║ 监控间隔: {server.core.status_check_interval}秒  ║
    ╚══════════════════════════════════════╝
    """)
    
    # 启动FastAPI服务器
    import uvicorn
    uvicorn.run(
        server.app, 
        host=host, 
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    # 可以在此处通过命令行参数配置
    parser = argparse.ArgumentParser(description="启动联邦学习服务器")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--prototypes", type=int, default=5, help="原型数量")
    
    args = parser.parse_args()
    
    start_server(
        host=args.host,
        port=args.port,
        n_prototypes=args.prototypes
    )