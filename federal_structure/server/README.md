# 联邦学习服务器

联邦学习服务器用于协调多个客户端共同训练机器学习模型，同时保护各客户端的数据隐私。

## 功能特性

- **客户端管理**：注册、状态监控、心跳检测
- **联邦聚类**：根据客户端数据特征进行行为模式聚类
- **模型聚合**：实现联邦平均算法，聚合各客户端的模型更新
- **API接口**：提供RESTful API供客户端调用

## 架构设计

服务器采用模块化设计，分为以下几个组件：

- `models.py`：数据模型定义
- `ml_models.py`：机器学习模型定义
- `core.py`：核心业务逻辑
- `api.py`：API路由定义
- `server.py`：服务器启动入口

## 依赖安装

```bash
pip install -r requirements.txt
```

## 启动服务器

```bash
python server.py --host 0.0.0.0 --port 8000 --prototypes 2
```

参数说明：

- `--host`：监听地址，默认为0.0.0.0
- `--port`：监听端口，默认为8000
- `--prototypes`：原型数量，默认为5

## API接口

- `GET /`：服务器状态信息
- `POST /api/client/register`：客户端注册
- `POST /api/client/status`：客户端状态更新
- `POST /api/data/collect`：数据征收
- `GET /api/training/start`：启动训练
- `POST /api/model/update`：提交模型更新
- `GET /api/system/prototypes`：获取全局原型
- `GET /api/system/clients`：获取客户端状态
- `POST /api/federated/aggregate`：手动触发模型聚合

## 数据流

1. 客户端注册并提交数据指纹
2. 客户端提交高频行为数据
3. 服务器执行聚类生成全局原型
4. 客户端根据原型进行本地训练
5. 客户端提交模型更新
6. 服务器聚合模型更新
