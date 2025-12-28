# 联邦学习客户端

联邦学习客户端用于在本地处理用户行为数据，训练模型并参与联邦学习过程，同时保护用户隐私。

## 功能特性

- **隐私保护**：对敏感信息进行过滤和脱敏处理
- **数据处理**：将用户行为数据转换为行为字符串
- **本地训练**：根据服务器原型进行本地模型训练
- **联邦通信**：与服务器进行安全通信

## 架构设计

客户端采用模块化设计，分为以下几个组件：

- `models.py`：机器学习模型定义
- `privacy_filter.py`：隐私过滤器
- `data_processor.py`：数据处理模块
- `communication.py`：通信模块
- `training.py`：训练模块
- `client.py`：客户端主控逻辑

## 依赖安装

```bash
pip install -r requirements.txt
```

## 启动客户端

```bash
python client.py --server http://localhost:8000 --privacy medium --data ./data.json
```

参数说明：
- `--server`：服务器地址，默认为http://localhost:8000
- `--client-id`：客户端ID（默认自动生成）
- `--data`：数据文件路径（JSON或JSONL格式）
- `--privacy`：隐私保护级别（low/medium/high），默认为medium
- `--simulate`：使用模拟数据
- `--register-only`：仅注册不训练

## 隐私保护级别

- `low`：保留最多信息
- `medium`：平衡隐私与效用（推荐）
- `high`：最大程度隐私保护

## 数据处理流程

1. 加载用户行为数据
2. 应用隐私过滤器处理敏感信息
3. 生成行为字符串
4. 向服务器注册并提交数据指纹
5. 提交高频行为数据用于聚类
6. 接收全局原型并进行本地训练
7. 提交模型更新到服务器

## 数据格式

客户端支持以下数据格式：
- JSON：数组格式
- JSONL：每行一个JSON对象

数据应包含以下字段：
- `timestamp`：时间戳
- `environment`：环境信息
  - `process_name`：进程名称
  - `window_title`：窗口标题
- `event_type`：事件类型（key_press, mouse_click, mouse_scroll）
- `keyboard_event`：键盘事件（可选）
- `mouse_event`：鼠标事件（可选）