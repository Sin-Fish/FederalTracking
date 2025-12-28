```
# 使用默认设置运行（自动生成模拟数据）
python federated_client.py

# 指定数据文件
python federated_client.py --data ./my_activity_log.json

# 指定服务器地址
python federated_client.py --server http://192.168.1.100:8000

# 指定隐私级别
python federated_client.py --privacy high

# 仅注册到服务器
python federated_client.py --register-only --data ./my_data.json
```

| 端点                         | 方法 | 功能             | 请求体/参数        |
| ---------------------------- | ---- | ---------------- | ------------------ |
| `/`                        | GET  | 服务器状态检查   | 无                 |
| `/api/client/register`     | POST | 客户端报到       | `ClientRegister` |
| `/api/client/status`       | POST | 客户端状态上报   | `ClientStatus`   |
| `/api/data/collect`        | POST | 数据征收         | `DataSubmission` |
| `/api/training/start`      | GET  | 启动联邦训练     | 无                 |
| `/api/model/update`        | POST | 接收模型更新     | `ModelUpdate`    |
| `/api/system/prototypes`   | GET  | 获取全局原型     | 无                 |
| `/api/system/clients`      | GET  | 获取客户端状态   | 无                 |
| `/api/federated/aggregate` | POST | 手动触发模型聚合 | 无                 |
