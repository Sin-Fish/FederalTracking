```
# 基本启动
python federated_server.py

# 带参数启动
python federated_server.py --host 127.0.0.1 --port 8080 --prototypes 5
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
