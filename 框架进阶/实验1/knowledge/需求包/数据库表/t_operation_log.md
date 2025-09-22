# 操作日志表 (t_operation_log)

## 1. 表概述

操作日志表是智慧酒店包商品模板化需求的系统日志表，存储系统的各种操作日志信息，包括用户操作、系统操作、业务操作、错误日志等，为系统监控、问题排查、审计追踪、性能分析等提供数据支持。

### 1.1 表定位
- **表类型**：系统日志表
- **主要用途**：存储操作日志信息
- **业务模块**：系统监控、日志管理、审计追踪
- **数据特点**：日志数据，更新频率高

### 1.2 表特点
- **记录完整**：记录所有重要操作
- **结构清晰**：字段定义明确，数据类型合理
- **扩展性好**：支持日志信息的扩展
- **查询高效**：优化的索引设计

## 2. 表结构

### 2.1 字段定义

| 字段名 | 数据类型 | 长度 | 是否为空 | 默认值 | 说明 |
|--------|----------|------|----------|--------|------|
| log_id | VARCHAR | 50 | NOT NULL | - | 日志ID，主键 |
| user_id | VARCHAR | 50 | NULL | NULL | 用户ID |
| operation_type | VARCHAR | 50 | NOT NULL | - | 操作类型 |
| operation_module | VARCHAR | 50 | NOT NULL | - | 操作模块 |
| operation_action | VARCHAR | 100 | NOT NULL | - | 操作动作 |
| operation_description | VARCHAR | 500 | NULL | NULL | 操作描述 |
| request_url | VARCHAR | 500 | NULL | NULL | 请求URL |
| request_method | VARCHAR | 10 | NULL | NULL | 请求方法 |
| request_params | TEXT | - | NULL | NULL | 请求参数 |
| response_data | TEXT | - | NULL | NULL | 响应数据 |
| operation_status | VARCHAR | 20 | NOT NULL | 'success' | 操作状态 |
| error_message | TEXT | - | NULL | NULL | 错误信息 |
| ip_address | VARCHAR | 50 | NULL | NULL | IP地址 |
| user_agent | VARCHAR | 500 | NULL | NULL | 用户代理 |
| operation_time | DATETIME | - | NOT NULL | CURRENT_TIMESTAMP | 操作时间 |
| duration | INT | - | NULL | NULL | 操作耗时(毫秒) |
| create_time | DATETIME | - | NOT NULL | CURRENT_TIMESTAMP | 创建时间 |

### 2.2 字段详细说明

#### 2.2.1 主键字段
- **log_id**：日志唯一标识，格式为"LOG_YYYYMMDD_序号"
- 示例：LOG_20241220_001

#### 2.2.2 用户字段
- **user_id**：用户ID，关联用户表

#### 2.2.3 操作字段
- **operation_type**：操作类型，可选值：login(登录)、logout(登出)、create(创建)、update(更新)、delete(删除)、query(查询)、export(导出)、import(导入)
- **operation_module**：操作模块，可选值：user(用户)、product(产品)、order(订单)、system(系统)、auth(认证)
- **operation_action**：操作动作，具体的操作描述
- **operation_description**：操作描述，详细的操作说明

#### 2.2.4 请求字段
- **request_url**：请求URL
- **request_method**：请求方法，GET、POST、PUT、DELETE等
- **request_params**：请求参数，JSON格式
- **response_data**：响应数据，JSON格式

#### 2.2.5 状态字段
- **operation_status**：操作状态，可选值：success(成功)、failed(失败)、error(错误)
- **error_message**：错误信息，操作失败时的错误详情

#### 2.2.6 环境字段
- **ip_address**：IP地址
- **user_agent**：用户代理
- **operation_time**：操作时间
- **duration**：操作耗时，单位毫秒

#### 2.2.7 系统字段
- **create_time**：记录创建时间

## 3. 索引设计

### 3.1 主键索引
```sql
PRIMARY KEY (log_id)
```

### 3.2 普通索引
```sql
-- 用户ID索引
CREATE INDEX idx_user_id ON t_operation_log(user_id);

-- 操作类型索引
CREATE INDEX idx_operation_type ON t_operation_log(operation_type);

-- 操作模块索引
CREATE INDEX idx_operation_module ON t_operation_log(operation_module);

-- 操作状态索引
CREATE INDEX idx_operation_status ON t_operation_log(operation_status);

-- 操作时间索引
CREATE INDEX idx_operation_time ON t_operation_log(operation_time);

-- 创建时间索引
CREATE INDEX idx_create_time ON t_operation_log(create_time);

-- IP地址索引
CREATE INDEX idx_ip_address ON t_operation_log(ip_address);
```

### 3.3 复合索引
```sql
-- 用户和操作类型复合索引
CREATE INDEX idx_user_operation_type ON t_operation_log(user_id, operation_type);

-- 模块和操作类型复合索引
CREATE INDEX idx_module_operation_type ON t_operation_log(operation_module, operation_type);

-- 状态和时间复合索引
CREATE INDEX idx_status_time ON t_operation_log(operation_status, operation_time);

-- 用户和时间复合索引
CREATE INDEX idx_user_time ON t_operation_log(user_id, operation_time);
```

## 4. 约束条件

### 4.1 主键约束
- **log_id**：主键，唯一且非空

### 4.2 外键约束
```sql
-- 关联用户表
ALTER TABLE t_operation_log 
ADD CONSTRAINT fk_log_user_id 
FOREIGN KEY (user_id) REFERENCES t_user_info(user_id);
```

### 4.3 检查约束
```sql
-- 操作类型约束
ALTER TABLE t_operation_log 
ADD CONSTRAINT chk_operation_type 
CHECK (operation_type IN ('login', 'logout', 'create', 'update', 'delete', 'query', 'export', 'import'));

-- 操作状态约束
ALTER TABLE t_operation_log 
ADD CONSTRAINT chk_operation_status 
CHECK (operation_status IN ('success', 'failed', 'error'));

-- 请求方法约束
ALTER TABLE t_operation_log 
ADD CONSTRAINT chk_request_method 
CHECK (request_method IS NULL OR request_method IN ('GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'));

-- 操作耗时约束
ALTER TABLE t_operation_log 
ADD CONSTRAINT chk_duration 
CHECK (duration IS NULL OR duration >= 0);
```

## 5. 示例数据

### 5.1 测试数据
```sql
INSERT INTO t_operation_log (
    log_id, user_id, operation_type, operation_module, operation_action,
    operation_description, request_url, request_method, request_params,
    response_data, operation_status, error_message, ip_address, user_agent,
    operation_time, duration, create_time
) VALUES 
(
    'LOG_20241220_001', 'USER_20241220_001', 'login', 'auth', 'user_login',
    '用户登录', '/api/auth/login', 'POST', '{"username": "zhangsan", "password": "***"}',
    '{"code": 200, "message": "登录成功", "data": {"token": "***"}}', 'success', NULL,
    '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    '2024-12-20 09:00:00', 150, '2024-12-20 09:00:00'
),
(
    'LOG_20241220_002', 'USER_20241220_001', 'create', 'order', 'create_reservation',
    '创建预约订单', '/api/reservation/create', 'POST', '{"product_id": "PROD_001", "customer_info": "***"}',
    '{"code": 200, "message": "创建成功", "data": {"order_id": "RES_001"}}', 'success', NULL,
    '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    '2024-12-20 09:30:00', 300, '2024-12-20 09:30:00'
),
(
    'LOG_20241220_003', 'USER_20241220_002', 'update', 'product', 'update_product_price',
    '更新产品价格', '/api/product/update', 'PUT', '{"product_id": "PROD_001", "price": 50}',
    NULL, 'failed', '产品不存在', '192.168.1.101', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    '2024-12-20 10:00:00', 50, '2024-12-20 10:00:00'
);
```

### 5.2 查询示例
```sql
-- 查询用户的操作日志
SELECT 
    log_id, operation_type, operation_module, operation_action,
    operation_status, operation_time, duration
FROM t_operation_log 
WHERE user_id = 'USER_20241220_001' 
ORDER BY operation_time DESC 
LIMIT 10;

-- 查询指定类型的操作日志
SELECT 
    log_id, user_id, operation_action, operation_status, operation_time
FROM t_operation_log 
WHERE operation_type = 'login' 
ORDER BY operation_time DESC;

-- 查询失败的操作日志
SELECT 
    log_id, user_id, operation_type, operation_action, error_message, operation_time
FROM t_operation_log 
WHERE operation_status = 'failed' 
ORDER BY operation_time DESC;

-- 查询指定时间范围的操作日志
SELECT 
    log_id, user_id, operation_type, operation_module, operation_action,
    operation_status, operation_time, duration
FROM t_operation_log 
WHERE operation_time BETWEEN '2024-12-20 00:00:00' AND '2024-12-20 23:59:59'
ORDER BY operation_time DESC;

-- 查询操作日志详细信息
SELECT 
    l.log_id, l.user_id, l.operation_type, l.operation_module, l.operation_action,
    l.operation_description, l.operation_status, l.error_message, l.operation_time, l.duration,
    u.username, u.real_name
FROM t_operation_log l
LEFT JOIN t_user_info u ON l.user_id = u.user_id
WHERE l.log_id = 'LOG_20241220_001';
```

## 6. 数据关系

### 6.1 关联表
- **t_user_info**：通过user_id关联用户表

### 6.2 关系说明
- 日志记录可以关联用户信息
- 用户删除时需要考虑日志记录的保留

## 7. 业务规则

### 7.1 数据规则
- 日志ID必须唯一
- 操作类型必须符合预定义值
- 操作状态必须符合预定义值
- 操作耗时必须大于等于0

### 7.2 业务规则
- 所有重要操作都需要记录日志
- 敏感信息需要脱敏处理
- 日志记录不能修改
- 日志查询需要权限控制

## 8. 日志分类

### 8.1 按操作类型分类
- **登录日志**：用户登录、登出操作
- **业务日志**：业务操作记录
- **系统日志**：系统操作记录
- **错误日志**：错误和异常记录

### 8.2 按操作模块分类
- **用户模块**：用户管理相关操作
- **产品模块**：产品管理相关操作
- **订单模块**：订单管理相关操作
- **系统模块**：系统管理相关操作
- **认证模块**：认证相关操作

## 9. 性能优化

### 9.1 查询优化
- 使用操作时间索引进行时间范围查询
- 使用用户ID索引进行用户操作查询
- 使用操作类型索引进行类型筛选
- 使用复合索引进行多条件查询

### 9.2 存储优化
- 考虑日志数据的分区存储
- 定期归档历史日志数据
- 压缩存储历史日志
- 清理过期的日志数据

## 10. 数据维护

### 10.1 数据清理
- 定期清理过期的日志数据
- 清理无效的日志记录
- 清理重复的日志数据
- 清理过大的日志记录

### 10.2 数据备份
- 定期备份重要日志数据
- 备份日志变更历史
- 备份日志关联信息
- 备份日志分析结果

## 11. 安全考虑

### 11.1 数据安全
- 敏感信息需要脱敏处理
- 日志信息需要权限控制
- 日志访问需要审计记录
- 日志导出需要特殊权限

### 11.2 访问控制
- 日志查询需要登录验证
- 日志管理需要管理员权限
- 日志导出需要审核流程
- 日志删除需要特殊权限

## 12. 更新记录

### 版本1.0.0 (2024-12-20)
- 创建表结构初始版本
- 定义字段和索引
- 建立约束条件
- 完善示例数据和查询

## 13. 相关文档

- [数据库表工程](./数据库表工程.md)
- [用户信息表](./t_user_info.md)
- [系统配置表](./t_system_config.md)

## 14. 联系方式

如有问题或建议，请联系开发团队。
