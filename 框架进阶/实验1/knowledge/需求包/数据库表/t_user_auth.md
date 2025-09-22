# 用户认证表 (t_user_auth)

## 1. 表概述

用户认证表是智慧酒店包商品模板化需求的核心安全表，存储用户的认证信息，包括登录凭证、密码信息、认证状态、权限信息等，为用户登录、权限验证、安全控制等提供数据支持。

### 1.1 表定位
- **表类型**：核心安全表
- **主要用途**：存储用户认证信息
- **业务模块**：用户认证、权限管理、安全控制
- **数据特点**：安全数据，更新频率中等

### 1.2 表特点
- **安全可靠**：存储敏感的认证信息
- **结构清晰**：字段定义明确，数据类型合理
- **扩展性好**：支持多种认证方式
- **关联紧密**：与用户表紧密关联

## 2. 表结构

### 2.1 字段定义

| 字段名 | 数据类型 | 长度 | 是否为空 | 默认值 | 说明 |
|--------|----------|------|----------|--------|------|
| auth_id | VARCHAR | 50 | NOT NULL | - | 认证ID，主键 |
| user_id | VARCHAR | 50 | NOT NULL | - | 用户ID |
| auth_type | VARCHAR | 20 | NOT NULL | 'password' | 认证类型 |
| auth_value | VARCHAR | 500 | NOT NULL | - | 认证值 |
| salt | VARCHAR | 100 | NULL | NULL | 盐值 |
| auth_status | VARCHAR | 20 | NOT NULL | 'active' | 认证状态 |
| last_login_time | DATETIME | - | NULL | NULL | 最后登录时间 |
| last_login_ip | VARCHAR | 50 | NULL | NULL | 最后登录IP |
| login_count | INT | - | NOT NULL | 0 | 登录次数 |
| failed_count | INT | - | NOT NULL | 0 | 失败次数 |
| lock_time | DATETIME | - | NULL | NULL | 锁定时间 |
| unlock_time | DATETIME | - | NULL | NULL | 解锁时间 |
| create_time | DATETIME | - | NOT NULL | CURRENT_TIMESTAMP | 创建时间 |
| update_time | DATETIME | - | NOT NULL | CURRENT_TIMESTAMP | 更新时间 |
| create_user | VARCHAR | 50 | NULL | NULL | 创建用户 |
| update_user | VARCHAR | 50 | NULL | NULL | 更新用户 |

### 2.2 字段详细说明

#### 2.2.1 主键字段
- **auth_id**：认证唯一标识，格式为"AUTH_YYYYMMDD_序号"
- 示例：AUTH_20241220_001

#### 2.2.2 关联字段
- **user_id**：用户ID，关联用户表

#### 2.2.3 认证字段
- **auth_type**：认证类型，可选值：password(密码)、sms(短信)、email(邮箱)、oauth(第三方)
- **auth_value**：认证值，如密码哈希、验证码等
- **salt**：盐值，用于密码加密

#### 2.2.4 状态字段
- **auth_status**：认证状态，可选值：active(激活)、inactive(停用)、locked(锁定)、expired(过期)

#### 2.2.5 登录信息字段
- **last_login_time**：最后登录时间
- **last_login_ip**：最后登录IP地址
- **login_count**：登录次数
- **failed_count**：失败次数
- **lock_time**：锁定时间
- **unlock_time**：解锁时间

#### 2.2.6 系统字段
- **create_time**：记录创建时间
- **update_time**：记录最后更新时间
- **create_user**：创建用户ID
- **update_user**：最后更新用户ID

## 3. 索引设计

### 3.1 主键索引
```sql
PRIMARY KEY (auth_id)
```

### 3.2 唯一索引
```sql
-- 用户ID和认证类型唯一索引
CREATE UNIQUE INDEX uk_user_auth_type ON t_user_auth(user_id, auth_type);
```

### 3.3 普通索引
```sql
-- 用户ID索引
CREATE INDEX idx_user_id ON t_user_auth(user_id);

-- 认证类型索引
CREATE INDEX idx_auth_type ON t_user_auth(auth_type);

-- 认证状态索引
CREATE INDEX idx_auth_status ON t_user_auth(auth_status);

-- 最后登录时间索引
CREATE INDEX idx_last_login_time ON t_user_auth(last_login_time);

-- 创建时间索引
CREATE INDEX idx_create_time ON t_user_auth(create_time);

-- 更新时间索引
CREATE INDEX idx_update_time ON t_user_auth(update_time);
```

### 3.4 复合索引
```sql
-- 用户和状态复合索引
CREATE INDEX idx_user_status ON t_user_auth(user_id, auth_status);

-- 状态和时间复合索引
CREATE INDEX idx_status_time ON t_user_auth(auth_status, last_login_time);
```

## 4. 约束条件

### 4.1 主键约束
- **auth_id**：主键，唯一且非空

### 4.2 唯一约束
- **user_id + auth_type**：用户ID和认证类型组合唯一

### 4.3 外键约束
```sql
-- 关联用户表
ALTER TABLE t_user_auth 
ADD CONSTRAINT fk_auth_user_id 
FOREIGN KEY (user_id) REFERENCES t_user_info(user_id);
```

### 4.4 检查约束
```sql
-- 认证类型约束
ALTER TABLE t_user_auth 
ADD CONSTRAINT chk_auth_type 
CHECK (auth_type IN ('password', 'sms', 'email', 'oauth'));

-- 认证状态约束
ALTER TABLE t_user_auth 
ADD CONSTRAINT chk_auth_status 
CHECK (auth_status IN ('active', 'inactive', 'locked', 'expired'));

-- 登录次数约束
ALTER TABLE t_user_auth 
ADD CONSTRAINT chk_login_count 
CHECK (login_count >= 0);

-- 失败次数约束
ALTER TABLE t_user_auth 
ADD CONSTRAINT chk_failed_count 
CHECK (failed_count >= 0);
```

## 5. 示例数据

### 5.1 测试数据
```sql
INSERT INTO t_user_auth (
    auth_id, user_id, auth_type, auth_value, salt, auth_status,
    last_login_time, last_login_ip, login_count, failed_count,
    lock_time, unlock_time, create_time, update_time, create_user, update_user
) VALUES 
(
    'AUTH_20241220_001', 'USER_20241220_001', 'password', 
    'e10adc3949ba59abbe56e057f20f883e', 'salt123', 'active',
    '2024-12-20 09:00:00', '192.168.1.100', 10, 0,
    NULL, NULL, '2024-12-20 10:00:00', '2024-12-20 10:00:00', 'admin', 'admin'
),
(
    'AUTH_20241220_002', 'USER_20241220_001', 'sms', 
    '123456', NULL, 'active',
    '2024-12-20 09:30:00', '192.168.1.100', 5, 1,
    NULL, NULL, '2024-12-20 10:00:00', '2024-12-20 10:00:00', 'admin', 'admin'
);
```

### 5.2 查询示例
```sql
-- 查询用户的认证信息
SELECT 
    auth_id, auth_type, auth_status, last_login_time, last_login_ip, login_count
FROM t_user_auth 
WHERE user_id = 'USER_20241220_001' 
AND auth_status = 'active';

-- 查询指定类型的认证信息
SELECT 
    auth_id, user_id, auth_status, last_login_time, login_count
FROM t_user_auth 
WHERE auth_type = 'password' 
AND auth_status = 'active';

-- 查询被锁定的认证信息
SELECT 
    auth_id, user_id, auth_type, lock_time, unlock_time, failed_count
FROM t_user_auth 
WHERE auth_status = 'locked';

-- 查询认证详细信息
SELECT 
    a.auth_id, a.user_id, a.auth_type, a.auth_status,
    a.last_login_time, a.last_login_ip, a.login_count, a.failed_count,
    u.username, u.real_name, u.phone
FROM t_user_auth a
LEFT JOIN t_user_info u ON a.user_id = u.user_id
WHERE a.auth_id = 'AUTH_20241220_001';
```

## 6. 数据关系

### 6.1 关联表
- **t_user_info**：通过user_id关联用户表

### 6.2 关系说明
- 一个用户可以有多种认证方式
- 认证记录依赖用户记录存在

## 7. 业务规则

### 7.1 数据规则
- 认证ID必须唯一
- 用户ID和认证类型组合必须唯一
- 认证状态必须符合预定义值
- 登录次数和失败次数必须大于等于0

### 7.2 业务规则
- 只有激活状态的认证才能使用
- 失败次数超过限制会自动锁定
- 锁定时间到期后自动解锁
- 认证信息修改需要权限验证

## 8. 安全考虑

### 8.1 数据安全
- 密码信息必须加密存储
- 认证值需要哈希处理
- 盐值需要随机生成
- 敏感信息需要权限控制

### 8.2 访问控制
- 认证信息查询需要管理员权限
- 认证信息修改需要特殊权限
- 认证信息删除需要审核流程
- 登录日志需要记录和监控

## 9. 性能优化

### 9.1 查询优化
- 使用用户ID索引进行用户认证查询
- 使用认证类型索引进行类型筛选
- 使用状态索引进行状态筛选
- 使用复合索引进行多条件查询

### 9.2 存储优化
- 认证值字段使用合适的长度
- 盐值字段使用固定长度
- 考虑认证数据的分区存储
- 定期清理过期的认证信息

## 10. 数据维护

### 10.1 数据清理
- 定期清理过期的认证信息
- 清理无效的认证记录
- 清理过期的锁定记录
- 清理重复的认证数据

### 10.2 数据备份
- 定期备份认证数据
- 备份认证信息变更历史
- 备份认证关联信息
- 备份认证日志信息

## 11. 更新记录

### 版本1.0.0 (2024-12-20)
- 创建表结构初始版本
- 定义字段和索引
- 建立约束条件
- 完善示例数据和查询

## 12. 相关文档

- [数据库表工程](./数据库表工程.md)
- [用户信息表](./t_user_info.md)
- [个人预约订单表](./t_gr_reservation_order.md)
- [企业订购订单表](./t_enterprise_order.md)

## 13. 联系方式

如有问题或建议，请联系开发团队。
