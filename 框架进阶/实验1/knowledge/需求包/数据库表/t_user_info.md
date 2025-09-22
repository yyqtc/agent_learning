# 用户信息表 (t_user_info)

## 1. 表概述

用户信息表是智慧酒店包商品模板化需求的核心基础表，存储所有用户的基本信息，包括用户身份、联系方式、认证状态等，为用户管理、权限控制、业务关联等提供数据支持。

### 1.1 表定位
- **表类型**：核心基础表
- **主要用途**：存储用户基本信息
- **业务模块**：用户管理、权限管理、业务关联
- **数据特点**：基础数据，更新频率中等

### 1.2 表特点
- **数据完整**：包含用户的基本信息
- **结构清晰**：字段定义明确，数据类型合理
- **扩展性好**：支持用户信息的扩展和修改
- **关联紧密**：与其他业务表紧密关联

## 2. 表结构

### 2.1 字段定义

| 字段名 | 数据类型 | 长度 | 是否为空 | 默认值 | 说明 |
|--------|----------|------|----------|--------|------|
| user_id | VARCHAR | 50 | NOT NULL | - | 用户ID，主键 |
| username | VARCHAR | 100 | NOT NULL | - | 用户名 |
| nickname | VARCHAR | 100 | NULL | NULL | 昵称 |
| real_name | VARCHAR | 100 | NULL | NULL | 真实姓名 |
| phone | VARCHAR | 20 | NULL | NULL | 手机号码 |
| email | VARCHAR | 100 | NULL | NULL | 邮箱地址 |
| user_type | VARCHAR | 20 | NOT NULL | 'individual' | 用户类型 |
| user_status | VARCHAR | 20 | NOT NULL | 'active' | 用户状态 |
| avatar_url | VARCHAR | 500 | NULL | NULL | 头像URL |
| gender | VARCHAR | 10 | NULL | NULL | 性别 |
| birthday | DATE | - | NULL | NULL | 生日 |
| address | VARCHAR | 500 | NULL | NULL | 地址 |
| company_name | VARCHAR | 200 | NULL | NULL | 公司名称 |
| company_address | VARCHAR | 500 | NULL | NULL | 公司地址 |
| create_time | DATETIME | - | NOT NULL | CURRENT_TIMESTAMP | 创建时间 |
| update_time | DATETIME | - | NOT NULL | CURRENT_TIMESTAMP | 更新时间 |
| create_user | VARCHAR | 50 | NULL | NULL | 创建用户 |
| update_user | VARCHAR | 50 | NULL | NULL | 更新用户 |

### 2.2 字段详细说明

#### 2.2.1 主键字段
- **user_id**：用户唯一标识，格式为"USER_YYYYMMDD_序号"
- 示例：USER_20241220_001

#### 2.2.2 基本信息字段
- **username**：用户名，用于登录和显示
- **nickname**：昵称，用于显示
- **real_name**：真实姓名，用于业务处理
- **phone**：手机号码，用于联系和验证
- **email**：邮箱地址，用于联系和验证

#### 2.2.3 类型和状态字段
- **user_type**：用户类型，可选值：individual(个人)、enterprise(企业)、admin(管理员)
- **user_status**：用户状态，可选值：active(激活)、inactive(停用)、locked(锁定)、deleted(删除)

#### 2.2.4 扩展信息字段
- **avatar_url**：头像URL
- **gender**：性别，可选值：male(男)、female(女)、other(其他)
- **birthday**：生日
- **address**：地址
- **company_name**：公司名称
- **company_address**：公司地址

#### 2.2.5 系统字段
- **create_time**：记录创建时间
- **update_time**：记录最后更新时间
- **create_user**：创建用户ID
- **update_user**：最后更新用户ID

## 3. 索引设计

### 3.1 主键索引
```sql
PRIMARY KEY (user_id)
```

### 3.2 唯一索引
```sql
-- 用户名唯一索引
CREATE UNIQUE INDEX uk_username ON t_user_info(username);

-- 手机号码唯一索引
CREATE UNIQUE INDEX uk_phone ON t_user_info(phone);

-- 邮箱唯一索引
CREATE UNIQUE INDEX uk_email ON t_user_info(email);
```

### 3.3 普通索引
```sql
-- 用户类型索引
CREATE INDEX idx_user_type ON t_user_info(user_type);

-- 用户状态索引
CREATE INDEX idx_user_status ON t_user_info(user_status);

-- 创建时间索引
CREATE INDEX idx_create_time ON t_user_info(create_time);

-- 更新时间索引
CREATE INDEX idx_update_time ON t_user_info(update_time);
```

### 3.4 复合索引
```sql
-- 类型和状态复合索引
CREATE INDEX idx_type_status ON t_user_info(user_type, user_status);

-- 状态和时间复合索引
CREATE INDEX idx_status_time ON t_user_info(user_status, create_time);
```

## 4. 约束条件

### 4.1 主键约束
- **user_id**：主键，唯一且非空

### 4.2 唯一约束
- **username**：用户名唯一
- **phone**：手机号码唯一
- **email**：邮箱地址唯一

### 4.3 检查约束
```sql
-- 用户类型约束
ALTER TABLE t_user_info 
ADD CONSTRAINT chk_user_type 
CHECK (user_type IN ('individual', 'enterprise', 'admin'));

-- 用户状态约束
ALTER TABLE t_user_info 
ADD CONSTRAINT chk_user_status 
CHECK (user_status IN ('active', 'inactive', 'locked', 'deleted'));

-- 性别约束
ALTER TABLE t_user_info 
ADD CONSTRAINT chk_gender 
CHECK (gender IN ('male', 'female', 'other'));

-- 手机号码格式约束
ALTER TABLE t_user_info 
ADD CONSTRAINT chk_phone_format 
CHECK (phone IS NULL OR phone REGEXP '^1[3-9][0-9]{9}$');

-- 邮箱格式约束
ALTER TABLE t_user_info 
ADD CONSTRAINT chk_email_format 
CHECK (email IS NULL OR email REGEXP '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$');
```

## 5. 示例数据

### 5.1 测试数据
```sql
INSERT INTO t_user_info (
    user_id, username, nickname, real_name, phone, email,
    user_type, user_status, avatar_url, gender, birthday, address,
    company_name, company_address, create_time, update_time, create_user, update_user
) VALUES 
(
    'USER_20241220_001', 'zhangsan', '张三', '张三', '13800138000', 'zhangsan@example.com',
    'individual', 'active', 'https://example.com/avatars/zhangsan.jpg', 'male', '1990-01-01', '北京市东城区某某街道',
    NULL, NULL, '2024-12-20 10:00:00', '2024-12-20 10:00:00', 'admin', 'admin'
),
(
    'USER_20241220_002', 'lisi', '李四', '李四', '13900139000', 'lisi@example.com',
    'enterprise', 'active', 'https://example.com/avatars/lisi.jpg', 'female', '1985-05-15', '上海市浦东新区某某路',
    '测试公司', '上海市浦东新区测试大厦', '2024-12-20 10:00:00', '2024-12-20 10:00:00', 'admin', 'admin'
);
```

### 5.2 查询示例
```sql
-- 查询所有激活状态的用户
SELECT 
    user_id, username, nickname, real_name, phone, email, user_type, create_time
FROM t_user_info 
WHERE user_status = 'active' 
ORDER BY create_time DESC;

-- 查询指定类型的用户
SELECT 
    user_id, username, real_name, phone, company_name
FROM t_user_info 
WHERE user_type = 'enterprise' 
AND user_status = 'active';

-- 根据手机号码查询用户
SELECT 
    user_id, username, real_name, email, user_type, user_status
FROM t_user_info 
WHERE phone = '13800138000';

-- 查询用户详细信息
SELECT 
    user_id, username, nickname, real_name, phone, email,
    user_type, user_status, avatar_url, gender, birthday, address,
    company_name, company_address, create_time, update_time
FROM t_user_info 
WHERE user_id = 'USER_20241220_001';
```

## 6. 数据关系

### 6.1 关联表
- **t_user_auth**：通过user_id关联用户认证表
- **t_gr_reservation_order**：通过user_id关联预约订单表
- **t_enterprise_order**：通过user_id关联企业订单表

### 6.2 关系说明
- 一个用户对应一个认证记录
- 一个用户可以有多个预约订单
- 一个用户可以有多个企业订单

## 7. 业务规则

### 7.1 数据规则
- 用户ID必须唯一
- 用户名不能重复
- 手机号码不能重复
- 邮箱地址不能重复
- 用户类型和状态必须符合预定义值

### 7.2 业务规则
- 只有激活状态的用户才能进行业务操作
- 用户信息修改需要权限验证
- 用户删除需要检查是否有关联订单
- 企业用户必须填写公司信息

## 8. 数据安全

### 8.1 敏感信息保护
- 手机号码需要脱敏显示
- 邮箱地址需要脱敏显示
- 身份证信息需要加密存储
- 地址信息需要权限控制

### 8.2 访问控制
- 用户信息查询需要登录验证
- 用户信息修改需要身份验证
- 用户信息删除需要管理员权限
- 敏感信息访问需要特殊权限

## 9. 性能优化

### 9.1 查询优化
- 使用用户名索引进行登录验证
- 使用手机号码索引进行手机验证
- 使用邮箱索引进行邮箱验证
- 使用复合索引进行多条件查询

### 9.2 存储优化
- 头像URL字段长度适中
- 地址字段使用合适的长度
- 考虑用户数据的分区存储
- 定期清理无效用户数据

## 10. 数据维护

### 10.1 数据清理
- 定期清理已删除的用户
- 清理无效的头像URL
- 清理过期的用户信息
- 清理重复的用户数据

### 10.2 数据备份
- 定期备份用户数据
- 备份用户信息变更历史
- 备份用户关联信息
- 备份用户认证信息

## 11. 更新记录

### 版本1.0.0 (2024-12-20)
- 创建表结构初始版本
- 定义字段和索引
- 建立约束条件
- 完善示例数据和查询

## 12. 相关文档

- [数据库表工程](./数据库表工程.md)
- [用户认证表](./t_user_auth.md)
- [个人预约订单表](./t_gr_reservation_order.md)
- [企业订购订单表](./t_enterprise_order.md)

## 13. 联系方式

如有问题或建议，请联系开发团队。
