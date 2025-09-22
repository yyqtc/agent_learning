# 系统配置表 (t_system_config)

## 1. 表概述

系统配置表是智慧酒店包商品模板化需求的系统配置表，存储系统的各种配置信息，包括系统参数、业务配置、功能开关、环境配置等，为系统运行、业务控制、功能管理提供数据支持。

### 1.1 表定位
- **表类型**：系统配置表
- **主要用途**：存储系统配置信息
- **业务模块**：系统管理、配置管理、功能控制
- **数据特点**：配置数据，更新频率低

### 1.2 表特点
- **配置灵活**：支持多种配置类型
- **结构清晰**：字段定义明确，数据类型合理
- **扩展性好**：支持配置信息的动态扩展
- **管理方便**：支持配置的集中管理

## 2. 表结构

### 2.1 字段定义

| 字段名 | 数据类型 | 长度 | 是否为空 | 默认值 | 说明 |
|--------|----------|------|----------|--------|------|
| config_id | VARCHAR | 50 | NOT NULL | - | 配置ID，主键 |
| config_key | VARCHAR | 100 | NOT NULL | - | 配置键 |
| config_value | TEXT | - | NOT NULL | - | 配置值 |
| config_type | VARCHAR | 20 | NOT NULL | 'string' | 配置类型 |
| config_group | VARCHAR | 50 | NOT NULL | 'system' | 配置分组 |
| config_description | VARCHAR | 500 | NULL | NULL | 配置描述 |
| config_status | VARCHAR | 20 | NOT NULL | 'active' | 配置状态 |
| is_encrypted | TINYINT | 1 | NOT NULL | 0 | 是否加密 |
| is_readonly | TINYINT | 1 | NOT NULL | 0 | 是否只读 |
| sort_order | INT | - | NOT NULL | 0 | 排序顺序 |
| create_time | DATETIME | - | NOT NULL | CURRENT_TIMESTAMP | 创建时间 |
| update_time | DATETIME | - | NOT NULL | CURRENT_TIMESTAMP | 更新时间 |
| create_user | VARCHAR | 50 | NULL | NULL | 创建用户 |
| update_user | VARCHAR | 50 | NULL | NULL | 更新用户 |

### 2.2 字段详细说明

#### 2.2.1 主键字段
- **config_id**：配置唯一标识，格式为"CONFIG_YYYYMMDD_序号"
- 示例：CONFIG_20241220_001

#### 2.2.2 配置字段
- **config_key**：配置键，用于配置的唯一标识
- **config_value**：配置值，存储配置的具体内容
- **config_type**：配置类型，可选值：string(字符串)、number(数字)、boolean(布尔)、json(JSON)、array(数组)

#### 2.2.3 分组字段
- **config_group**：配置分组，用于配置的分类管理
- **config_description**：配置描述，说明配置的用途和含义

#### 2.2.4 状态字段
- **config_status**：配置状态，可选值：active(激活)、inactive(停用)、deprecated(废弃)

#### 2.2.5 属性字段
- **is_encrypted**：是否加密，0-否，1-是
- **is_readonly**：是否只读，0-否，1-是
- **sort_order**：排序顺序，用于配置的显示顺序

#### 2.2.6 系统字段
- **create_time**：记录创建时间
- **update_time**：记录最后更新时间
- **create_user**：创建用户ID
- **update_user**：最后更新用户ID

## 3. 索引设计

### 3.1 主键索引
```sql
PRIMARY KEY (config_id)
```

### 3.2 唯一索引
```sql
-- 配置键唯一索引
CREATE UNIQUE INDEX uk_config_key ON t_system_config(config_key);
```

### 3.3 普通索引
```sql
-- 配置分组索引
CREATE INDEX idx_config_group ON t_system_config(config_group);

-- 配置类型索引
CREATE INDEX idx_config_type ON t_system_config(config_type);

-- 配置状态索引
CREATE INDEX idx_config_status ON t_system_config(config_status);

-- 排序顺序索引
CREATE INDEX idx_sort_order ON t_system_config(sort_order);

-- 创建时间索引
CREATE INDEX idx_create_time ON t_system_config(create_time);

-- 更新时间索引
CREATE INDEX idx_update_time ON t_system_config(update_time);
```

### 3.4 复合索引
```sql
-- 分组和状态复合索引
CREATE INDEX idx_group_status ON t_system_config(config_group, config_status);

-- 分组和排序复合索引
CREATE INDEX idx_group_sort ON t_system_config(config_group, sort_order);
```

## 4. 约束条件

### 4.1 主键约束
- **config_id**：主键，唯一且非空

### 4.2 唯一约束
- **config_key**：配置键唯一

### 4.3 检查约束
```sql
-- 配置类型约束
ALTER TABLE t_system_config 
ADD CONSTRAINT chk_config_type 
CHECK (config_type IN ('string', 'number', 'boolean', 'json', 'array'));

-- 配置状态约束
ALTER TABLE t_system_config 
ADD CONSTRAINT chk_config_status 
CHECK (config_status IN ('active', 'inactive', 'deprecated'));

-- 加密标志约束
ALTER TABLE t_system_config 
ADD CONSTRAINT chk_is_encrypted 
CHECK (is_encrypted IN (0, 1));

-- 只读标志约束
ALTER TABLE t_system_config 
ADD CONSTRAINT chk_is_readonly 
CHECK (is_readonly IN (0, 1));

-- 排序顺序约束
ALTER TABLE t_system_config 
ADD CONSTRAINT chk_sort_order 
CHECK (sort_order >= 0);
```

## 5. 示例数据

### 5.1 测试数据
```sql
INSERT INTO t_system_config (
    config_id, config_key, config_value, config_type, config_group,
    config_description, config_status, is_encrypted, is_readonly, sort_order,
    create_time, update_time, create_user, update_user
) VALUES 
(
    'CONFIG_20241220_001', 'system.name', '智慧酒店包管理系统', 'string', 'system',
    '系统名称', 'active', 0, 1, 1,
    '2024-12-20 10:00:00', '2024-12-20 10:00:00', 'admin', 'admin'
),
(
    'CONFIG_20241220_002', 'system.version', '1.0.0', 'string', 'system',
    '系统版本', 'active', 0, 1, 2,
    '2024-12-20 10:00:00', '2024-12-20 10:00:00', 'admin', 'admin'
),
(
    'CONFIG_20241220_003', 'business.max_reservation_days', '30', 'number', 'business',
    '最大预约天数', 'active', 0, 0, 10,
    '2024-12-20 10:00:00', '2024-12-20 10:00:00', 'admin', 'admin'
),
(
    'CONFIG_20241220_004', 'business.enable_sms_notification', 'true', 'boolean', 'business',
    '启用短信通知', 'active', 0, 0, 20,
    '2024-12-20 10:00:00', '2024-12-20 10:00:00', 'admin', 'admin'
),
(
    'CONFIG_20241220_005', 'business.price_config', '{"month": [40, 50, 80, 100, 120], "year": [400, 500, 800, 1000, 1200]}', 'json', 'business',
    '价格配置', 'active', 0, 0, 30,
    '2024-12-20 10:00:00', '2024-12-20 10:00:00', 'admin', 'admin'
),
(
    'CONFIG_20241220_006', 'security.password_min_length', '8', 'number', 'security',
    '密码最小长度', 'active', 0, 0, 100,
    '2024-12-20 10:00:00', '2024-12-20 10:00:00', 'admin', 'admin'
),
(
    'CONFIG_20241220_007', 'security.max_login_attempts', '5', 'number', 'security',
    '最大登录尝试次数', 'active', 0, 0, 110,
    '2024-12-20 10:00:00', '2024-12-20 10:00:00', 'admin', 'admin'
),
(
    'CONFIG_20241220_008', 'api.rate_limit', '1000', 'number', 'api',
    'API速率限制', 'active', 0, 0, 200,
    '2024-12-20 10:00:00', '2024-12-20 10:00:00', 'admin', 'admin'
);
```

### 5.2 查询示例
```sql
-- 查询指定分组的配置
SELECT 
    config_key, config_value, config_type, config_description, sort_order
FROM t_system_config 
WHERE config_group = 'system' 
AND config_status = 'active' 
ORDER BY sort_order ASC;

-- 查询指定类型的配置
SELECT 
    config_key, config_value, config_group, config_description
FROM t_system_config 
WHERE config_type = 'json' 
AND config_status = 'active';

-- 查询可编辑的配置
SELECT 
    config_key, config_value, config_type, config_group, config_description
FROM t_system_config 
WHERE is_readonly = 0 
AND config_status = 'active' 
ORDER BY config_group, sort_order;

-- 查询配置详细信息
SELECT 
    config_id, config_key, config_value, config_type, config_group,
    config_description, config_status, is_encrypted, is_readonly, sort_order,
    create_time, update_time
FROM t_system_config 
WHERE config_key = 'system.name';
```

## 6. 数据关系

### 6.1 关联表
- 无直接关联表，作为系统基础配置表

### 6.2 关系说明
- 配置表为系统提供基础配置支持
- 其他业务表可以引用配置信息

## 7. 业务规则

### 7.1 数据规则
- 配置ID必须唯一
- 配置键必须唯一
- 配置类型必须符合预定义值
- 配置状态必须符合预定义值

### 7.2 业务规则
- 只有激活状态的配置才能使用
- 只读配置不能通过普通接口修改
- 加密配置需要特殊处理
- 配置修改需要权限验证

## 8. 配置分组

### 8.1 系统配置 (system)
- 系统名称、版本、环境等基础配置

### 8.2 业务配置 (business)
- 业务规则、流程配置、功能开关等

### 8.3 安全配置 (security)
- 安全策略、密码规则、访问控制等

### 8.4 API配置 (api)
- API限制、超时配置、重试策略等

### 8.5 通知配置 (notification)
- 通知方式、模板配置、发送策略等

## 9. 性能优化

### 9.1 查询优化
- 使用配置分组索引进行分组查询
- 使用配置类型索引进行类型筛选
- 使用状态索引进行状态筛选
- 使用复合索引进行多条件查询

### 9.2 存储优化
- 配置值字段使用TEXT类型，支持大文本
- 考虑配置数据的分区存储
- 定期清理废弃的配置
- 缓存常用配置信息

## 10. 数据维护

### 10.1 数据清理
- 定期清理废弃的配置
- 清理无效的配置值
- 清理重复的配置记录
- 清理过期的配置历史

### 10.2 数据备份
- 定期备份配置数据
- 备份配置变更历史
- 备份配置模板
- 备份配置关联信息

## 11. 安全考虑

### 11.1 数据安全
- 敏感配置需要加密存储
- 配置信息需要权限控制
- 配置修改需要审核流程
- 配置访问需要日志记录

### 11.2 访问控制
- 配置查询需要登录验证
- 配置管理需要管理员权限
- 配置修改需要特殊权限
- 配置删除需要审核流程

## 12. 更新记录

### 版本1.0.0 (2024-12-20)
- 创建表结构初始版本
- 定义字段和索引
- 建立约束条件
- 完善示例数据和查询

## 13. 相关文档

- [数据库表工程](./数据库表工程.md)
- [用户信息表](./t_user_info.md)
- [操作日志表](./t_operation_log.md)

## 14. 联系方式

如有问题或建议，请联系开发团队。
