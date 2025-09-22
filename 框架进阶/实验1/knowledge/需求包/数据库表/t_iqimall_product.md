# 省级产品信息表 (t_iqimall_product)

## 1. 表概述

省级产品信息表是智慧酒店包商品模板化需求的核心数据表，存储所有产品的基本信息，包括产品名称、价格、状态、描述等关键信息，为产品展示、规格选择、价格计算等业务功能提供数据支持。

### 1.1 表定位
- **表类型**：核心业务表
- **主要用途**：存储产品基本信息
- **业务模块**：产品管理、价格管理、订单管理
- **数据特点**：相对稳定，更新频率较低

### 1.2 表特点
- **数据完整**：包含产品的基本信息
- **结构清晰**：字段定义明确，数据类型合理
- **扩展性好**：支持产品信息的扩展和修改
- **性能优化**：合理的索引设计

## 2. 表结构

### 2.1 字段定义

| 字段名 | 数据类型 | 长度 | 是否为空 | 默认值 | 说明 |
|--------|----------|------|----------|--------|------|
| product_id | VARCHAR | 50 | NOT NULL | - | 产品ID，主键 |
| product_name | VARCHAR | 200 | NOT NULL | - | 产品名称 |
| product_price | DECIMAL | 10,2 | NOT NULL | 0.00 | 产品价格 |
| product_status | VARCHAR | 20 | NOT NULL | 'active' | 产品状态 |
| product_description | TEXT | - | NULL | NULL | 产品描述 |
| product_image | VARCHAR | 500 | NULL | NULL | 产品图片URL |
| product_category | VARCHAR | 50 | NULL | NULL | 产品分类 |
| create_time | DATETIME | - | NOT NULL | CURRENT_TIMESTAMP | 创建时间 |
| update_time | DATETIME | - | NOT NULL | CURRENT_TIMESTAMP | 更新时间 |
| create_user | VARCHAR | 50 | NULL | NULL | 创建用户 |
| update_user | VARCHAR | 50 | NULL | NULL | 更新用户 |

### 2.2 字段详细说明

#### 2.2.1 主键字段
- **product_id**：产品唯一标识，格式为"PROD_YYYYMMDD_序号"
- 示例：PROD_20241220_001

#### 2.2.2 基本信息字段
- **product_name**：产品名称，用于页面显示
- **product_price**：产品基础价格，用于价格计算
- **product_status**：产品状态，可选值：active(激活)、inactive(停用)、draft(草稿)

#### 2.2.3 描述信息字段
- **product_description**：产品详细描述，支持HTML格式
- **product_image**：产品主图片URL
- **product_category**：产品分类，用于产品分组

#### 2.2.4 系统字段
- **create_time**：记录创建时间
- **update_time**：记录最后更新时间
- **create_user**：创建用户ID
- **update_user**：最后更新用户ID

## 3. 索引设计

### 3.1 主键索引
```sql
PRIMARY KEY (product_id)
```

### 3.2 普通索引
```sql
-- 产品状态索引
CREATE INDEX idx_product_status ON t_iqimall_product(product_status);

-- 产品分类索引
CREATE INDEX idx_product_category ON t_iqimall_product(product_category);

-- 创建时间索引
CREATE INDEX idx_create_time ON t_iqimall_product(create_time);

-- 更新时间索引
CREATE INDEX idx_update_time ON t_iqimall_product(update_time);
```

### 3.3 复合索引
```sql
-- 状态和分类复合索引
CREATE INDEX idx_status_category ON t_iqimall_product(product_status, product_category);
```

## 4. 约束条件

### 4.1 主键约束
- **product_id**：主键，唯一且非空

### 4.2 唯一约束
- **product_name**：产品名称唯一

### 4.3 检查约束
```sql
-- 产品状态约束
ALTER TABLE t_iqimall_product 
ADD CONSTRAINT chk_product_status 
CHECK (product_status IN ('active', 'inactive', 'draft'));

-- 产品价格约束
ALTER TABLE t_iqimall_product 
ADD CONSTRAINT chk_product_price 
CHECK (product_price >= 0);
```

### 4.4 外键约束
- 无外键约束

## 5. 示例数据

### 5.1 测试数据
```sql
INSERT INTO t_iqimall_product (
    product_id, product_name, product_price, product_status, 
    product_description, product_image, product_category,
    create_time, update_time, create_user, update_user
) VALUES 
(
    'PROD_20241220_001', '智慧酒店包', 40.00, 'active',
    '智慧酒店包产品，提供高速网络和智能电视服务',
    'https://example.com/images/smart-hotel-package.jpg',
    'hotel_package',
    '2024-12-20 10:00:00', '2024-12-20 10:00:00',
    'admin', 'admin'
),
(
    'PROD_20241220_002', '企业专线包', 100.00, 'active',
    '企业专线包产品，提供专线网络服务',
    'https://example.com/images/enterprise-package.jpg',
    'enterprise_package',
    '2024-12-20 10:00:00', '2024-12-20 10:00:00',
    'admin', 'admin'
);
```

### 5.2 查询示例
```sql
-- 查询所有激活状态的产品
SELECT * FROM t_iqimall_product 
WHERE product_status = 'active' 
ORDER BY create_time DESC;

-- 查询指定分类的产品
SELECT product_id, product_name, product_price 
FROM t_iqimall_product 
WHERE product_category = 'hotel_package' 
AND product_status = 'active';

-- 查询产品价格范围
SELECT * FROM t_iqimall_product 
WHERE product_price BETWEEN 30 AND 100 
AND product_status = 'active';
```

## 6. 数据关系

### 6.1 关联表
- **t_iqimall_product_ext_attr**：通过product_id关联产品扩展属性
- **t_gr_reservation_order**：通过product_id关联预约订单
- **t_enterprise_order**：通过product_id关联企业订单

### 6.2 关系说明
- 一个产品可以对应多个扩展属性记录
- 一个产品可以对应多个预约订单
- 一个产品可以对应多个企业订单

## 7. 业务规则

### 7.1 数据规则
- 产品ID必须唯一
- 产品名称不能重复
- 产品价格必须大于等于0
- 产品状态只能是预定义的值

### 7.2 业务规则
- 只有激活状态的产品才能被用户选择
- 产品价格变更需要记录变更历史
- 产品删除需要检查是否有关联订单

## 8. 性能优化

### 8.1 查询优化
- 使用产品状态索引进行状态筛选
- 使用分类索引进行分类查询
- 使用复合索引进行多条件查询

### 8.2 存储优化
- 产品描述字段使用TEXT类型，支持大文本
- 图片URL字段长度适中，避免过长
- 时间字段使用DATETIME类型，支持精确时间

## 9. 数据维护

### 9.1 数据清理
- 定期清理草稿状态的产品
- 清理无效的产品图片URL
- 清理过期的产品描述

### 9.2 数据备份
- 定期备份产品数据
- 备份产品图片文件
- 备份产品配置信息

## 10. 安全考虑

### 10.1 数据安全
- 产品价格信息需要权限控制
- 产品描述内容需要XSS防护
- 产品图片URL需要验证

### 10.2 访问控制
- 产品信息查询需要登录验证
- 产品管理需要管理员权限
- 产品价格修改需要审核

## 11. 更新记录

### 版本1.0.0 (2024-12-20)
- 创建表结构初始版本
- 定义字段和索引
- 建立约束条件
- 完善示例数据和查询

## 12. 相关文档

- [数据库表工程](./数据库表工程.md)
- [省级产品拓展信息表](./t_iqimall_product_ext_attr.md)
- [个人预约订单表](./t_gr_reservation_order.md)
- [企业订购订单表](./t_enterprise_order.md)

## 13. 联系方式

如有问题或建议，请联系开发团队。
