# 省级产品拓展信息表 (t_iqimall_product_ext_attr)

## 1. 表概述

省级产品拓展信息表是智慧酒店包商品模板化需求的产品扩展属性表，存储产品的详细配置信息，包括套餐类型、价格配置、带宽选项、SKU配置等扩展属性，为产品规格选择、价格计算、库存管理等业务功能提供数据支持。

### 1.1 表定位
- **表类型**：扩展属性表
- **主要用途**：存储产品扩展配置信息
- **业务模块**：产品管理、规格管理、价格管理
- **数据特点**：配置型数据，更新频率中等

### 1.2 表特点
- **配置灵活**：支持多种产品规格配置
- **扩展性强**：支持产品属性的动态扩展
- **关联紧密**：与主产品表紧密关联
- **业务复杂**：包含复杂的业务逻辑配置

## 2. 表结构

### 2.1 字段定义

| 字段名 | 数据类型 | 长度 | 是否为空 | 默认值 | 说明 |
|--------|----------|------|----------|--------|------|
| product_id | VARCHAR | 50 | NOT NULL | - | 产品ID，主键 |
| package_types | JSON | - | NULL | NULL | 套餐类型配置 |
| price_config | JSON | - | NULL | NULL | 价格配置信息 |
| bandwidth_options | JSON | - | NULL | NULL | 带宽选项配置 |
| sku_config | JSON | - | NULL | NULL | SKU配置信息 |
| tv_config | JSON | - | NULL | NULL | 电视配置信息 |
| screen_config | JSON | - | NULL | NULL | 屏幕配置信息 |
| create_time | DATETIME | - | NOT NULL | CURRENT_TIMESTAMP | 创建时间 |
| update_time | DATETIME | - | NOT NULL | CURRENT_TIMESTAMP | 更新时间 |
| create_user | VARCHAR | 50 | NULL | NULL | 创建用户 |
| update_user | VARCHAR | 50 | NULL | NULL | 更新用户 |

### 2.2 字段详细说明

#### 2.2.1 主键字段
- **product_id**：产品唯一标识，与主表关联

#### 2.2.2 配置字段
- **package_types**：套餐类型配置，JSON格式
- **price_config**：价格配置信息，JSON格式
- **bandwidth_options**：带宽选项配置，JSON格式
- **sku_config**：SKU配置信息，JSON格式
- **tv_config**：电视配置信息，JSON格式
- **screen_config**：屏幕配置信息，JSON格式

#### 2.2.3 系统字段
- **create_time**：记录创建时间
- **update_time**：记录最后更新时间
- **create_user**：创建用户ID
- **update_user**：最后更新用户ID

## 3. JSON字段结构

### 3.1 package_types 结构
```json
{
  "types": ["month", "year"],
  "default": "month",
  "description": {
    "month": "月套餐",
    "year": "年套餐"
  }
}
```

### 3.2 price_config 结构
```json
{
  "month": [40, 50, 80, 100, 120],
  "year": [400, 500, 800, 1000, 1200],
  "currency": "CNY",
  "discount_rules": {
    "year_discount": 0.1
  }
}
```

### 3.3 bandwidth_options 结构
```json
{
  "options": ["100M", "200M", "300M", "600M", "1000M"],
  "default": "100M",
  "description": {
    "100M": "100Mbps",
    "200M": "200Mbps",
    "300M": "300Mbps",
    "600M": "600Mbps",
    "1000M": "1000Mbps"
  }
}
```

### 3.4 sku_config 结构
```json
{
  "month": {
    "40": ["100M"],
    "50": ["200M"],
    "80": ["300M"],
    "100": ["600M"],
    "120": ["1000M"]
  },
  "year": {
    "400": ["100M"],
    "500": ["200M"],
    "800": ["300M"],
    "1000": ["600M"],
    "1200": ["1000M"]
  }
}
```

### 3.5 tv_config 结构
```json
{
  "options": ["商务电视IPTV", "标准电视IPTV", "高清电视IPTV"],
  "default": "商务电视IPTV",
  "description": {
    "商务电视IPTV": "商务级电视服务",
    "标准电视IPTV": "标准电视服务",
    "高清电视IPTV": "高清电视服务"
  }
}
```

### 3.6 screen_config 结构
```json
{
  "options": ["屏幕定制标准版", "屏幕定制高级版", "屏幕定制专业版"],
  "default": "屏幕定制标准版",
  "description": {
    "屏幕定制标准版": "标准屏幕定制服务",
    "屏幕定制高级版": "高级屏幕定制服务",
    "屏幕定制专业版": "专业屏幕定制服务"
  }
}
```

## 4. 索引设计

### 4.1 主键索引
```sql
PRIMARY KEY (product_id)
```

### 4.2 普通索引
```sql
-- 创建时间索引
CREATE INDEX idx_create_time ON t_iqimall_product_ext_attr(create_time);

-- 更新时间索引
CREATE INDEX idx_update_time ON t_iqimall_product_ext_attr(update_time);
```

## 5. 约束条件

### 5.1 主键约束
- **product_id**：主键，唯一且非空

### 5.2 外键约束
```sql
-- 关联主产品表
ALTER TABLE t_iqimall_product_ext_attr 
ADD CONSTRAINT fk_product_ext_attr_product_id 
FOREIGN KEY (product_id) REFERENCES t_iqimall_product(product_id);
```

### 5.3 检查约束
```sql
-- JSON字段格式检查
ALTER TABLE t_iqimall_product_ext_attr 
ADD CONSTRAINT chk_package_types_json 
CHECK (JSON_VALID(package_types));

ALTER TABLE t_iqimall_product_ext_attr 
ADD CONSTRAINT chk_price_config_json 
CHECK (JSON_VALID(price_config));
```

## 6. 示例数据

### 6.1 测试数据
```sql
INSERT INTO t_iqimall_product_ext_attr (
    product_id, package_types, price_config, bandwidth_options,
    sku_config, tv_config, screen_config,
    create_time, update_time, create_user, update_user
) VALUES 
(
    'PROD_20241220_001',
    '{"types": ["month", "year"], "default": "month", "description": {"month": "月套餐", "year": "年套餐"}}',
    '{"month": [40, 50, 80, 100, 120], "year": [400, 500, 800, 1000, 1200], "currency": "CNY", "discount_rules": {"year_discount": 0.1}}',
    '{"options": ["100M", "200M", "300M", "600M", "1000M"], "default": "100M", "description": {"100M": "100Mbps", "200M": "200Mbps", "300M": "300Mbps", "600M": "600Mbps", "1000M": "1000Mbps"}}',
    '{"month": {"40": ["100M"], "50": ["200M"], "80": ["300M"], "100": ["600M"], "120": ["1000M"]}, "year": {"400": ["100M"], "500": ["200M"], "800": ["300M"], "1000": ["600M"], "1200": ["1000M"]}}',
    '{"options": ["商务电视IPTV", "标准电视IPTV", "高清电视IPTV"], "default": "商务电视IPTV", "description": {"商务电视IPTV": "商务级电视服务", "标准电视IPTV": "标准电视服务", "高清电视IPTV": "高清电视服务"}}',
    '{"options": ["屏幕定制标准版", "屏幕定制高级版", "屏幕定制专业版"], "default": "屏幕定制标准版", "description": {"屏幕定制标准版": "标准屏幕定制服务", "屏幕定制高级版": "高级屏幕定制服务", "屏幕定制专业版": "专业屏幕定制服务"}}',
    '2024-12-20 10:00:00', '2024-12-20 10:00:00',
    'admin', 'admin'
);
```

### 6.2 查询示例
```sql
-- 查询产品扩展属性
SELECT 
    p.product_name,
    p.product_price,
    ext.package_types,
    ext.price_config,
    ext.bandwidth_options
FROM t_iqimall_product p
LEFT JOIN t_iqimall_product_ext_attr ext ON p.product_id = ext.product_id
WHERE p.product_id = 'PROD_20241220_001';

-- 查询套餐类型配置
SELECT 
    product_id,
    JSON_EXTRACT(package_types, '$.types') as package_types,
    JSON_EXTRACT(package_types, '$.default') as default_package
FROM t_iqimall_product_ext_attr
WHERE product_id = 'PROD_20241220_001';

-- 查询价格配置
SELECT 
    product_id,
    JSON_EXTRACT(price_config, '$.month') as month_prices,
    JSON_EXTRACT(price_config, '$.year') as year_prices
FROM t_iqimall_product_ext_attr
WHERE product_id = 'PROD_20241220_001';
```

## 7. 数据关系

### 7.1 关联表
- **t_iqimall_product**：通过product_id关联主产品表

### 7.2 关系说明
- 一个产品对应一个扩展属性记录
- 扩展属性记录依赖主产品表存在

## 8. 业务规则

### 8.1 数据规则
- 产品ID必须存在对应的主产品记录
- JSON字段必须符合预定义的结构
- 配置信息必须完整且有效

### 8.2 业务规则
- 套餐类型配置必须包含默认值
- 价格配置必须与套餐类型对应
- SKU配置必须与价格和带宽对应

## 9. 性能优化

### 9.1 查询优化
- 使用JSON函数进行配置查询
- 缓存常用配置信息
- 优化JSON字段的查询性能

### 9.2 存储优化
- JSON字段使用合适的存储引擎
- 定期优化JSON字段结构
- 考虑JSON字段的索引优化

## 10. 数据维护

### 10.1 数据清理
- 定期清理无效的配置信息
- 清理过期的价格配置
- 清理无用的SKU配置

### 10.2 数据备份
- 定期备份配置数据
- 备份配置变更历史
- 备份配置模板

## 11. 安全考虑

### 11.1 数据安全
- 配置信息需要权限控制
- JSON字段需要格式验证
- 配置变更需要审核

### 11.2 访问控制
- 配置查询需要登录验证
- 配置管理需要管理员权限
- 配置修改需要审核流程

## 12. 更新记录

### 版本1.0.0 (2024-12-20)
- 创建表结构初始版本
- 定义JSON字段结构
- 建立约束条件
- 完善示例数据和查询

## 13. 相关文档

- [数据库表工程](./数据库表工程.md)
- [省级产品信息表](./t_iqimall_product.md)
- [个人预约订单表](./t_gr_reservation_order.md)
- [企业订购订单表](./t_enterprise_order.md)

## 14. 联系方式

如有问题或建议，请联系开发团队。
