# Common Utils 工具方法使用文档

## 概述
本文档详细说明了 `src/common/utils` 目录中所有工具方法的使用方法，用于指导智能体在工程中开发页面时正确使用各种通用工具函数。

## 目录结构
```
src/common/utils/
├── tools.js              # 核心工具方法
├── request.js            # 请求工具
├── rsa_util.js           # RSA加密工具
├── rsaUtils.js           # RSA工具类
├── dynamic-component.js  # 动态组件工具
├── base64.min.js         # Base64编码工具
├── jsencrypt.min.js      # JSEncrypt加密库
├── gio/                  # GIO统计工具
│   ├── gio-alip.js
│   ├── gio-minp.js
│   └── gio-uniapp.js
├── gmcc/                 # 移动云通信工具
│   └── [多个子模块]
└── sdk/                  # SDK工具
    ├── gio-uniapp.js
    ├── gioCompress.js
    └── sdc_loadmp9.7.24.1.js
```

## 1. 核心工具方法 (tools.js)

### 1.1 基础工具方法

#### 获取Token
```javascript
import { tools } from '@/common/utils/tools.js';

// 获取当前用户token
const token = tools.getToken();
console.log('当前token:', token);
```

#### 返回首页
```javascript
// 返回首页（带参数）
tools.backToHome('param1=value1&param2=value2');

// 返回首页（无参数）
tools.backToHome();
```

#### 生成UUID
```javascript
// 生成UUID
const uuid = tools.getUuid();
console.log('生成的UUID:', uuid);
// 输出: "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx"
```

#### 生成订单号
```javascript
// 生成订单号，格式：Y + 时间（YYYYMMDDHHMMSS）+ 六位随机数
const orderId = tools.genOrderId();
console.log('生成的订单号:', orderId);
// 输出: "Y20240115143025123456"
```

#### URL参数解析
```javascript
// 解析URL参数
const paramValue = tools.RequestParamParse('paramName', 'https://example.com?paramName=value');
console.log('参数值:', paramValue);
// 输出: "value"

// 使用当前页面URL
const currentParam = tools.RequestParamParse('id');
console.log('当前页面id参数:', currentParam);
```

#### 手机号验证
```javascript
// 验证手机号格式
const isValidPhone = tools.isPhoneNumber('13800138000');
console.log('手机号是否有效:', isValidPhone);
// 输出: true

const isInvalidPhone = tools.isPhoneNumber('123456');
console.log('手机号是否有效:', isInvalidPhone);
// 输出: false
```

### 1.2 请求头生成

#### 生成SMSP请求头
```javascript
// 生成SMSP请求头
const smspHeaders = tools.getSmspHeaders();
console.log('SMSP请求头:', smspHeaders);
// 输出: { idtype: '0', apptype: '2', id: '021119', ... }
```

#### 生成YX请求头
```javascript
// 生成YX请求头（不带token）
const yxHeaders = tools.getYxHeaders();
console.log('YX请求头:', yxHeaders);

// 生成YX请求头（带token）
const yxHeadersWithToken = tools.getYxHeaders(true);
console.log('带token的YX请求头:', yxHeadersWithToken);
```

### 1.3 登录相关方法

#### 请求App登录
```javascript
// 调起app登录
tools.requestAppLogin();
```

#### H5退出登录
```javascript
// H5退出登录回到首页
tools.outLoginH5();
```

#### 集团单点登录
```javascript
// 集团单点登录
tools.unionLogin()
  .then(() => {
    console.log('集团单点登录成功');
  })
  .catch(() => {
    console.log('集团单点登录失败');
  });
```

#### 获取认证码
```javascript
// 获取认证码
tools.getAuthCode(true) // true表示允许弹出原生登录框
  .then(() => {
    console.log('获取认证码成功');
  })
  .catch(() => {
    console.log('获取认证码失败');
  });
```

### 1.4 配置和初始化

#### 获取配置信息
```javascript
// 获取公共配置信息
tools.getConfigInfo(false) // false表示不跳过登录流程
  .then(() => {
    console.log('配置信息获取成功');
  })
  .catch(() => {
    console.log('配置信息获取失败');
  });
```

#### 初始化登录
```javascript
// 首次进入页面进行登录
tools.initLogin()
  .then(() => {
    console.log('登录初始化成功');
  })
  .catch(() => {
    console.log('登录初始化失败');
  });
```

### 1.5 页面相关方法

#### 获取当前页面信息
```javascript
// 获取当前页面路径和参数
const currentPageInfo = tools.getCurrentPages();
console.log('当前页面信息:', currentPageInfo);
// 输出: "/pages/index/index?id=123&name=test"
```

#### 页面浏览统计
```javascript
// 集团、省侧页面浏览码
tools.insetH5PageShow('产品详情页');
```

### 1.6 URL处理工具

#### 处理外部链接跳转
```javascript
// 处理外部链接跳转（添加jctg参数）
const externalUrl = tools.jumpExternal('https://example.com/page');
console.log('处理后的URL:', externalUrl);
```

#### 处理查询字符串
```javascript
// 处理查询字符串
const newUrl = tools.handleQueryString('https://example.com', 'param', 'value');
console.log('处理后的URL:', newUrl);
// 输出: "https://example.com?param=value"
```

#### 获取URL参数
```javascript
// 获取URL中指定参数的值
const paramValue = tools.getQueryString('id', 'https://example.com?id=123&name=test');
console.log('参数值:', paramValue);
// 输出: "123"
```

### 1.7 单点登录处理

#### 处理单点登录
```javascript
// 处理地址单点
tools.handleSinglePoint('https://example.com/page')
  .then(processedUrl => {
    console.log('处理后的单点URL:', processedUrl);
  })
  .catch(() => {
    console.log('单点处理失败');
  });
```

#### 获取单点Token
```javascript
// 根据sourceid获取token
tools.getSinglePointToken('017018')
  .then(token => {
    console.log('单点Token:', token);
  })
  .catch(() => {
    console.log('获取单点Token失败');
  });
```

### 1.8 渠道和分享处理

#### 添加渠道ID
```javascript
// 集团双十一插码拼参
const urlWithChannel = tools.addChannelId('https://example.com/page');
console.log('添加渠道ID后的URL:', urlWithChannel);
```

#### 跳转App URL
```javascript
// 集团跳转特殊处理
tools.jumpAppUrl('https://example.com/page');
```

### 1.9 数据加密和插码

#### 号码插码处理
```javascript
// 号码插码入参处理
const encryptedPhone = tools.processingCode('13800138000');
console.log('加密后的手机号:', encryptedPhone);
```

#### 设置用户ID
```javascript
// 号码插码
tools.setUserId('13800138000');
```

#### 初始化插码
```javascript
// 初始化插码
tools.initRecord({
  query: {
    channelId: 'CH001',
    yx: 'YX001'
  }
}, {
  merchantsId: 'MERCHANT001',
  merchantsName: '测试商户'
});
```

### 1.10 版本判断

#### 版本号比较
```javascript
// 判断版本号是否大于等于目标版本号
const isNewer = tools.versionJudgment('9.9.1', '9.9.0');
console.log('版本是否更新:', isNewer);
// 输出: true
```

## 2. 请求工具 (request.js)

### 2.1 普通请求
```javascript
import { request } from '@/common/utils/request.js';

// 发起普通请求
const makeRequest = async () => {
  try {
    const response = await request({
      url: '/api/data',
      method: 'POST',
      data: {
        param1: 'value1',
        param2: 'value2'
      },
      headers: {
        'Content-Type': 'application/json'
      }
    });
    console.log('请求成功:', response);
    return response;
  } catch (error) {
    console.error('请求失败:', error);
    throw error;
  }
};
```

### 2.2 加密请求
```javascript
import { encryptRequest } from '@/common/utils/request.js';

// 发起加密请求
const makeEncryptedRequest = async () => {
  try {
    const response = await encryptRequest({
      url: '/api/sensitive-data',
      method: 'POST',
      data: {
        sensitiveInfo: '机密数据'
      }
    });
    console.log('加密请求成功:', response);
    return response;
  } catch (error) {
    console.error('加密请求失败:', error);
    throw error;
  }
};
```

## 3. 加密工具 (rsa_util.js & rsaUtils.js)

### 3.1 RSA加密解密
```javascript
import { $encruption, $decryption } from '@/common/utils/rsa_util.js';

// RSA加密
const encryptedData = $encruption('要加密的数据', '公钥');
console.log('加密结果:', encryptedData);

// RSA解密
const decryptedData = $decryption('加密的数据', '私钥');
console.log('解密结果:', decryptedData);
```

### 3.2 RSA工具类
```javascript
import rsaUtils from '@/common/utils/rsaUtils.js';

// 获取随机数
const randomKey = rsaUtils.getRandom(16);
console.log('随机密钥:', randomKey);

// 使用公钥加密字符串
const encrypted = rsaUtils.encryptStrByKey('公钥', '要加密的数据');
console.log('加密结果:', encrypted);

// 使用私钥解密字符串
const decrypted = rsaUtils.decryptStrByKey('私钥', '加密的数据');
console.log('解密结果:', decrypted);

// HMAC-SHA256签名
const signature = rsaUtils.encryptHmacSHA256ToHex('密钥', '要签名的数据');
console.log('签名结果:', signature);

// AES加密
const aesEncrypted = rsaUtils.AESEncrypt('要加密的数据', 'AES密钥');
console.log('AES加密结果:', aesEncrypted);

// AES解密
const aesDecrypted = rsaUtils.AESDecrypt('加密的数据', 'AES密钥');
console.log('AES解密结果:', aesDecrypted);

// 处理加密参数
const encryptedParams = rsaUtils.handleEncryptParams('随机密钥', {
  param1: 'value1',
  param2: 'value2'
});
console.log('加密参数:', encryptedParams);
```

## 4. 动态组件工具 (dynamic-component.js)

### 4.1 地图组件使用
```javascript
import dynamicComponent from '@/common/utils/dynamic-component.js';

// 打开地图组件
const openMapSelector = async () => {
  try {
    const result = await dynamicComponent.openMap({
      props: {
        // 地图组件属性
        center: { lat: 39.9, lng: 116.4 },
        zoom: 10
      }
    });
    
    if (result) {
      console.log('选择的位置:', result);
      // 处理选择结果
    } else {
      console.log('用户取消了选择');
    }
  } catch (error) {
    console.error('地图组件打开失败:', error);
  }
};

// 手动关闭地图组件
const closeMapSelector = () => {
  dynamicComponent.closeMap();
};
```

## 5. GIO统计工具 (gio/)

### 5.1 GIO初始化
```javascript
// 在main.js中初始化GIO
import gio from '@/common/utils/gio/gio-uniapp.js';

// 初始化GIO统计
gdp('init', '项目ID', '数据源ID', '应用ID', {
  debug: true,
  dataCollect: true,
  autotrack: true
});
```

### 5.2 用户标识
```javascript
// 设置用户ID
gdp('setUserId', 'user123');

// 清除用户ID
gdp('clearUserId');
```

### 5.3 事件追踪
```javascript
// 追踪自定义事件
gdp('track', 'button_click', {
  button_name: '提交按钮',
  page_name: '产品详情页'
});

// 设置用户属性
gdp('setUserAttributes', {
  user_level: 'VIP',
  user_type: '企业用户'
});
```

### 5.4 页面浏览
```javascript
// 页面浏览统计（自动触发，也可手动调用）
gdp('track', 'pageview', {
  page_name: '产品列表页',
  page_url: '/pages/product/list'
});
```

## 6. GMCC移动云通信工具 (gmcc/)

### 6.1 GMCC基础使用
```javascript
import { gmcc } from '@/common/utils/gmcc/index.js';

// 使用GMCC相关功能
// 具体API需要根据gmcc模块的具体实现来使用
```

## 7. SDK工具 (sdk/)

### 7.1 GIO压缩工具
```javascript
// 使用GIO压缩工具
// 具体API需要根据压缩工具的具体实现来使用
```

## 在组件中的使用方式

### 1. 基础工具使用
```javascript
// 在Vue组件中
import { tools } from '@/common/utils/tools.js';

export default {
  data() {
    return {
      userInfo: null
    };
  },
  mounted() {
    // 获取用户token
    const token = tools.getToken();
    if (token) {
      console.log('用户已登录');
    }
    
    // 验证手机号
    const phone = '13800138000';
    if (tools.isPhoneNumber(phone)) {
      console.log('手机号格式正确');
    }
  },
  methods: {
    // 生成订单号
    createOrder() {
      const orderId = tools.genOrderId();
      console.log('生成订单号:', orderId);
    },
    
    // 返回首页
    goHome() {
      tools.backToHome();
    }
  }
};
```

### 2. 请求工具使用
```javascript
// 在Vue组件中
import { request, encryptRequest } from '@/common/utils/request.js';

export default {
  methods: {
    // 普通请求
    async fetchData() {
      try {
        const response = await request({
          url: '/api/data',
          method: 'GET'
        });
        this.data = response.data;
      } catch (error) {
        console.error('请求失败:', error);
      }
    },
    
    // 加密请求
    async submitSensitiveData() {
      try {
        const response = await encryptRequest({
          url: '/api/sensitive',
          method: 'POST',
          data: this.formData
        });
        console.log('提交成功:', response);
      } catch (error) {
        console.error('提交失败:', error);
      }
    }
  }
};
```

### 3. 加密工具使用
```javascript
// 在Vue组件中
import rsaUtils from '@/common/utils/rsaUtils.js';

export default {
  methods: {
    // 加密敏感数据
    encryptData(data) {
      const randomKey = rsaUtils.getRandom(16);
      const encrypted = rsaUtils.AESEncrypt(JSON.stringify(data), randomKey);
      return {
        data: encrypted,
        key: randomKey
      };
    },
    
    // 解密数据
    decryptData(encryptedData, key) {
      const decrypted = rsaUtils.AESDecrypt(encryptedData, key);
      return JSON.parse(decrypted);
    }
  }
};
```

### 4. 动态组件使用
```javascript
// 在Vue组件中
import dynamicComponent from '@/common/utils/dynamic-component.js';

export default {
  methods: {
    // 打开地图选择器
    async openMap() {
      try {
        const result = await dynamicComponent.openMap({
          props: {
            center: this.currentLocation,
            zoom: 15
          }
        });
        
        if (result) {
          this.selectedLocation = result;
          console.log('选择的位置:', result);
        }
      } catch (error) {
        console.error('地图组件错误:', error);
      }
    }
  }
};
```

### 5. GIO统计使用
```javascript
// 在Vue组件中
export default {
  methods: {
    // 按钮点击统计
    onButtonClick(buttonName) {
      gdp('track', 'button_click', {
        button_name: buttonName,
        page_name: this.$route.name
      });
    },
    
    // 页面浏览统计
    onPageView() {
      gdp('track', 'pageview', {
        page_name: this.$route.name,
        page_url: this.$route.path
      });
    }
  }
};
```

## 使用规范和最佳实践

### 1. 工具方法调用规范
```javascript
// 正确的调用方式
import { tools } from '@/common/utils/tools.js';

// 错误：不要直接修改tools对象
// tools.someMethod = function() {} // ❌

// 正确：使用提供的API
const result = tools.someMethod(); // ✅
```

### 2. 错误处理规范
```javascript
// 所有工具方法都应该有适当的错误处理
try {
  const result = tools.someMethod();
  // 处理成功结果
} catch (error) {
  console.error('工具方法调用失败:', error);
  // 处理错误情况
}
```

### 3. 异步方法使用规范
```javascript
// 异步方法应该使用Promise或async/await
async function handleAsyncOperation() {
  try {
    const result = await tools.asyncMethod();
    return result;
  } catch (error) {
    console.error('异步操作失败:', error);
    throw error;
  }
}
```

### 4. 加密安全规范
```javascript
// 敏感数据必须加密
const sensitiveData = {
  phone: '13800138000',
  idCard: '123456789012345678'
};

// 使用AES加密
const encrypted = rsaUtils.AESEncrypt(JSON.stringify(sensitiveData), randomKey);

// 传输时使用加密请求
await encryptRequest({
  url: '/api/sensitive',
  data: { encryptedData: encrypted }
});
```

## 注意事项

1. **平台兼容性**: 部分工具方法在不同平台（H5、小程序、App）表现可能不同
2. **加密安全**: 敏感数据必须使用加密工具进行处理
3. **错误处理**: 所有工具方法调用都应该有适当的错误处理
4. **性能考虑**: 避免频繁调用计算密集型工具方法
5. **版本兼容**: 注意工具方法的版本兼容性
6. **内存管理**: 及时清理不需要的组件实例
7. **调试模式**: 开发环境可以开启调试模式查看详细信息

## 常用开发模式

### 1. 页面初始化模式
```javascript
export default {
  async mounted() {
    try {
      // 获取配置信息
      await tools.getConfigInfo();
      
      // 检查登录状态
      const token = tools.getToken();
      if (!token) {
        // 执行登录流程
        await tools.initLogin();
      }
      
      // 页面浏览统计
      tools.insetH5PageShow(this.$options.name);
      
      // 加载页面数据
      await this.loadPageData();
    } catch (error) {
      console.error('页面初始化失败:', error);
    }
  }
};
```

### 2. 表单提交模式
```javascript
export default {
  methods: {
    async submitForm() {
      try {
        // 表单验证
        if (!this.validateForm()) {
          return;
        }
        
        // 加密敏感数据
        const encryptedData = this.encryptFormData();
        
        // 提交数据
        const response = await encryptRequest({
          url: '/api/submit',
          method: 'POST',
          data: encryptedData
        });
        
        // 成功处理
        this.handleSuccess(response);
      } catch (error) {
        this.handleError(error);
      }
    },
    
    encryptFormData() {
      const randomKey = rsaUtils.getRandom(16);
      return {
        data: rsaUtils.AESEncrypt(JSON.stringify(this.formData), randomKey),
        key: randomKey
      };
    }
  }
};
```

### 3. 地图选择模式
```javascript
export default {
  methods: {
    async selectLocation() {
      try {
        const result = await dynamicComponent.openMap({
          props: {
            center: this.currentLocation,
            zoom: 15
          }
        });
        
        if (result) {
          this.selectedLocation = result;
          this.updateLocationInfo(result);
        }
      } catch (error) {
        console.error('地图选择失败:', error);
      }
    }
  }
};
```

这份文档提供了完整的common/utils工具方法使用指南，智能体可以根据此文档在开发页面时正确使用各种通用工具函数。
