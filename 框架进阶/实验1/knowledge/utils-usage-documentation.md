# Utils 工具函数使用文档

## 概述
本文档详细说明了 `src/utils` 目录中所有工具函数的使用方法，用于指导智能体在工程中开发页面时正确使用各种工具函数。

## 目录结构
```
src/utils/
├── index.js          # 基础工具函数
├── http.js           # HTTP请求封装
├── request.js        # 请求工具（带认证）
├── crypto.js         # 加密工具
├── myTools.js        # 自定义工具和存储
├── mixins.js         # Vue混入
├── imgUtil.js        # 图片处理工具
├── dictionary.js     # 项目配置字典
├── onlineData.js     # 在线数据模板
└── city_data.js      # 城市数据
```

## 1. 基础工具函数 (index.js)

### 1.1 时间格式化
```javascript
import { formatTime } from '@/utils/index.js';

// 格式化时间
const formattedTime = formatTime(new Date());
// 输出: "2024-01-15 14:30:25"

const formattedTime2 = formatTime('2024-01-15T14:30:25.000Z');
// 输出: "2024-01-15 14:30:25"
```

### 1.2 UUID生成
```javascript
import { generateUUID } from '@/utils/index.js';

// 生成UUID
const uuid = generateUUID();
// 输出: "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx"
```

### 1.3 页面URL获取
```javascript
import { getCurrentPageUrl } from '@/utils/index.js';

// 获取当前页面URL
const currentUrl = getCurrentPageUrl();
// H5环境: "https://example.com/pages/index/index"
// 小程序环境: "/pages/index/index"
```

### 1.4 客户端IP获取
```javascript
import { getClientIP } from '@/utils/index.js';

// 获取客户端IP（模拟）
const clientIP = getClientIP();
// 输出: "127.0.0.1"
```

### 1.5 移动设备检测
```javascript
import { isMobile } from '@/utils/index.js';

// 判断是否为移动设备
const isMobileDevice = isMobile();
// 输出: true/false
```

### 1.6 本地存储
```javascript
import { storage } from '@/utils/index.js';

// 存储数据
storage.set('userInfo', { name: '张三', age: 25 });

// 获取数据
const userInfo = storage.get('userInfo');
// 输出: { name: '张三', age: 25 }

// 删除数据
storage.remove('userInfo');

// 清空所有数据
storage.clear();
```

### 1.7 提示和加载
```javascript
import { showToast, showLoading, hideLoading } from '@/utils/index.js';

// 显示提示
showToast('操作成功', 'success');
showToast('操作失败', 'error');
showToast('提示信息', 'none');

// 显示加载
showLoading('加载中...');

// 隐藏加载
hideLoading();
```

## 2. HTTP请求工具 (http.js)

### 2.1 基础请求
```javascript
import { request } from '@/utils/http.js';

// GET请求
const getData = async () => {
  try {
    const response = await request({
      url: '/api/user/info',
      method: 'GET',
      data: { userId: 123 }
    });
    console.log(response);
  } catch (error) {
    console.error('请求失败:', error);
  }
};

// POST请求
const postData = async () => {
  try {
    const response = await request({
      url: '/api/user/update',
      method: 'POST',
      data: { name: '张三', age: 25 }
    });
    console.log(response);
  } catch (error) {
    console.error('请求失败:', error);
  }
};
```

### 2.2 加密请求
```javascript
import { encryptRequest } from '@/utils/http.js';

// 加密请求
const encryptedRequest = async () => {
  try {
    const response = await encryptRequest({
      url: '/api/sensitive/data',
      method: 'POST',
      data: { sensitiveInfo: '机密数据' },
      encrypt: true
    });
    console.log(response);
  } catch (error) {
    console.error('加密请求失败:', error);
  }
};
```

## 3. 认证请求工具 (request.js)

### 3.1 带认证的请求
```javascript
import request from '@/utils/request.js';

// 带认证头的请求
const authenticatedRequest = async () => {
  try {
    const response = await request({
      url: '/api/protected/data',
      method: 'GET',
      data: {},
      headers: {
        'Custom-Header': 'value'
      }
    });
    console.log(response);
  } catch (error) {
    console.error('认证请求失败:', error);
  }
};
```

## 4. 加密工具 (crypto.js)

### 4.1 生成Token
```javascript
import { generateTokenString } from '@/utils/crypto.js';

// 生成Token字符串
const token = generateTokenString();
// 输出: "25MgZWso67%@@yEq_1705123456789"
```

### 4.2 RSA加密
```javascript
import { encryptWithRsaPublicKey } from '@/utils/crypto.js';

// 使用RSA公钥加密
const encryptedData = encryptWithRsaPublicKey('要加密的数据');
// 输出: 加密后的字符串
```

## 5. 自定义工具和存储 (myTools.js)

### 5.1 数据存储
```javascript
import { myTools } from '@/utils/myTools.js';

// 存储产品信息
myTools.proInfo.set({
  productId: 'P001',
  productName: '企业宽带',
  price: 299
});

// 获取产品信息
const productInfo = myTools.proInfo.get();
// 输出: { productId: 'P001', productName: '企业宽带', price: 299 }

// 存储发票信息
myTools.invoiceApply.set({
  invoiceType: '1',
  invoiceTitle: '公司名称',
  taxNumber: '123456789012345678'
});

// 存储企业信息（本地存储）
myTools.groupInfo.set({
  companyName: '测试公司',
  region: '广东省',
  address: '深圳市南山区'
});
```

### 5.2 企业信息混入
```javascript
// 在Vue组件中使用
import { myTools } from '@/utils/myTools.js';

export default {
  mixins: [myTools.groupInfoMixins],
  methods: {
    async checkCompanyInfo() {
      // 检查企业信息是否完整
      await this.checkGroupInfo();
      console.log('企业信息:', this.groupInfo);
    }
  }
};
```

## 6. Vue混入 (mixins.js)

### 6.1 页面生命周期混入
```javascript
// 在Vue组件中使用
import { mixins } from '@/utils/mixins.js';

export default {
  mixins: [mixins],
  // 组件会自动获得onShow和onHide的生命周期处理
  // onShow: 绑定基础组件dom，处理H5页面显示
  // onHide: 处理页面隐藏逻辑
};
```

## 7. 图片处理工具 (imgUtil.js)

### 7.1 图片路径转Base64
```javascript
import { pathToBase64 } from '@/utils/imgUtil.js';

// 将图片路径转换为Base64
const convertImage = async () => {
  try {
    const base64 = await pathToBase64('/static/images/logo.png');
    console.log('Base64:', base64);
    // 输出: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
  } catch (error) {
    console.error('转换失败:', error);
  }
};
```

### 7.2 Base64转图片路径
```javascript
import { base64ToPath } from '@/utils/imgUtil.js';

// 将Base64转换为图片路径
const convertBase64 = async () => {
  try {
    const imagePath = await base64ToPath('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...');
    console.log('图片路径:', imagePath);
    // 输出: 本地图片路径
  } catch (error) {
    console.error('转换失败:', error);
  }
};
```

### 7.3 图片压缩
```javascript
import { compressBase64 } from '@/utils/imgUtil.js';

// 压缩Base64图片
const compressImage = async () => {
  try {
    const compressedBase64 = await compressBase64(
      'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...',
      0.8 // 压缩质量 0-1
    );
    console.log('压缩后:', compressedBase64);
  } catch (error) {
    console.error('压缩失败:', error);
  }
};
```

## 8. 项目配置字典 (dictionary.js)

### 8.1 项目配置
```javascript
import { PROJECT_CONFIG } from '@/utils/dictionary.js';

// 获取项目配置
const config = PROJECT_CONFIG;
console.log('项目标识:', config.ECOP_MINIKEY);
console.log('源ID:', config.SOURCE_ID);
console.log('请求白名单:', config.REQUEST_WHITE_LIST);
console.log('加密配置:', config.ENCRYPT_INFO);
```

### 8.2 配置使用示例
```javascript
import { PROJECT_CONFIG } from '@/utils/dictionary.js';

// 检查API是否在白名单中
const isInWhiteList = (url) => {
  return PROJECT_CONFIG.REQUEST_WHITE_LIST.includes(url);
};

// 使用示例
if (isInWhiteList('/common/uniteAuthentication/encryptSign')) {
  console.log('此API不需要加密');
} else {
  console.log('此API需要加密');
}
```

## 9. 在线数据模板 (onlineData.js)

### 9.1 订单数据模板
```javascript
import { onlineJSON } from '@/utils/onlineData.js';

// 创建订单数据
const createOrderData = (orderInfo) => {
  const orderData = {
    ...onlineJSON,
    orderId: generateUUID(), // 使用工具函数生成UUID
    orderType: orderInfo.orderType || "0",
    attachOrderNo: orderInfo.attachOrderNo || "",
    goodsId: orderInfo.goodsId,
    orderAttrIF: {
      ...onlineJSON.orderAttrIF,
      custProdGrowOperName: orderInfo.operatorName,
      custProdGrowOperId: orderInfo.operatorId,
      prdDealMan: orderInfo.dealerName,
      prdDealPhone: orderInfo.dealerPhone,
      prdDealCreditCode: orderInfo.idCard,
      contractRemark: orderInfo.remark
    },
    orderProdIF: {
      ...onlineJSON.orderProdIF,
      prodId: orderInfo.productId,
      buildingOid: orderInfo.buildingOid,
      orderCount: orderInfo.count,
      contractMonth: orderInfo.contractMonth,
      resourceAddress: orderInfo.address,
      mainServnumber: orderInfo.mainNumber,
      region: orderInfo.region
    }
  };
  
  return orderData;
};

// 使用示例
const orderInfo = {
  orderType: "1",
  goodsId: "G001",
  operatorName: "张三",
  operatorId: "OP001",
  dealerName: "李四",
  dealerPhone: "13800138000",
  idCard: "440301199001011234",
  productId: "P001",
  buildingOid: "B001",
  count: "1",
  contractMonth: "12",
  address: "深圳市南山区",
  mainNumber: "13800138000",
  region: "广东省"
};

const orderData = createOrderData(orderInfo);
console.log('订单数据:', orderData);
```

## 10. 城市数据 (city_data.js)

### 10.1 城市数据使用
```javascript
import cityData from '@/utils/city_data.js';

// 获取所有省份
const provinces = cityData.map(province => ({
  value: province.value,
  label: province.label
}));

// 根据省份获取城市
const getCitiesByProvince = (provinceValue) => {
  const province = cityData.find(p => p.value === provinceValue);
  return province ? province.children : [];
};

// 使用示例
const guangdongCities = getCitiesByProvince(440000);
console.log('广东省城市:', guangdongCities);
```

## 在组件中的使用方式

### 1. 基础工具函数使用
```javascript
// 在Vue组件中
import { formatTime, storage, showToast } from '@/utils/index.js';

export default {
  data() {
    return {
      userInfo: null
    };
  },
  mounted() {
    // 加载用户信息
    this.userInfo = storage.get('userInfo');
    
    // 显示欢迎信息
    if (this.userInfo) {
      showToast(`欢迎回来，${this.userInfo.name}`);
    }
  },
  methods: {
    saveUserInfo() {
      storage.set('userInfo', this.userInfo);
      showToast('保存成功');
    }
  }
};
```

### 2. HTTP请求使用
```javascript
// 在Vue组件中
import { request, encryptRequest } from '@/utils/http.js';

export default {
  methods: {
    async fetchData() {
      try {
        showLoading('加载中...');
        const response = await request({
          url: '/api/data',
          method: 'GET'
        });
        this.data = response;
      } catch (error) {
        showToast('加载失败', 'error');
      } finally {
        hideLoading();
      }
    },
    
    async submitSensitiveData() {
      try {
        const response = await encryptRequest({
          url: '/api/sensitive/submit',
          method: 'POST',
          data: this.formData,
          encrypt: true
        });
        showToast('提交成功');
      } catch (error) {
        showToast('提交失败', 'error');
      }
    }
  }
};
```

### 3. 图片处理使用
```javascript
// 在Vue组件中
import { pathToBase64, compressBase64 } from '@/utils/imgUtil.js';

export default {
  methods: {
    async handleImageUpload(file) {
      try {
        // 转换为Base64
        const base64 = await pathToBase64(file.path);
        
        // 压缩图片
        const compressedBase64 = await compressBase64(base64, 0.8);
        
        // 上传到服务器
        await this.uploadImage(compressedBase64);
      } catch (error) {
        showToast('图片处理失败', 'error');
      }
    }
  }
};
```

## 注意事项

1. **存储限制**：uni-app的存储有大小限制，建议合理使用存储空间
2. **加密安全**：敏感数据必须使用加密请求，确保数据安全
3. **图片处理**：大图片建议先压缩再上传，避免内存溢出
4. **错误处理**：所有异步操作都应该有适当的错误处理
5. **平台兼容**：部分工具函数在不同平台（H5、小程序、App）表现可能不同
6. **性能优化**：频繁的存储操作可能影响性能，建议合理使用
7. **数据格式**：存储和传输的数据格式要保持一致

## 常用开发模式

### 1. 页面数据初始化
```javascript
export default {
  async mounted() {
    // 检查企业信息
    await this.checkGroupInfo();
    
    // 加载用户数据
    this.userInfo = storage.get('userInfo');
    
    // 获取页面数据
    await this.fetchPageData();
  }
};
```

### 2. 表单数据提交
```javascript
export default {
  methods: {
    async submitForm() {
      try {
        showLoading('提交中...');
        
        // 验证表单
        if (!this.validateForm()) {
          return;
        }
        
        // 提交数据
        const response = await encryptRequest({
          url: '/api/form/submit',
          method: 'POST',
          data: this.formData,
          encrypt: true
        });
        
        showToast('提交成功');
        this.resetForm();
      } catch (error) {
        showToast('提交失败', 'error');
      } finally {
        hideLoading();
      }
    }
  }
};
```

### 3. 图片上传处理
```javascript
export default {
  methods: {
    async handleImageSelect() {
      try {
        // 选择图片
        const res = await uni.chooseImage({
          count: 1,
          sizeType: ['compressed'],
          sourceType: ['album', 'camera']
        });
        
        const filePath = res.tempFilePaths[0];
        
        // 转换为Base64
        const base64 = await pathToBase64(filePath);
        
        // 压缩图片
        const compressedBase64 = await compressBase64(base64, 0.8);
        
        // 上传
        await this.uploadImage(compressedBase64);
      } catch (error) {
        showToast('图片处理失败', 'error');
      }
    }
  }
};
```

这份文档提供了完整的utils工具函数使用指南，智能体可以根据此文档在开发页面时正确使用各种工具函数。
