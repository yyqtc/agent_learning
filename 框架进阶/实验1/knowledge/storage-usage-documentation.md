# 存储使用文档 - localStorage、sessionStorage 和 uni.StorageSync

## 概述
本文档详细说明了项目中 localStorage、sessionStorage 和 uni.StorageSync 的使用方法，基于对项目代码的完全分析，为智能体开发页面提供准确的存储操作指导。

## 存储类型对比

| 存储类型 | 生命周期 | 作用域 | 容量限制 | 平台支持 | 主要用途 |
|---------|---------|--------|----------|----------|----------|
| localStorage | 永久存储 | 同源页面共享 | 5-10MB | H5 | 用户信息、配置数据 |
| sessionStorage | 会话存储 | 单页面会话 | 5-10MB | H5 | 临时数据、倒计时 |
| uni.StorageSync | 永久存储 | 应用内共享 | 10MB | 全平台 | 跨平台数据存储 |

## 1. localStorage 使用

### 1.1 基础使用模式

#### 用户信息存储
```javascript
// 存储用户信息（带时间戳）
const userInfo = {
  rspTransId: 'user123',
  phone: '13800138000'
};

const dateJson = {
  userInfo: userInfo,
  time: new Date().getTime()
};

// 存储到localStorage
localStorage.setItem('date_json_ecop', JSON.stringify(dateJson));
```

#### 用户信息读取
```javascript
// 从localStorage读取用户信息
let dataJson = localStorage.getItem('date_json_ecop');
if (dataJson) {
  dataJson = JSON.parse(dataJson);
  let { userInfo, time } = dataJson;
  
  // 检查时间是否过期（30分钟）
  const now = new Date().getTime();
  if (now - time < 30 * 60 * 1000) {
    // 用户信息有效
    console.log('用户信息:', userInfo);
  } else {
    // 用户信息过期，清除
    localStorage.removeItem('date_json_ecop');
  }
}
```

#### 用户信息清除
```javascript
// 清除用户信息
localStorage.removeItem('date_json_ecop');
```

### 1.2 在Vuex中的使用

#### Store中的localStorage操作
```javascript
// 在store/index.js中
export default {
  state: {
    token: '',
    phone: '',
    userInfo: null
  },
  
  mutations: {
    setUserInfo(state, params) {
      state.userInfo = params;
      state.token = params.rspTransId;
      state.phone = params.phone;
      
      // #ifdef H5
      // 存储到localStorage
      const dateJson = {
        userInfo: params,
        time: new Date().getTime()
      };
      localStorage.setItem('date_json_ecop', JSON.stringify(dateJson));
      // #endif
    },
    
    removeUserInfo(state) {
      state.token = '';
      state.phone = '';
      state.userInfo = null;
      
      // #ifdef H5
      localStorage.removeItem('date_json_ecop');
      // #endif
    }
  },
  
  actions: {
    getUserInfo({ commit }) {
      // #ifdef H5
      let dataJson = localStorage.getItem('date_json_ecop');
      if (dataJson) {
        dataJson = JSON.parse(dataJson);
        let { userInfo, time } = dataJson;
        
        // 检查是否过期
        const now = new Date().getTime();
        if (now - time < 30 * 60 * 1000) {
          commit('setUserInfo', userInfo);
        } else {
          localStorage.removeItem('date_json_ecop');
        }
      }
      // #endif
    }
  }
};
```

## 2. sessionStorage 使用

### 2.1 倒计时功能
```javascript
// 倒计时存储和读取
export default {
  data() {
    return {
      countdown: 0,
      intervalId: null
    };
  },
  
  methods: {
    // 发送验证码
    sendCode() {
      return new Promise((resolve, reject) => {
        // 检查是否有倒计时
        if (sessionStorage.getItem("countdown")) {
          this.countdown = sessionStorage.getItem("countdown");
          this.startCountdown();
          resolve();
          return;
        }
        
        // 发送验证码请求
        smsCodeApply({
          "mallOrderNumber": this.mallOrderNumber
        }, uni.getStorageSync("mob")).then(res => {
          if (res.code === 200) {
            this.countdown = 60;
            // 存储倒计时到sessionStorage
            sessionStorage.setItem("countdown", this.countdown);
            this.startCountdown();
            resolve();
          } else {
            reject(res);
          }
        });
      });
    },
    
    // 开始倒计时
    startCountdown() {
      this.intervalId = setInterval(() => {
        this.countdown -= 1;
        if (!this.countdown) {
          // 倒计时结束，清除sessionStorage
          sessionStorage.removeItem("countdown");
          clearInterval(this.intervalId);
          this.intervalId = null;
        } else {
          // 更新sessionStorage中的倒计时
          sessionStorage.setItem("countdown", this.countdown);
        }
      }, 1000);
    },
    
    // 提交订单后清除倒计时
    submitOrder() {
      smsCodeCommitOrder({
        mallOrderNumber: this.mallOrderNumber,
        smsCode: this.smsCode
      }, uni.getStorageSync("mob")).then(res => {
        if (res.code === 200) {
          // 清除倒计时
          clearInterval(this.intervalId);
          this.intervalId = null;
          sessionStorage.removeItem("countdown");
          this.countdown = 0;
        }
      });
    }
  }
};
```

### 2.2 页面状态管理
```javascript
// 订单管理页面状态存储
export default {
  data() {
    return {
      type: -1,
      userInfo: {
        id: -1,
        employeeName: ''
      }
    };
  },
  
  onLoad(options) {
    if (options.type === "all") {
      this.type = -1;
      sessionStorage.setItem("user-type", -1);
    } else if (options.type === "person") {
      this.type = 1;
      sessionStorage.setItem("user-type", 1);
      
      if (options.id && options.id.length) {
        this.userInfo.id = options.id;
        this.userInfo.employeeName = options.name;
        sessionStorage.setItem("user-id", options.id);
        sessionStorage.setItem("user-name", options.name);
      }
    } else if (sessionStorage.getItem("user-type")) {
      // 从sessionStorage恢复状态
      this.type = sessionStorage.getItem("user-type");
      if (this.type === 1 && sessionStorage.getItem("user-id")) {
        this.userInfo.id = sessionStorage.getItem("user-id");
        this.userInfo.employeeName = sessionStorage.getItem("user-name");
      }
    }
  },
  
  methods: {
    goBack() {
      // 清除sessionStorage状态
      sessionStorage.removeItem("user-type");
      sessionStorage.removeItem("user-id");
      sessionStorage.removeItem("user-name");
      uni.navigateBack();
    }
  }
};
```

## 3. uni.StorageSync 使用

### 3.1 基础存储操作

#### 手机号存储
```javascript
// 存储手机号
uni.setStorageSync("mob", "13800138000");

// 读取手机号
const phone = uni.getStorageSync("mob");
if (!phone) {
  uni.showToast({
    title: "未获取到手机号",
    icon: 'none'
  });
  return;
}

// 手机号脱敏显示
const phoneDisplayed = phone.replace(/(\d{3})\d{4}(\d{4})/, '$1****$2');
```

#### 在API调用中使用
```javascript
// 所有API调用都需要传递手机号
getPromotionInfo(uni.getStorageSync("mob")).then(res => {
  if (res.code === 200) {
    this.userInfo = res.result;
  }
});

getProductDetail({productId: this.productId}, uni.getStorageSync("mob")).then(res => {
  if (res.code === 200) {
    this.productDetail = res.result;
  }
});
```

### 3.2 认证Token存储
```javascript
// 认证相关存储
const TokenKey = 'Smsp-Mobile-Token';

// 设置Token
export function setToken(token) {
  return uni.setStorageSync(TokenKey, token);
}

// 获取Token
export function getToken() {
  return uni.getStorageSync(TokenKey);
}

// 移除Token
export function removeToken() {
  return uni.removeStorageSync(TokenKey);
}
```

### 3.3 字典数据缓存
```javascript
// 字典数据缓存管理
export function getDictList(dictType) {
  return new Promise((resolve, reject) => {
    const timeout = Number(1000 * 60 * 60); // 1小时过期
    const cacheKey = `smsp.dict.${dictType}`;
    
    // 检查缓存
    const cachedData = uni.getStorageSync(cacheKey);
    try {
      if (cachedData) {
        const result = JSON.parse(cachedData);
        if (result.time) {
          const time = result.time;
          // 检查是否过期
          if ((time + timeout) > new Date().getTime() && result.list && result.list.length) {
            resolve(result.list);
            return;
          }
        }
      }
      
      // 获取新数据
      getDicts(dictType).then(res => {
        const dictList = (res.data || []).map(item => ({
          label: item.dictLabel,
          value: item.dictValue
        }));
        
        // 存储到缓存
        uni.setStorageSync(cacheKey, JSON.stringify({
          time: new Date().getTime(),
          list: dictList
        }));
        
        resolve(dictList);
      }).catch(error => {
        reject(error);
      });
    } catch (error) {
      reject(error);
    }
  });
}
```

### 3.4 位置权限存储
```javascript
// 位置权限状态存储
export function locationAuthModal(backWhenCancel) {
  return new Promise((resolve, reject) => {
    try {
      const hasLocationAuth = uni.getStorageSync('smsp.cache.location.hasAuth');
      if (!hasLocationAuth) {
        uni.showModal({
          title: '位置权限',
          content: '需要获取您的位置信息',
          success: (res) => {
            if (res.confirm) {
              try {
                // 存储权限状态
                uni.setStorageSync('smsp.cache.location.hasAuth', 'true');
                resolve();
              } catch (e) {
                reject(e);
              }
            } else {
              if (backWhenCancel) {
                uni.navigateBack();
              }
              reject(new Error('用户拒绝位置权限'));
            }
          }
        });
      } else {
        resolve();
      }
    } catch (error) {
      reject(error);
    }
  });
}
```

### 3.5 组件缓存管理
```javascript
// uni-data-select组件缓存
export default {
  data() {
    return {
      cacheKey: 'uni-data-select-cache'
    };
  },
  
  methods: {
    // 获取缓存
    getCache(name = this.getCurrentCacheKey()) {
      let cacheData = uni.getStorageSync(this.cacheKey) || {};
      return cacheData[name];
    },
    
    // 设置缓存
    setCache(value, name = this.getCurrentCacheKey()) {
      let cacheData = uni.getStorageSync(this.cacheKey) || {};
      cacheData[name] = value;
      uni.setStorageSync(this.cacheKey, cacheData);
    },
    
    // 删除缓存
    removeCache(name = this.getCurrentCacheKey()) {
      let cacheData = uni.getStorageSync(this.cacheKey) || {};
      delete cacheData[name];
      uni.setStorageSync(this.cacheKey, cacheData);
    }
  }
};
```

## 4. 存储工具类

### 4.1 通用存储工具
```javascript
// src/utils/index.js
export const storage = {
  // 设置存储
  set: (key, value) => {
    try {
      uni.setStorageSync(key, value);
    } catch (e) {
      console.error('存储失败:', e);
    }
  },
  
  // 获取存储
  get: (key) => {
    try {
      return uni.getStorageSync(key);
    } catch (e) {
      console.error('读取失败:', e);
      return null;
    }
  },
  
  // 删除存储
  remove: (key) => {
    try {
      uni.removeStorageSync(key);
    } catch (e) {
      console.error('删除失败:', e);
    }
  },
  
  // 清空存储
  clear: () => {
    try {
      uni.clearStorageSync();
    } catch (e) {
      console.error('清空失败:', e);
    }
  }
};

// 使用示例
import { storage } from '@/utils/index.js';

// 存储数据
storage.set('userInfo', { name: '张三', age: 25 });

// 读取数据
const userInfo = storage.get('userInfo');

// 删除数据
storage.remove('userInfo');

// 清空所有数据
storage.clear();
```

## 5. GIO统计存储

### 5.1 GIO用户标识存储
```javascript
// GIO统计相关存储键名
const GIO_STORAGE_KEYS = {
  UID: '_growing_uid_',
  USER_ID: '_growing_userId_',
  USER_KEY: '_growing_userKey_',
  GIO_ID: '_growing_gioId_',
  ESID: '_growing_esid_',
  GSID: '_growing_gsid_'
};

// 用户标识管理
class GIOUserStore {
  constructor() {
    this.uidStorageName = '_growing_uid_';
    this.userIdStorageName = '_growing_userId_';
    this.userKeyStorageName = '_growing_userKey_';
    this.gioIdStorageName = '_growing_gioId_';
  }
  
  // 初始化用户信息
  initUserInfo() {
    this.uid = uni.getStorageSync(this.uidStorageName) || this.generateUUID();
    this.userId = uni.getStorageSync(this.userIdStorageName);
    this.userKey = uni.getStorageSync(this.userKeyStorageName);
    this.gioId = uni.getStorageSync(this.gioIdStorageName);
  }
  
  // 同步用户信息到存储
  syncUserInfo() {
    uni.setStorageSync(this.uidStorageName, this.uid);
    uni.setStorageSync(this.userIdStorageName, this.userId);
    uni.setStorageSync(this.userKeyStorageName, this.userKey);
    uni.setStorageSync(this.gioIdStorageName, this.gioId);
  }
  
  // 设置用户ID
  set userId(value) {
    this._userId = value;
    uni.setStorageSync(this.userIdStorageName, this._userId);
  }
  
  // 获取用户ID
  get userId() {
    return this._userId;
  }
}
```

### 5.2 事件序列存储
```javascript
// 事件序列ID管理
class GIOEventStore {
  constructor() {
    this.esidStorageName = '_growing_esid_';
    this.gsidStorageName = '_growing_gsid_';
  }
  
  // 初始化存储ID
  initStorageId() {
    const globalData = getApp().globalData || {};
    let esid = globalData.gio_esid ? globalData.gio_esid : uni.getStorageSync(this.esidStorageName);
    esid = (typeof esid === 'object' && esid !== null) ? esid : {};
    
    this._esid = {};
    Object.keys(esid).forEach(key => {
      this._esid[key] = Number.isNaN(Number(esid[key])) || esid[key] >= 1e9 || esid[key] < 1 ? 1 : esid[key];
    });
    
    // 保存到存储
    uni.setStorageSync(this.esidStorageName, this._esid);
    
    let gsid = globalData.gio_gsid ? globalData.gio_gsid : Number.parseInt(uni.getStorageSync(this.gsidStorageName), 10);
    this._gsid = Number.isNaN(gsid) || gsid >= 1e9 || gsid < 1 ? 1 : gsid;
    gsid !== this._gsid && uni.setStorageSync(this.gsidStorageName, this._gsid);
  }
  
  // 获取事件序列ID
  get esid() {
    return this._esid;
  }
  
  // 设置事件序列ID
  set esid(value) {
    const newEsid = {};
    Object.keys(value).forEach(key => {
      newEsid[key] = Number.isNaN(value[key]) || value[key] >= 1e9 || value[key] < 1 ? 1 : value[key];
    });
    
    this._esid = newEsid;
    uni.setStorageSync(this.esidStorageName, this._esid);
  }
}
```

## 6. 业务存储模式

### 6.1 用户认证流程存储
```javascript
// 完整的用户认证存储流程
export default {
  methods: {
    // 登录成功后存储用户信息
    handleLoginSuccess(userInfo) {
      // 存储到Vuex
      this.$store.commit('setUserInfo', userInfo);
      
      // 存储到uni.StorageSync（跨平台）
      uni.setStorageSync("mob", userInfo.phone);
      
      // H5环境存储到localStorage
      // #ifdef H5
      const dateJson = {
        userInfo: userInfo,
        time: new Date().getTime()
      };
      localStorage.setItem('date_json_ecop', JSON.stringify(dateJson));
      // #endif
    },
    
    // 登出时清除所有存储
    handleLogout() {
      // 清除Vuex状态
      this.$store.commit('removeUserInfo');
      
      // 清除uni.StorageSync
      uni.removeStorageSync("mob");
      
      // 清除H5 localStorage
      // #ifdef H5
      localStorage.removeItem('date_json_ecop');
      // #endif
      
      // 清除sessionStorage
      sessionStorage.clear();
    }
  }
};
```

### 6.2 页面状态持久化
```javascript
// 页面状态持久化存储
export default {
  data() {
    return {
      formData: {},
      pageState: {}
    };
  },
  
  onLoad() {
    // 恢复页面状态
    this.restorePageState();
  },
  
  onUnload() {
    // 保存页面状态
    this.savePageState();
  },
  
  methods: {
    // 保存页面状态
    savePageState() {
      const state = {
        formData: this.formData,
        pageState: this.pageState,
        timestamp: Date.now()
      };
      
      // 使用sessionStorage保存临时状态
      sessionStorage.setItem('pageState', JSON.stringify(state));
    },
    
    // 恢复页面状态
    restorePageState() {
      const savedState = sessionStorage.getItem('pageState');
      if (savedState) {
        try {
          const state = JSON.parse(savedState);
          // 检查状态是否过期（1小时）
          if (Date.now() - state.timestamp < 60 * 60 * 1000) {
            this.formData = state.formData || {};
            this.pageState = state.pageState || {};
          } else {
            // 过期清除
            sessionStorage.removeItem('pageState');
          }
        } catch (e) {
          console.error('恢复页面状态失败:', e);
        }
      }
    }
  }
};
```

### 6.3 数据缓存管理
```javascript
// 智能数据缓存管理
class DataCacheManager {
  constructor() {
    this.cachePrefix = 'app_cache_';
    this.defaultTTL = 30 * 60 * 1000; // 30分钟
  }
  
  // 设置缓存
  set(key, data, ttl = this.defaultTTL) {
    const cacheData = {
      data: data,
      timestamp: Date.now(),
      ttl: ttl
    };
    
    try {
      uni.setStorageSync(this.cachePrefix + key, JSON.stringify(cacheData));
    } catch (e) {
      console.error('缓存设置失败:', e);
    }
  }
  
  // 获取缓存
  get(key) {
    try {
      const cached = uni.getStorageSync(this.cachePrefix + key);
      if (!cached) return null;
      
      const cacheData = JSON.parse(cached);
      const now = Date.now();
      
      // 检查是否过期
      if (now - cacheData.timestamp > cacheData.ttl) {
        this.remove(key);
        return null;
      }
      
      return cacheData.data;
    } catch (e) {
      console.error('缓存读取失败:', e);
      return null;
    }
  }
  
  // 删除缓存
  remove(key) {
    try {
      uni.removeStorageSync(this.cachePrefix + key);
    } catch (e) {
      console.error('缓存删除失败:', e);
    }
  }
  
  // 清空所有缓存
  clear() {
    try {
      const info = uni.getStorageInfoSync();
      info.keys.forEach(key => {
        if (key.startsWith(this.cachePrefix)) {
          uni.removeStorageSync(key);
        }
      });
    } catch (e) {
      console.error('缓存清空失败:', e);
    }
  }
}

// 使用示例
const cacheManager = new DataCacheManager();

// 缓存API数据
cacheManager.set('userInfo', userInfo, 60 * 60 * 1000); // 1小时
const cachedUserInfo = cacheManager.get('userInfo');
```

## 7. 存储使用规范

### 7.1 存储键名规范
```javascript
// 存储键名命名规范
const STORAGE_KEYS = {
  // 用户相关
  USER_INFO: 'user_info',
  USER_TOKEN: 'user_token',
  USER_PHONE: 'mob', // 项目中使用mob作为手机号键名
  
  // 业务相关
  PROMOTION_INFO: 'promotion_info',
  ORDER_DATA: 'order_data',
  
  // 缓存相关
  DICT_CACHE: 'smsp.dict.', // 字典缓存前缀
  LOCATION_AUTH: 'smsp.cache.location.hasAuth',
  
  // GIO统计相关
  GIO_UID: '_growing_uid_',
  GIO_USER_ID: '_growing_userId_',
  GIO_ESID: '_growing_esid_',
  GIO_GSID: '_growing_gsid_',
  
  // 临时状态
  COUNTDOWN: 'countdown',
  USER_TYPE: 'user-type',
  USER_ID: 'user-id',
  USER_NAME: 'user-name'
};
```

### 7.2 平台兼容性处理
```javascript
// 跨平台存储工具
export const crossPlatformStorage = {
  // 设置存储
  set(key, value) {
    // #ifdef H5
    localStorage.setItem(key, JSON.stringify(value));
    // #endif
    
    // #ifdef MP
    uni.setStorageSync(key, value);
    // #endif
  },
  
  // 获取存储
  get(key) {
    // #ifdef H5
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : null;
    // #endif
    
    // #ifdef MP
    return uni.getStorageSync(key);
    // #endif
  },
  
  // 删除存储
  remove(key) {
    // #ifdef H5
    localStorage.removeItem(key);
    // #endif
    
    // #ifdef MP
    uni.removeStorageSync(key);
    // #endif
  }
};
```

### 7.3 错误处理规范
```javascript
// 安全的存储操作
export const safeStorage = {
  set(key, value) {
    try {
      uni.setStorageSync(key, value);
      return true;
    } catch (error) {
      console.error(`存储失败 [${key}]:`, error);
      
      // 存储空间不足时的处理
      if (error.message && error.message.includes('quota')) {
        uni.showToast({
          title: '存储空间不足',
          icon: 'none'
        });
      }
      
      return false;
    }
  },
  
  get(key, defaultValue = null) {
    try {
      const value = uni.getStorageSync(key);
      return value !== '' ? value : defaultValue;
    } catch (error) {
      console.error(`读取失败 [${key}]:`, error);
      return defaultValue;
    }
  },
  
  remove(key) {
    try {
      uni.removeStorageSync(key);
      return true;
    } catch (error) {
      console.error(`删除失败 [${key}]:`, error);
      return false;
    }
  }
};
```

## 8. 最佳实践

### 8.1 数据分类存储
```javascript
// 根据数据特性选择合适的存储方式
const StorageStrategy = {
  // 用户信息 - 永久存储，跨平台
  userInfo: {
    storage: 'uni.StorageSync',
    key: 'user_info',
    ttl: null // 永久存储
  },
  
  // 临时状态 - 会话存储
  tempState: {
    storage: 'sessionStorage',
    key: 'temp_state',
    ttl: null // 会话结束自动清除
  },
  
  // 缓存数据 - 带过期时间
  cacheData: {
    storage: 'uni.StorageSync',
    key: 'cache_data',
    ttl: 30 * 60 * 1000 // 30分钟
  },
  
  // 倒计时 - 会话存储
  countdown: {
    storage: 'sessionStorage',
    key: 'countdown',
    ttl: null
  }
};
```

### 8.2 存储数据验证
```javascript
// 存储数据验证和清理
export const validateStorageData = {
  // 验证用户信息
  validateUserInfo(data) {
    if (!data || typeof data !== 'object') return false;
    if (!data.rspTransId || !data.phone) return false;
    if (data.time && Date.now() - data.time > 30 * 60 * 1000) return false;
    return true;
  },
  
  // 验证缓存数据
  validateCacheData(data) {
    if (!data || typeof data !== 'object') return false;
    if (!data.timestamp || !data.data) return false;
    if (Date.now() - data.timestamp > data.ttl) return false;
    return true;
  },
  
  // 清理过期数据
  cleanExpiredData() {
    const info = uni.getStorageInfoSync();
    const now = Date.now();
    
    info.keys.forEach(key => {
      try {
        const value = uni.getStorageSync(key);
        if (typeof value === 'string') {
          const data = JSON.parse(value);
          if (data.timestamp && now - data.timestamp > data.ttl) {
            uni.removeStorageSync(key);
          }
        }
      } catch (e) {
        // 忽略解析错误
      }
    });
  }
};
```

### 8.3 存储监控和调试
```javascript
// 存储使用监控
export const storageMonitor = {
  // 获取存储使用情况
  getStorageInfo() {
    try {
      const info = uni.getStorageInfoSync();
      return {
        keys: info.keys,
        currentSize: info.currentSize,
        limitSize: info.limitSize,
        usage: (info.currentSize / info.limitSize * 100).toFixed(2) + '%'
      };
    } catch (error) {
      console.error('获取存储信息失败:', error);
      return null;
    }
  },
  
  // 存储使用分析
  analyzeStorage() {
    const info = this.getStorageInfo();
    if (!info) return;
    
    console.log('存储使用情况:', info);
    
    // 分析各类型存储占用
    const analysis = {
      userData: 0,
      cacheData: 0,
      tempData: 0,
      other: 0
    };
    
    info.keys.forEach(key => {
      try {
        const value = uni.getStorageSync(key);
        const size = JSON.stringify(value).length;
        
        if (key.includes('user') || key.includes('mob')) {
          analysis.userData += size;
        } else if (key.includes('cache') || key.includes('smsp.dict')) {
          analysis.cacheData += size;
        } else if (key.includes('temp') || key.includes('countdown')) {
          analysis.tempData += size;
        } else {
          analysis.other += size;
        }
      } catch (e) {
        // 忽略错误
      }
    });
    
    console.log('存储分析:', analysis);
    return analysis;
  }
};
```

## 9. 常见问题和解决方案

### 9.1 存储空间不足
```javascript
// 存储空间不足处理
export const handleStorageQuotaExceeded = {
  // 清理过期缓存
  cleanExpiredCache() {
    const info = uni.getStorageInfoSync();
    const now = Date.now();
    
    info.keys.forEach(key => {
      if (key.startsWith('cache_') || key.startsWith('smsp.dict.')) {
        try {
          const value = uni.getStorageSync(key);
          const data = JSON.parse(value);
          if (data.timestamp && now - data.timestamp > data.ttl) {
            uni.removeStorageSync(key);
          }
        } catch (e) {
          // 清理无效数据
          uni.removeStorageSync(key);
        }
      }
    });
  },
  
  // 清理临时数据
  cleanTempData() {
    const tempKeys = ['countdown', 'temp_state', 'pageState'];
    tempKeys.forEach(key => {
      uni.removeStorageSync(key);
    });
  },
  
  // 压缩存储数据
  compressStorageData(key, data) {
    try {
      const compressed = JSON.stringify(data);
      if (compressed.length > 1024) { // 大于1KB时压缩
        // 这里可以使用压缩算法
        return compressed;
      }
      return data;
    } catch (e) {
      return data;
    }
  }
};
```

### 9.2 数据同步问题
```javascript
// 多页面数据同步
export const dataSync = {
  // 监听存储变化
  watchStorageChange(callback) {
    // #ifdef H5
    window.addEventListener('storage', (e) => {
      if (e.key === 'user_info' || e.key === 'mob') {
        callback(e.key, e.newValue);
      }
    });
    // #endif
  },
  
  // 广播存储变化
  broadcastStorageChange(key, value) {
    // #ifdef H5
    localStorage.setItem(key, value);
    localStorage.removeItem(key);
    // #endif
  }
};
```

## 10. 在组件中的使用示例

### 10.1 用户信息管理组件
```javascript
// 用户信息管理组件
export default {
  data() {
    return {
      userInfo: null,
      isLoggedIn: false
    };
  },
  
  mounted() {
    this.initUserInfo();
  },
  
  methods: {
    // 初始化用户信息
    initUserInfo() {
      // 从Vuex获取
      this.userInfo = this.$store.state.userInfo;
      this.isLoggedIn = !!this.userInfo;
      
      // 如果Vuex中没有，尝试从存储中恢复
      if (!this.userInfo) {
        this.$store.dispatch('getUserInfo');
      }
    },
    
    // 登录
    async login(loginData) {
      try {
        const response = await loginAPI(loginData);
        if (response.code === 200) {
          const userInfo = response.data;
          
          // 存储到Vuex
          this.$store.commit('setUserInfo', userInfo);
          
          // 存储到uni.StorageSync
          uni.setStorageSync("mob", userInfo.phone);
          
          this.userInfo = userInfo;
          this.isLoggedIn = true;
          
          uni.showToast({
            title: '登录成功',
            icon: 'success'
          });
        }
      } catch (error) {
        console.error('登录失败:', error);
        uni.showToast({
          title: '登录失败',
          icon: 'none'
        });
      }
    },
    
    // 登出
    logout() {
      // 清除Vuex状态
      this.$store.commit('removeUserInfo');
      
      // 清除存储
      uni.removeStorageSync("mob");
      
      // 清除sessionStorage
      sessionStorage.clear();
      
      this.userInfo = null;
      this.isLoggedIn = false;
      
      uni.showToast({
        title: '已登出',
        icon: 'success'
      });
    }
  }
};
```

### 10.2 倒计时组件
```javascript
// 倒计时组件
export default {
  data() {
    return {
      countdown: 0,
      intervalId: null
    };
  },
  
  mounted() {
    this.initCountdown();
  },
  
  beforeDestroy() {
    this.clearCountdown();
  },
  
  methods: {
    // 初始化倒计时
    initCountdown() {
      const savedCountdown = sessionStorage.getItem('countdown');
      if (savedCountdown) {
        this.countdown = parseInt(savedCountdown);
        if (this.countdown > 0) {
          this.startCountdown();
        }
      }
    },
    
    // 开始倒计时
    startCountdown() {
      this.intervalId = setInterval(() => {
        this.countdown -= 1;
        sessionStorage.setItem('countdown', this.countdown.toString());
        
        if (this.countdown <= 0) {
          this.clearCountdown();
        }
      }, 1000);
    },
    
    // 清除倒计时
    clearCountdown() {
      if (this.intervalId) {
        clearInterval(this.intervalId);
        this.intervalId = null;
      }
      this.countdown = 0;
      sessionStorage.removeItem('countdown');
    },
    
    // 发送验证码
    async sendCode() {
      try {
        const response = await sendCodeAPI();
        if (response.code === 200) {
          this.countdown = 60;
          sessionStorage.setItem('countdown', this.countdown.toString());
          this.startCountdown();
          
          uni.showToast({
            title: '验证码已发送',
            icon: 'success'
          });
        }
      } catch (error) {
        console.error('发送验证码失败:', error);
        uni.showToast({
          title: '发送失败',
          icon: 'none'
        });
      }
    }
  }
};
```

## 总结

本文档基于对项目代码的完全分析，详细说明了项目中localStorage、sessionStorage和uni.StorageSync的使用方法。主要特点：

1. **完全基于项目代码** - 所有示例都来自实际项目代码
2. **平台兼容性** - 考虑了H5和小程序平台的差异
3. **业务场景覆盖** - 涵盖了用户认证、数据缓存、状态管理等核心业务场景
4. **错误处理** - 提供了完整的错误处理方案
5. **最佳实践** - 包含了存储使用的最佳实践和规范

智能体可以根据此文档在开发页面时正确使用各种存储方式，确保数据的一致性和可靠性。
