# Vuex Store 全局状态管理使用文档

## 概述
本文档详细说明了 `src/store/index.js` 中所有全局状态变量和修改状态的方法，用于指导智能体在工程中开发页面时正确使用Vuex状态管理。

## 全局状态变量 (State)

### 1. 用户认证相关
| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `token` | String | `''` | 用户认证令牌 |
| `phone` | String | `''` | 用户手机号码 |
| `ylUserInfo` | Object | `{}` | 用户详细信息对象 |

### 2. 加密配置相关
| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `encryptEnabled` | Boolean | `true` | 加密功能开关 |
| `requestSecret` | String | `''` | 请求加密密钥 |
| `responseSecret` | String | `''` | 响应加密密钥 |
| `signSecret` | String | `''` | 签名密钥 |

### 3. 地图组件相关
| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `mapVisible` | Boolean | `false` | 地图组件显示状态 |
| `mapCallback` | Function | `null` | 地图选择完成回调函数 |

### 4. 登录相关
| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `Handle4G` | Boolean | `false` | 4G登录状态，true进行4G登录，false获取脱敏手机号 |
| `confuseMobile` | String | `''` | 4G获取的脱敏手机号 |
| `curryVal` | Number | `60` | 登录倒计时秒数 |
| `TimerLogin` | Number | `null` | 登录倒计时定时器ID |
| `disableBtn` | Boolean | `false` | 获取验证码按钮禁用状态 |
| `loginPage` | Boolean | `false` | 是否进入登录页标识 |

### 5. 业务相关
| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `jctg` | String | `''` | 宽带入参标识 |
| `buildingInfo` | Object | `{}` | 楼宇信息对象 |
| `grpord` | String | `null` | 订单号 |
| `ordType` | String | `'0'` | 订单类型：'0'存量订购，'1'新开户一键办理 |
| `qrCodeId` | String | `''` | 二维码ID |
| `region` | String | `null` | 地区信息 |
| `groupId` | String | `null` | 组ID |
| `customerManagerInfo` | Object | `null` | 客户经理信息 |

## 状态修改方法 (Mutations)

### 1. 用户认证相关
```javascript
// 设置用户token
setToken(state, payload)
// 使用示例：this.$store.commit('setToken', 'your-token-here')

// 设置用户手机号
setPhone(state, payload)
// 使用示例：this.$store.commit('setPhone', '13800138000')

// 设置用户详细信息
setYlUserInfo(state, payload)
// 使用示例：this.$store.commit('setYlUserInfo', { servnumber: 'xxx', ... })
```

### 2. 加密配置相关
```javascript
// 设置加密配置信息
setEncryptInfo(state, payload)
// 使用示例：this.$store.commit('setEncryptInfo', {
//   encryptEnabled: true,
//   requestSecret: 'xxx',
//   responseSecret: 'xxx',
//   signSecret: 'xxx'
// })
```

### 3. 地图组件相关
```javascript
// 设置地图显示状态
setMapVisible(state, payload)
// 使用示例：this.$store.commit('setMapVisible', true)

// 设置地图回调函数
setMapCallback(state, payload)
// 使用示例：this.$store.commit('setMapCallback', (data) => { console.log(data) })
```

### 4. 登录相关
```javascript
// 设置脱敏手机号
setConfuseMobile(state, payload)
// 使用示例：this.$store.commit('setConfuseMobile', '138****8000')

// 设置4G登录状态
setHandle4G(state, payload)
// 使用示例：this.$store.commit('setHandle4G', true)

// 设置登录倒计时
setCurryVal(state, payload)
// 使用示例：this.$store.commit('setCurryVal', 60)

// 设置验证码按钮状态
setDisableBtn(state, payload)
// 使用示例：this.$store.commit('setDisableBtn', true)

// 设置登录页状态
setLoginPage(state, payload)
// 使用示例：this.$store.commit('setLoginPage', true)
```

### 5. 业务相关
```javascript
// 设置宽带标识
setJctg(state, payload)
// 使用示例：this.$store.commit('setJctg', 'broadband-flag')

// 设置楼宇信息
setBuildingInfo(state, payload)
// 使用示例：this.$store.commit('setBuildingInfo', { address: 'xxx', ... })

// 设置订单号
setGrpordInfo(state, payload)
// 使用示例：this.$store.commit('setGrpordInfo', 'ORDER123456')

// 设置订单类型
setOrderType(state, payload)
// 使用示例：this.$store.commit('setOrderType', '1')

// 设置二维码ID
setQrCodeId(state, payload)
// 使用示例：this.$store.commit('setQrCodeId', 'QR123456')

// 设置客户经理信息（注意：此mutation实际设置的是customerManagerInfo）
setCustManagerNo(state, payload)
// 使用示例：this.$store.commit('setCustManagerNo', { custManagerNo: 'CUST123456' })

// 设置地区信息
setRegion(state, payload)
// 使用示例：this.$store.commit('setRegion', 'beijing')

// 设置组ID
setGroupId(state, payload)
// 使用示例：this.$store.commit('setGroupId', 'GROUP123')

// 设置客户经理信息
setCustomerManagerInfo(state, data)
// 使用示例：this.$store.commit('setCustomerManagerInfo', { name: 'xxx', ... })
```

## 异步操作方法 (Actions)

### 1. 用户信息相关
```javascript
// 获取用户信息（带加密解密）
getYxUserInfo({ commit, dispatch })
// 使用示例：this.$store.dispatch('getYxUserInfo')

// 设置用户信息
setUserInfo({ commit }, params)
// 使用示例：this.$store.dispatch('setUserInfo', { rspTransId: 'xxx', phone: 'xxx' })

// 获取缓存的用户信息
getUserInfo({ commit, dispatch })
// 使用示例：this.$store.dispatch('getUserInfo')
```

### 2. 业务数据相关
```javascript
// 获取订单和楼宇信息
getGrpordAndBuilding({ commit, dispatch }, params)
// 使用示例：this.$store.dispatch('getGrpordAndBuilding', { token: 'xxx', servnumber: 'xxx' })

// 获取楼宇详细信息
getBuildingInfo({ commit }, id)
// 使用示例：this.$store.dispatch('getBuildingInfo', 'building123')
```

### 3. 地图组件相关
```javascript
// 打开地图组件
openMap({ commit }, { data = {}, callback = null })
// 使用示例：this.$store.dispatch('openMap', { 
//   data: { lat: 39.9, lng: 116.4 }, 
//   callback: (result) => console.log(result) 
// })

// 关闭地图组件
closeMap({ commit })
// 使用示例：this.$store.dispatch('closeMap')

// 地图选择完成
mapSelectComplete({ commit, state }, data)
// 使用示例：this.$store.dispatch('mapSelectComplete', { lat: 39.9, lng: 116.4 })
```

### 4. 登录相关
```javascript
// 登录倒计时
printTimeLogin({ commit, dispatch, state })
// 使用示例：this.$store.dispatch('printTimeLogin')

// 移除用户信息（已注释，但保留供参考）
// removeUserInfo({ commit })
// 功能：清除token、phone和localStorage中的用户信息
// 注意：此方法在代码中被注释，如需使用需要先取消注释
```

### 5. 已注释的方法
```javascript
// 移除用户信息（当前被注释）
// removeUserInfo({ commit })
// 功能说明：
// - 清除用户token
// - 清除用户手机号
// - 清除localStorage中的用户缓存
// 使用场景：用户登出时清理所有用户相关数据
// 注意：此方法在代码中被注释，如需使用需要先取消注释
```

## 在组件中的使用方式

### 1. 获取状态
```javascript
// 在Vue组件中
computed: {
  // 直接获取状态
  token() {
    return this.$store.state.token;
  },
  phone() {
    return this.$store.state.phone;
  },
  // 使用mapState辅助函数
  ...mapState(['token', 'phone', 'ylUserInfo'])
}
```

### 2. 修改状态
```javascript
// 在Vue组件中
methods: {
  // 直接提交mutation
  updateToken() {
    this.$store.commit('setToken', 'new-token');
  },
  // 使用mapMutations辅助函数
  ...mapMutations(['setToken', 'setPhone']),
  
  // 调用action
  async getUserData() {
    await this.$store.dispatch('getYxUserInfo');
  },
  // 使用mapActions辅助函数
  ...mapActions(['getYxUserInfo', 'setUserInfo'])
}
```

### 3. 监听状态变化
```javascript
// 在Vue组件中
watch: {
  '$store.state.token'(newVal, oldVal) {
    if (newVal) {
      console.log('用户已登录');
    } else {
      console.log('用户未登录');
    }
  }
}
```

## 注意事项

1. **状态持久化**：部分状态如 `token`、`phone` 等会在localStorage中缓存，缓存有效期为30分钟
2. **加密处理**：用户手机号等敏感信息会进行AES加密处理
3. **异步操作**：大部分数据获取都是异步操作，需要正确处理Promise
4. **地图组件**：地图相关状态需要配合地图组件使用，注意回调函数的设置
5. **登录流程**：登录相关状态有复杂的业务逻辑，包括4G登录、脱敏手机号等
6. **错误处理**：在actions中需要适当处理API调用失败的情况
7. **代码一致性**：确保mutation名称与实际设置的state属性一致

## 常用开发模式

### 1. 页面初始化时获取用户信息
```javascript
mounted() {
  this.$store.dispatch('getYxUserInfo');
}
```

### 2. 登录后设置用户信息
```javascript
async login(loginData) {
  try {
    const result = await loginAPI(loginData);
    this.$store.dispatch('setUserInfo', result);
    // 跳转到首页
    this.$router.push('/');
  } catch (error) {
    console.error('登录失败:', error);
  }
}
```

### 3. 使用地图选择地址
```javascript
openMapSelector() {
  this.$store.dispatch('openMap', {
    data: { lat: 39.9, lng: 116.4 },
    callback: (selectedData) => {
      console.log('选择的地址:', selectedData);
      // 处理选择结果
    }
  });
}
```

### 4. 检查登录状态
```javascript
computed: {
  isLoggedIn() {
    return !!this.$store.state.token;
  }
}
```

这份文档提供了完整的Vuex状态管理使用指南，智能体可以根据此文档在开发页面时正确使用全局状态变量和修改状态的方法。
