# API 调用文档

## 概述
本文档详细说明了 `src/api` 目录中所有API接口的调用方法，用于指导智能体在工程中开发页面时正确使用和复用已有的API，以及创建新的API时保持一致的风格。

## 目录结构
```
src/api/
├── index.js          # API入口文件
├── user.js           # 用户相关API
├── product.js        # 商品相关API
├── product_im.js     # 聊天页面商品API
├── system.js         # 系统相关API
├── workbench.js      # 工作台相关API
├── contract.js       # 合同相关API
├── file.js           # 文件相关API
├── message.js        # 消息相关API
├── visitor.js        # 访客相关API
├── promotion.js      # 推广相关API
└── api-wxl.js        # 微信联相关API
```

## API 调用规范

### 1. 基础请求方式
- **加密请求**: 使用 `encryptRequest` 进行敏感数据请求
- **普通请求**: 使用 `request` 或 `requestUtil` 进行一般数据请求
- **文件上传**: 使用 `uni.uploadFile` 进行文件上传

### 2. 请求头规范
- 所有请求都需要包含认证信息
- 加密请求需要包含 `m-r-s-k` 头部
- 文件上传需要设置正确的 `Content-Type`

### 3. 响应处理规范
- 统一处理成功和失败响应
- 错误信息统一显示
- 加载状态统一管理

## 1. 用户相关API (user.js)

### 1.1 获取用户信息
```javascript
import { getUserInfo } from '@/api/user.js';

// 获取用户信息
const getUserData = async () => {
  try {
    const response = await getUserInfo();
    console.log('用户信息:', response);
    return response;
  } catch (error) {
    console.error('获取用户信息失败:', error);
    throw error;
  }
};
```

### 1.2 上传身份证
```javascript
import { uploadIdCard } from '@/api/user.js';

// 上传身份证图片
const uploadIdCardImage = async (imageData) => {
  try {
    const response = await uploadIdCard({
      idCardFront: imageData.frontImage,
      idCardBack: imageData.backImage
    });
    console.log('身份证上传成功:', response);
    return response;
  } catch (error) {
    console.error('身份证上传失败:', error);
    throw error;
  }
};
```

### 1.3 获取地图Token
```javascript
import { getMapToken } from '@/api/user.js';

// 获取作战地图token
const getMapTokenData = async (params) => {
  try {
    const response = await getMapToken({
      mapType: 'mobile',
      ...params
    });
    console.log('地图Token:', response);
    return response;
  } catch (error) {
    console.error('获取地图Token失败:', error);
    throw error;
  }
};
```

### 1.4 在途订单相关
```javascript
import { 
  getOnWayOrder, 
  getOnWayOrderTemplate, 
  getOnWayOrderPersonal 
} from '@/api/user.js';

// 获取在途订单
const getOnWayOrderData = async (params) => {
  try {
    const response = await getOnWayOrder({
      orderType: 'all',
      pageNum: 1,
      pageSize: 10,
      ...params
    });
    console.log('在途订单:', response);
    return response;
  } catch (error) {
    console.error('获取在途订单失败:', error);
    throw error;
  }
};

// 获取在途订单模板
const getOrderTemplate = async (templateId) => {
  try {
    const response = await getOnWayOrderTemplate({
      templateId: templateId
    });
    console.log('订单模板:', response);
    return response;
  } catch (error) {
    console.error('获取订单模板失败:', error);
    throw error;
  }
};

// 获取个人预约订单
const getPersonalOrders = async (params) => {
  try {
    const response = await getOnWayOrderPersonal({
      pageNum: 1,
      pageSize: 10,
      status: 'pending',
      ...params
    });
    console.log('个人预约订单:', response);
    return response;
  } catch (error) {
    console.error('获取个人预约订单失败:', error);
    throw error;
  }
};
```

## 2. 商品相关API (product.js)

### 2.1 商品列表和详情
```javascript
import { 
  getProductContent, 
  getPersonalGoods,
  productAPI 
} from '@/api/product.js';

// 获取商品详情
const getProductDetail = async (goodsId) => {
  try {
    const response = await getProductContent(goodsId);
    console.log('商品详情:', response);
    return response;
  } catch (error) {
    console.error('获取商品详情失败:', error);
    throw error;
  }
};

// 获取个人订购商品
const getPersonalGoodsData = async (goodsId) => {
  try {
    const response = await getPersonalGoods(goodsId);
    console.log('个人订购商品:', response);
    return response;
  } catch (error) {
    console.error('获取个人订购商品失败:', error);
    throw error;
  }
};

// 使用productAPI获取商品列表
const getProductList = async (params) => {
  try {
    const response = await productAPI.getProductList({
      pageNum: 1,
      pageSize: 10,
      categoryId: '',
      keyword: '',
      ...params
    });
    console.log('商品列表:', response);
    return response;
  } catch (error) {
    console.error('获取商品列表失败:', error);
    throw error;
  }
};

// 使用productAPI获取商品详情
const getProductDetailById = async (productId) => {
  try {
    const response = await productAPI.getProductDetail(productId);
    console.log('商品详情:', response);
    return response;
  } catch (error) {
    console.error('获取商品详情失败:', error);
    throw error;
  }
};
```

### 2.2 商品预约和提交
```javascript
import { 
  sumbitPersonalGoods, 
  sumbitReservation 
} from '@/api/product.js';

// 个人订购商品提交预约
const submitPersonalOrder = async (orderData) => {
  try {
    const response = await sumbitPersonalGoods({
      goodsId: orderData.goodsId,
      quantity: orderData.quantity,
      contactInfo: orderData.contactInfo,
      address: orderData.address,
      ...orderData
    });
    console.log('预约提交成功:', response);
    return response;
  } catch (error) {
    console.error('预约提交失败:', error);
    throw error;
  }
};

// 商机推送
const submitReservation = async (reservationData) => {
  try {
    const response = await sumbitReservation({
      productId: reservationData.productId,
      customerInfo: reservationData.customerInfo,
      requirements: reservationData.requirements,
      ...reservationData
    });
    console.log('商机推送成功:', response);
    return response;
  } catch (error) {
    console.error('商机推送失败:', error);
    throw error;
  }
};
```

### 2.3 商品校验相关
```javascript
import { 
  checkMemWorkProdBRM, 
  checkBuildingIdMarkGroupRule, 
  checkDbBuilding, 
  checkMobileStatus 
} from '@/api/product.js';

// 工作网主号校验
const checkWorkProd = async (params) => {
  try {
    const response = await checkMemWorkProdBRM({
      mainNumber: params.mainNumber,
      buildingId: params.buildingId,
      ...params
    });
    console.log('工作网主号校验结果:', response);
    return response;
  } catch (error) {
    console.error('工作网主号校验失败:', error);
    throw error;
  }
};

// 楼宇ID校验
const checkBuildingRule = async (params) => {
  try {
    const response = await checkBuildingIdMarkGroupRule({
      buildingId: params.buildingId,
      groupId: params.groupId,
      ...params
    });
    console.log('楼宇ID校验结果:', response);
    return response;
  } catch (error) {
    console.error('楼宇ID校验失败:', error);
    throw error;
  }
};

// 根据区县校验客户经理信息
const checkBuilding = async (params) => {
  try {
    const response = await checkDbBuilding({
      region: params.region,
      district: params.district,
      ...params
    });
    console.log('客户经理校验结果:', response);
    return response;
  } catch (error) {
    console.error('客户经理校验失败:', error);
    throw error;
  }
};

// 办理号码状态字典
const getMobileStatus = async (params) => {
  try {
    const response = await checkMobileStatus({
      dictType: 'mobile_status',
      ...params
    });
    console.log('号码状态字典:', response);
    return response;
  } catch (error) {
    console.error('获取号码状态字典失败:', error);
    throw error;
  }
};
```

## 3. 系统相关API (system.js)

### 3.1 楼宇和订单信息
```javascript
import { 
  getGrpordAndBuilding, 
  getBuildingInfo, 
  getCustomerManagerInfo 
} from '@/api/system.js';

// 获取楼宇ID和订单信息
const getBuildingAndOrder = async (params) => {
  try {
    const response = await getGrpordAndBuilding({
      token: params.token,
      servnumber: params.servnumber,
      ...params
    });
    console.log('楼宇和订单信息:', response);
    return response;
  } catch (error) {
    console.error('获取楼宇和订单信息失败:', error);
    throw error;
  }
};

// 获取楼宇详细信息
const getBuildingDetails = async (buildingId) => {
  try {
    const response = await getBuildingInfo(buildingId);
    console.log('楼宇详细信息:', response);
    return response;
  } catch (error) {
    console.error('获取楼宇详细信息失败:', error);
    throw error;
  }
};

// 获取客户经理信息
const getManagerInfo = async (accountManagerId) => {
  try {
    const response = await getCustomerManagerInfo(accountManagerId);
    console.log('客户经理信息:', response);
    return response;
  } catch (error) {
    console.error('获取客户经理信息失败:', error);
    throw error;
  }
};
```

## 4. 工作台相关API (workbench.js)

### 4.1 二维码管理
```javascript
import { 
  getQRCodeList, 
  addQRCode, 
  getQRCodeDetail, 
  getQRCodeImg 
} from '@/api/workbench.js';

// 获取二维码列表
const getQRCodeListData = async (params) => {
  try {
    const response = await getQRCodeList({
      pageNum: 1,
      pageSize: 10,
      status: 'active',
      ...params
    });
    console.log('二维码列表:', response);
    return response;
  } catch (error) {
    console.error('获取二维码列表失败:', error);
    throw error;
  }
};

// 新增二维码
const createQRCode = async (qrCodeData) => {
  try {
    const response = await addQRCode({
      qrName: qrCodeData.name,
      buildingId: qrCodeData.buildingId,
      agentId: qrCodeData.agentId,
      ...qrCodeData
    });
    console.log('二维码创建成功:', response);
    return response;
  } catch (error) {
    console.error('二维码创建失败:', error);
    throw error;
  }
};

// 获取二维码详情
const getQRCodeDetails = async (qrCodeId) => {
  try {
    const response = await getQRCodeDetail(qrCodeId);
    console.log('二维码详情:', response);
    return response;
  } catch (error) {
    console.error('获取二维码详情失败:', error);
    throw error;
  }
};

// 生成二维码图片
const generateQRCodeImage = async (params) => {
  try {
    const response = await getQRCodeImg({
      qrCodeId: params.qrCodeId,
      size: params.size || 200,
      ...params
    });
    console.log('二维码图片生成成功:', response);
    return response;
  } catch (error) {
    console.error('二维码图片生成失败:', error);
    throw error;
  }
};
```

### 4.2 楼宇和渠道管理
```javascript
import { 
  getBuildingList, 
  getChannelList, 
  getAgentList, 
  getCustomerList 
} from '@/api/workbench.js';

// 获取楼宇列表
const getBuildingListData = async (params) => {
  try {
    const response = await getBuildingList({
      pageNum: 1,
      pageSize: 10,
      region: '',
      keyword: '',
      ...params
    });
    console.log('楼宇列表:', response);
    return response;
  } catch (error) {
    console.error('获取楼宇列表失败:', error);
    throw error;
  }
};

// 获取渠道代理列表
const getChannelListData = async (params) => {
  try {
    const response = await getChannelList({
      pageNum: 1,
      pageSize: 10,
      status: 'active',
      ...params
    });
    console.log('渠道代理列表:', response);
    return response;
  } catch (error) {
    console.error('获取渠道代理列表失败:', error);
    throw error;
  }
};

// 获取代理商列表
const getAgentListData = async (params) => {
  try {
    const response = await getAgentList({
      pageNum: 1,
      pageSize: 10,
      agentType: '',
      ...params
    });
    console.log('代理商列表:', response);
    return response;
  } catch (error) {
    console.error('获取代理商列表失败:', error);
    throw error;
  }
};

// 获取客户经理列表
const getCustomerListData = async (params) => {
  try {
    const response = await getCustomerList({
      pageNum: 1,
      pageSize: 10,
      region: '',
      ...params
    });
    console.log('客户经理列表:', response);
    return response;
  } catch (error) {
    console.error('获取客户经理列表失败:', error);
    throw error;
  }
};
```

### 4.3 推广员信息
```javascript
import { getPromotionInfo } from '@/api/workbench.js';

// 获取推广员用户信息
const getPromotionUserInfo = async (phoneNumber) => {
  try {
    const response = await getPromotionInfo(phoneNumber);
    console.log('推广员信息:', response);
    return response;
  } catch (error) {
    console.error('获取推广员信息失败:', error);
    throw error;
  }
};
```

## 5. 合同相关API (contract.js)

### 5.1 合同生成和提交
```javascript
import { 
  doGenerateContract, 
  doOfflineSubmit, 
  doOnlineSubmit 
} from '@/api/contract.js';

// 生成合同
const generateContract = async (contractData) => {
  try {
    const response = await doGenerateContract({
      orderId: contractData.orderId,
      contractType: contractData.contractType,
      templateId: contractData.templateId,
      ...contractData
    });
    console.log('合同生成成功:', response);
    return response;
  } catch (error) {
    console.error('合同生成失败:', error);
    throw error;
  }
};

// 线下签约提交
const submitOfflineContract = async (submitData) => {
  try {
    const response = await doOfflineSubmit({
      orderId: submitData.orderId,
      signatureData: submitData.signatureData,
      ...submitData
    });
    console.log('线下签约提交成功:', response);
    return response;
  } catch (error) {
    console.error('线下签约提交失败:', error);
    throw error;
  }
};

// 线上签约
const submitOnlineContract = async (contractData) => {
  try {
    const response = await doOnlineSubmit({
      orderId: contractData.orderId,
      contractContent: contractData.contractContent,
      ...contractData
    });
    console.log('线上签约成功:', response);
    return response;
  } catch (error) {
    console.error('线上签约失败:', error);
    throw error;
  }
};
```

### 5.2 订单和签约信息
```javascript
import { 
  getOrderList, 
  getRmInfo, 
  getHomeInfo, 
  getPersonRmInfo, 
  getOrderTrack 
} from '@/api/contract.js';

// 获取待授权待签约列表
const getOrderListData = async (params) => {
  try {
    const response = await getOrderList({
      pageNum: 1,
      pageSize: 10,
      status: 'pending',
      ...params
    });
    console.log('订单列表:', response);
    return response;
  } catch (error) {
    console.error('获取订单列表失败:', error);
    throw error;
  }
};

// 回显签约信息
const getContractInfo = async (requirementOid) => {
  try {
    const response = await getRmInfo(requirementOid);
    console.log('签约信息:', response);
    return response;
  } catch (error) {
    console.error('获取签约信息失败:', error);
    throw error;
  }
};

// 返回首页信息
const getHomePageInfo = async (orderId) => {
  try {
    const response = await getHomeInfo(orderId);
    console.log('首页信息:', response);
    return response;
  } catch (error) {
    console.error('获取首页信息失败:', error);
    throw error;
  }
};

// 回显一键办理信息
const getPersonContractInfo = async (personData) => {
  try {
    const response = await getPersonRmInfo({
      personId: personData.personId,
      orderType: personData.orderType,
      ...personData
    });
    console.log('一键办理信息:', response);
    return response;
  } catch (error) {
    console.error('获取一键办理信息失败:', error);
    throw error;
  }
};

// 查询订单轨迹
const getOrderTracking = async (trackData) => {
  try {
    const response = await getOrderTrack({
      orderId: trackData.orderId,
      ...trackData
    });
    console.log('订单轨迹:', response);
    return response;
  } catch (error) {
    console.error('获取订单轨迹失败:', error);
    throw error;
  }
};
```

## 6. 文件相关API (file.js)

### 6.1 文件上传
```javascript
import fileAPI from '@/api/file.js';

// 上传文件
const uploadFile = async (fileData) => {
  try {
    const response = await fileAPI.upload({
      filePath: fileData.filePath,
      name: fileData.name || 'file',
      formData: fileData.formData || {},
      uploadUrl: fileData.uploadUrl || '/workbench/file/upload'
    });
    console.log('文件上传成功:', response);
    return response;
  } catch (error) {
    console.error('文件上传失败:', error);
    throw error;
  }
};

// 删除文件
const deleteFile = async (fileName) => {
  try {
    const response = await fileAPI.deleteFile(fileName);
    console.log('文件删除成功:', response);
    return response;
  } catch (error) {
    console.error('文件删除失败:', error);
    throw error;
  }
};

// 获取文件下载URL
const getFileDownloadUrl = async (fileName) => {
  try {
    const response = await fileAPI.getDownloadUrl(fileName);
    console.log('文件下载URL:', response);
    return response;
  } catch (error) {
    console.error('获取文件下载URL失败:', error);
    throw error;
  }
};

// 检查文件是否存在
const checkFileExists = async (fileName) => {
  try {
    const response = await fileAPI.checkFileExists(fileName);
    console.log('文件存在性检查:', response);
    return response;
  } catch (error) {
    console.error('文件存在性检查失败:', error);
    throw error;
  }
};
```

## 7. 消息相关API (message.js)

### 7.1 消息发送和接收
```javascript
import messageAPI from '@/api/message.js';

// 发送消息
const sendMessage = async (messageData) => {
  try {
    const response = await messageAPI.send({
      visitor_id: messageData.visitorId,
      message_type: messageData.messageType || 'text',
      content: messageData.content,
      ...messageData
    });
    console.log('消息发送成功:', response);
    return response;
  } catch (error) {
    console.error('消息发送失败:', error);
    throw error;
  }
};

// 获取历史消息
const getMessageHistory = async (visitorId) => {
  try {
    const response = await messageAPI.getMessages(visitorId);
    console.log('历史消息:', response);
    return response;
  } catch (error) {
    console.error('获取历史消息失败:', error);
    throw error;
  }
};

// 分页获取消息
const getMessagesByPage = async (params) => {
  try {
    const response = await messageAPI.getMessagesPages({
      visitor_id: params.visitorId,
      pageNum: params.pageNum || 1,
      pageSize: params.pageSize || 20,
      ...params
    });
    console.log('分页消息:', response);
    return response;
  } catch (error) {
    console.error('获取分页消息失败:', error);
    throw error;
  }
};
```

### 7.2 消息状态管理
```javascript
// 获取未读消息数量
const getUnreadCount = async (visitorId) => {
  try {
    const response = await messageAPI.getUnreadCount(visitorId);
    console.log('未读消息数量:', response);
    return response;
  } catch (error) {
    console.error('获取未读消息数量失败:', error);
    throw error;
  }
};

// 标记消息为已读
const markMessageAsRead = async (visitorId) => {
  try {
    const response = await messageAPI.readMessage(visitorId);
    console.log('消息标记为已读:', response);
    return response;
  } catch (error) {
    console.error('标记消息为已读失败:', error);
    throw error;
  }
};

// 批量标记消息为已读
const batchMarkAsRead = async (params) => {
  try {
    const response = await messageAPI.batchMarkRead({
      visitor_ids: params.visitorIds,
      message_ids: params.messageIds,
      ...params
    });
    console.log('批量标记为已读:', response);
    return response;
  } catch (error) {
    console.error('批量标记为已读失败:', error);
    throw error;
  }
};
```

### 7.3 文件消息处理
```javascript
// 上传图片消息
const uploadImageMessage = async (file, visitorId) => {
  try {
    const response = await messageAPI.uploadImage(file, visitorId);
    console.log('图片上传成功:', response);
    return response;
  } catch (error) {
    console.error('图片上传失败:', error);
    throw error;
  }
};

// 上传文件消息
const uploadFileMessage = async (file, visitorId) => {
  try {
    const response = await messageAPI.uploadFile(file, visitorId);
    console.log('文件上传成功:', response);
    return response;
  } catch (error) {
    console.error('文件上传失败:', error);
    throw error;
  }
};

// 下载文件
const downloadFileMessage = async (fileUrl) => {
  try {
    const response = await messageAPI.downloadFile(fileUrl);
    console.log('文件下载成功:', response);
    return response;
  } catch (error) {
    console.error('文件下载失败:', error);
    throw error;
  }
};
```

## 8. 访客相关API (visitor.js)

### 8.1 访客登录
```javascript
import visitorAPI from '@/api/visitor.js';

// 访客登录
const visitorLogin = async (loginData) => {
  try {
    const response = await visitorAPI.login({
      phone: loginData.phone,
      smsCode: loginData.smsCode,
      ...loginData
    });
    console.log('访客登录成功:', response);
    return response;
  } catch (error) {
    console.error('访客登录失败:', error);
    throw error;
  }
};

// 访客Redis登录
const visitorRedisLogin = async (loginData) => {
  try {
    const response = await visitorAPI.loginRedis({
      visitorId: loginData.visitorId,
      ...loginData
    });
    console.log('访客Redis登录成功:', response);
    return response;
  } catch (error) {
    console.error('访客Redis登录失败:', error);
    throw error;
  }
};
```

## 9. 推广相关API (promotion.js)

### 9.1 推广员管理
```javascript
import { 
  getPromotionInfo, 
  getPromotionList, 
  getPromotionDetail, 
  addPromotion, 
  updatePromotion, 
  deletePromotion 
} from '@/api/promotion.js';

// 获取推广员统计信息
const getPromotionStatistics = async (phoneNumber) => {
  try {
    const response = await getPromotionInfo(phoneNumber);
    console.log('推广员统计信息:', response);
    return response;
  } catch (error) {
    console.error('获取推广员统计信息失败:', error);
    throw error;
  }
};

// 获取推广员列表
const getPromotionListData = async (params, phoneNumber) => {
  try {
    const response = await getPromotionList({
      pageNum: 1,
      pageSize: 10,
      ...params
    }, phoneNumber);
    console.log('推广员列表:', response);
    return response;
  } catch (error) {
    console.error('获取推广员列表失败:', error);
    throw error;
  }
};

// 获取推广员详情
const getPromotionDetails = async (id, phoneNumber) => {
  try {
    const response = await getPromotionDetail(id, phoneNumber);
    console.log('推广员详情:', response);
    return response;
  } catch (error) {
    console.error('获取推广员详情失败:', error);
    throw error;
  }
};

// 新增推广员
const createPromotion = async (promotionData, phoneNumber) => {
  try {
    const response = await addPromotion({
      name: promotionData.name,
      phone: promotionData.phone,
      department: promotionData.department,
      ...promotionData
    }, phoneNumber);
    console.log('推广员创建成功:', response);
    return response;
  } catch (error) {
    console.error('推广员创建失败:', error);
    throw error;
  }
};

// 修改推广员
const updatePromotionData = async (promotionData, phoneNumber) => {
  try {
    const response = await updatePromotion({
      id: promotionData.id,
      name: promotionData.name,
      phone: promotionData.phone,
      department: promotionData.department,
      ...promotionData
    }, phoneNumber);
    console.log('推广员修改成功:', response);
    return response;
  } catch (error) {
    console.error('推广员修改失败:', error);
    throw error;
  }
};

// 删除推广员
const deletePromotionData = async (id, phoneNumber) => {
  try {
    const response = await deletePromotion(id, phoneNumber);
    console.log('推广员删除成功:', response);
    return response;
  } catch (error) {
    console.error('推广员删除失败:', error);
    throw error;
  }
};
```

### 9.2 订单和产品管理
```javascript
import { 
  getPromotionOrderStatistics, 
  getPromotionOrderList, 
  getMyOrderStatistics, 
  getOrderDetail, 
  getMyOrderList 
} from '@/api/promotion.js';

// 获取渠道订单统计
const getChannelOrderStats = async (phoneNumber) => {
  try {
    const response = await getPromotionOrderStatistics(phoneNumber);
    console.log('渠道订单统计:', response);
    return response;
  } catch (error) {
    console.error('获取渠道订单统计失败:', error);
    throw error;
  }
};

// 获取渠道订单列表
const getChannelOrderList = async (params, phoneNumber) => {
  try {
    const response = await getPromotionOrderList({
      pageNum: 1,
      pageSize: 10,
      ...params
    }, phoneNumber);
    console.log('渠道订单列表:', response);
    return response;
  } catch (error) {
    console.error('获取渠道订单列表失败:', error);
    throw error;
  }
};

// 获取我的订单统计
const getMyOrderStats = async (phoneNumber) => {
  try {
    const response = await getMyOrderStatistics(phoneNumber);
    console.log('我的订单统计:', response);
    return response;
  } catch (error) {
    console.error('获取我的订单统计失败:', error);
    throw error;
  }
};

// 获取订单详情
const getOrderDetails = async (orderData, phoneNumber) => {
  try {
    const response = await getOrderDetail({
      orderId: orderData.orderId,
      ...orderData
    }, phoneNumber);
    console.log('订单详情:', response);
    return response;
  } catch (error) {
    console.error('获取订单详情失败:', error);
    throw error;
  }
};

// 获取我的订单列表
const getMyOrderListData = async (params, phoneNumber) => {
  try {
    const response = await getMyOrderList({
      pageNum: 1,
      pageSize: 10,
      ...params
    }, phoneNumber);
    console.log('我的订单列表:', response);
    return response;
  } catch (error) {
    console.error('获取我的订单列表失败:', error);
    throw error;
  }
};
```

## 10. 微信联相关API (api-wxl.js)

### 10.1 产品分类和列表
```javascript
import { 
  getPins, 
  getCats, 
  getProducts, 
  getBans, 
  getGoods, 
  getMediaList 
} from '@/api/api-wxl.js';

// 获取栏目导航
const getNavigationTabs = async () => {
  try {
    const response = await getPins();
    console.log('栏目导航:', response);
    return response;
  } catch (error) {
    console.error('获取栏目导航失败:', error);
    throw error;
  }
};

// 获取产品分类
const getProductCategories = async () => {
  try {
    const response = await getCats();
    console.log('产品分类:', response);
    return response;
  } catch (error) {
    console.error('获取产品分类失败:', error);
    throw error;
  }
};

// 获取产品列表
const getProductListData = async (params) => {
  try {
    const response = await getProducts({
      pageNum: 1,
      pageSize: 10,
      categoryId: '',
      pinId: '',
      ...params
    });
    console.log('产品列表:', response);
    return response;
  } catch (error) {
    console.error('获取产品列表失败:', error);
    throw error;
  }
};

// 获取横幅图
const getBannerImages = async () => {
  try {
    const response = await getBans();
    console.log('横幅图:', response);
    return response;
  } catch (error) {
    console.error('获取横幅图失败:', error);
    throw error;
  }
};

// 获取移动企下产品列表
const getMobileProducts = async () => {
  try {
    const response = await getGoods();
    console.log('移动企下产品列表:', response);
    return response;
  } catch (error) {
    console.error('获取移动企下产品列表失败:', error);
    throw error;
  }
};

// 获取首页媒体列表
const getMediaListData = async () => {
  try {
    const response = await getMediaList();
    console.log('首页媒体列表:', response);
    return response;
  } catch (error) {
    console.error('获取首页媒体列表失败:', error);
    throw error;
  }
};
```

## 在组件中的使用方式

### 1. 基础API调用
```javascript
// 在Vue组件中
import { getUserInfo } from '@/api/user.js';
import { getProductContent } from '@/api/product.js';

export default {
  data() {
    return {
      userInfo: null,
      productInfo: null
    };
  },
  async mounted() {
    try {
      // 获取用户信息
      this.userInfo = await getUserInfo();
      
      // 获取商品信息
      this.productInfo = await getProductContent('GOODS001');
    } catch (error) {
      console.error('数据加载失败:', error);
    }
  }
};
```

### 2. 带加载状态的API调用
```javascript
// 在Vue组件中
import { showLoading, hideLoading, showToast } from '@/utils/index.js';
import { getProductList } from '@/api/product.js';

export default {
  data() {
    return {
      productList: [],
      loading: false
    };
  },
  methods: {
    async loadProductList() {
      try {
        this.loading = true;
        showLoading('加载中...');
        
        const response = await getProductList({
          pageNum: 1,
          pageSize: 10
        });
        
        this.productList = response.data || [];
      } catch (error) {
        showToast('加载失败', 'error');
        console.error('加载商品列表失败:', error);
      } finally {
        this.loading = false;
        hideLoading();
      }
    }
  }
};
```

### 3. 表单提交API调用
```javascript
// 在Vue组件中
import { sumbitPersonalGoods } from '@/api/product.js';
import { showToast } from '@/utils/index.js';

export default {
  data() {
    return {
      formData: {
        goodsId: '',
        quantity: 1,
        contactInfo: {},
        address: ''
      }
    };
  },
  methods: {
    async submitOrder() {
      try {
        // 表单验证
        if (!this.validateForm()) {
          return;
        }
        
        showLoading('提交中...');
        
        const response = await sumbitPersonalGoods(this.formData);
        
        showToast('提交成功');
        this.resetForm();
      } catch (error) {
        showToast('提交失败', 'error');
        console.error('订单提交失败:', error);
      } finally {
        hideLoading();
      }
    },
    
    validateForm() {
      if (!this.formData.goodsId) {
        showToast('请选择商品', 'error');
        return false;
      }
      if (!this.formData.contactInfo.phone) {
        showToast('请输入联系电话', 'error');
        return false;
      }
      return true;
    }
  }
};
```

## API 开发规范

### 1. 新建API文件规范
```javascript
// 新建API文件示例: src/api/newModule.js
import { request, encryptRequest } from '@/utils/http.js';
import { tools } from '@/common/utils/tools.js';

const baseURL = '/iqimall/api';

// 1. 普通请求示例
export function getNewData(params) {
  return request({
    baseURL,
    url: '/new-module/getData',
    method: 'post',
    data: params,
    headers: {
      'Content-Type': 'application/json'
    }
  });
}

// 2. 加密请求示例
export function getSensitiveData(params) {
  return encryptRequest({
    baseURL,
    url: '/new-module/sensitiveData',
    method: 'post',
    data: params,
    header: {
      token: tools.getToken(),
      ...tools.getYxHeaders()
    }
  });
}

// 3. 文件上传示例
export function uploadNewFile(fileData) {
  return new Promise((resolve, reject) => {
    uni.uploadFile({
      url: baseURL + '/new-module/upload',
      filePath: fileData.filePath,
      name: 'file',
      formData: fileData.formData || {},
      header: {
        token: tools.getToken(),
        ...tools.getYxHeaders()
      },
      success: (res) => {
        try {
          const result = JSON.parse(res.data);
          if (result.code === 200) {
            resolve(result);
          } else {
            reject(new Error(result.message || '上传失败'));
          }
        } catch (error) {
          reject(new Error('解析响应失败'));
        }
      },
      fail: (err) => {
        reject(new Error(err.errMsg || '上传失败'));
      }
    });
  });
}
```

### 2. API命名规范
- **获取数据**: `get` + 功能名，如 `getUserInfo`, `getProductList`
- **提交数据**: `submit` + 功能名，如 `submitOrder`, `submitForm`
- **创建数据**: `add` + 功能名，如 `addUser`, `addProduct`
- **更新数据**: `update` + 功能名，如 `updateUser`, `updateProduct`
- **删除数据**: `delete` + 功能名，如 `deleteUser`, `deleteProduct`
- **校验数据**: `check` + 功能名，如 `checkUser`, `checkProduct`

### 3. 参数传递规范
```javascript
// 1. 单个参数
export function getDataById(id) {
  return request({
    url: `/api/data/${id}`,
    method: 'get'
  });
}

// 2. 对象参数
export function getDataByParams(params) {
  return request({
    url: '/api/data',
    method: 'post',
    data: params
  });
}

// 3. 混合参数
export function getDataWithIdAndParams(id, params) {
  return request({
    url: `/api/data/${id}`,
    method: 'post',
    data: params
  });
}
```

### 4. 错误处理规范
```javascript
// 在API调用中统一处理错误
export async function callAPI(apiFunction, ...args) {
  try {
    const response = await apiFunction(...args);
    return response;
  } catch (error) {
    console.error('API调用失败:', error);
    // 统一错误处理
    if (error.message.includes('网络')) {
      showToast('网络连接失败', 'error');
    } else if (error.message.includes('权限')) {
      showToast('权限不足', 'error');
    } else {
      showToast('操作失败', 'error');
    }
    throw error;
  }
}
```

## 注意事项

1. **加密请求**: 敏感数据必须使用 `encryptRequest` 进行加密传输
2. **认证信息**: 所有请求都需要包含正确的认证头部信息
3. **错误处理**: 统一处理API调用失败的情况，提供用户友好的错误提示
4. **加载状态**: 长时间操作的API调用需要显示加载状态
5. **参数验证**: 在调用API前验证必要参数，避免无效请求
6. **响应处理**: 统一处理API响应格式，确保数据一致性
7. **缓存策略**: 对于不经常变化的数据，考虑使用缓存机制
8. **重试机制**: 对于网络请求失败的情况，考虑实现重试机制

## 常用开发模式

### 1. 列表页面API调用
```javascript
export default {
  data() {
    return {
      list: [],
      loading: false,
      pagination: {
        pageNum: 1,
        pageSize: 10,
        total: 0
      }
    };
  },
  methods: {
    async loadList() {
      try {
        this.loading = true;
        const response = await getListAPI({
          pageNum: this.pagination.pageNum,
          pageSize: this.pagination.pageSize
        });
        this.list = response.data || [];
        this.pagination.total = response.total || 0;
      } catch (error) {
        showToast('加载失败', 'error');
      } finally {
        this.loading = false;
      }
    }
  }
};
```

### 2. 表单提交API调用
```javascript
export default {
  data() {
    return {
      formData: {},
      submitting: false
    };
  },
  methods: {
    async submitForm() {
      if (!this.validateForm()) return;
      
      try {
        this.submitting = true;
        showLoading('提交中...');
        
        const response = await submitAPI(this.formData);
        showToast('提交成功');
        this.resetForm();
      } catch (error) {
        showToast('提交失败', 'error');
      } finally {
        this.submitting = false;
        hideLoading();
      }
    }
  }
};
```

这份文档提供了完整的API调用指南，智能体可以根据此文档在开发页面时正确使用和复用已有的API，并在创建新API时保持一致的风格和规范。
