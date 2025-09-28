# AI Agent 学习与实践项目

> 🚀 我学习agent过程中路径图、阅读的文献、编写的程序都会放在这里，希望能和更多agent开发者交流

## 📈 学习进度概览

项目按照4个阶段设计学习路径，目前已完成：

1. **第一阶段：基础筑基** ✅ - 掌握Agent本质与核心循环
2. **第二阶段：框架进阶** 🚧 - 掌握工业级框架与复杂任务规划  
3. **第三阶段：系统工程** 📋 - 构建高可用多Agent系统
4. **第四阶段：前沿突破** 📋 - 建立技术壁垒，实现领域闭环

详细学习路线请查看：[开发学习路线.html](开发学习路线.html)

---

## 🏗️ 第一阶段：基础筑基
**目标**：掌握Agent本质与核心循环，脱离框架实现基础功能

### 核心目标
掌握Agent本质与核心循环，脱离框架实现基础功能

### 技术要点

#### 手动实现ReAct循环
用Python原生代码构建200行内的Agent内核（Thought推理生成 → Action工具解析 → Observation执行反馈），重点设计标准化工具调用JSON协议（含错误重试机制），集成日志跟踪系统（logging模块）。

#### 现代概念映射
将经典Agent理论转化为LLM实现：
- 感知 → API/文本输入解析器
- 决策 → Chain-of-Thought提示词模板  
- 执行 → 工具路由字典（如 {"search": google_search}）

### 项目实践

#### 内核实现
- `agent_core.py` - 纯Python实现的Agent内核，包含完整的ReAct循环
- `agent_core_qwen.py` - 集成通义千问API的完整Agent实现

**核心特性**：
- ✅ 工具注册系统（装饰器模式）
- ✅ 标准化工具调用JSON协议
- ✅ 错误重试机制
- ✅ 日志跟踪系统

#### 命令行多功能智能体
开发命令行多功能助手（天气查询+单位换算+网页摘要），支持5类工具动态调用，并与LangChain的AgentExecutor对比理解框架价值。

**实际工具集**：
- `get_web_code` - 获取网页代码
- `revert_pound_to_kg` - 将磅转换为千克
- `revert_meter_to_cm` - 将厘米转换为米
- `compute_bmi` - 计算BMI
- `get_city_weather` - 获取城市天气（集成和风天气API）

**技术对比分析**：
- `与LangChain的AgentExecutor对比结果.md` - 详细的技术对比分析
- 自定义Agent支持并行工具执行
- 与LangChain框架的深度对比

### 资源建议
精读OpenAI《构建Agent实战指南》基础部分，配合LangChain官方文档的Tool Calling章节。

---

## 🚀 第二阶段：框架进阶
**目标**：掌握工业级框架，解决长上下文与复杂任务规划

### 核心目标
掌握工业级框架，解决长上下文与复杂任务规划

### 技术要点

#### LangChain深度实战
- 使用LCRL构建多步骤工作流（如数据获取→分析→生成报告）
- 集成短期记忆（ConversationSummaryMemory）和长期记忆（ChromaDB向量库），应用RAG增强技术如子文档检索（降低Token开销）和HyDE（假设答案引导检索）

#### 规划算法落地
实现Plan-and-Execute模式：
- Planner Agent拆解任务（如"分析销售数据并生成PPT"）
- Executor Agent调用工具链分步执行

### 项目实践

#### exp1 - 智能开发助手
**核心概念**：基于todo_list的Agent执行控制

**应用场景**：需求文档解析 → 页面结构分析 → API开发规划

**技术实现**：
- LangChain + ChromaDB向量数据库
- 多文档RAG检索（6个向量数据库）
- 工具链集成（文件操作、图片处理）
- ConversationBufferMemory对话记忆

**实际功能**：
- 需求文档向量化存储
- API调用说明检索
- 公共工具方法检索
- 存储使用说明检索
- 全局状态管理检索
- 工具方法检索

#### 向量存储和检索器
**功能**：PDF文档处理与向量化存储

**技术实现**：
- PyPDFLoader文档加载
- MarkdownHeaderTextSplitter文本分割
- ChromaDB向量存储
- DashScope嵌入模型（text-embedding-v2）

#### 聊天机器人
**架构**：FastAPI + LangServe + DeepSeek Chat

**特性**：
- RESTful API接口
- 支持流式响应
- 消息历史管理
- Token计数和消息修剪

**实际实现**：
- `robot.py` - 服务端实现
- `client.py` - 客户端实现（支持手动和自动模式）
- `trim_messages源码.py` - 消息修剪源码

#### 简单应用
**功能**：翻译服务

**技术栈**：
- FastAPI + LangServe
- DeepSeek Chat模型
- ChatPromptTemplate提示模板
- StrOutputParser输出解析

**实际文件**：
- `server.py` - 服务端实现
- `client.py` - 客户端实现
- `simple-llm-app.py` - 简单LLM应用示例

### 资源建议
学习LangChain中文教程中的Chain组件，参考AutoGPT开源项目架构。

---

## 📖 文献及学习总结

### 学习资料
- `LangChain Tool Calling学习总结.html` - 精美的HTML学习笔记
- `OpenAI-构建Agent实战指南.pdf` - OpenAI官方指南
- `REACT-SYNERGIZING REASONING AND ACTING IN.pdf` - ReAct论文原文
- `ReAct论文总结.docx` - 论文中文总结

## 🛠️ 技术栈

### 核心框架
- **LangChain** - 主流Agent框架
- **FastAPI** - Web服务框架
- **ChromaDB** - 向量数据库
- **DashScope** - 阿里云大模型服务

### 开发工具
- **Python 3.12.9** - 主要开发语言
- **Git** - 版本控制

### API服务
- **通义千问** - 阿里云大模型API
- **DeepSeek** - 深度求索大模型API
- **和风天气** - 天气数据API

---

## 🚀 快速开始

### 环境准备
```bash
# 克隆项目
git clone <your-repo-url>
cd agent

# 安装依赖
pip install -r requirements.txt
```

### 配置设置
1. 复制 `config.default.json` 为 `config.json`
2. 填入你的API密钥（每个实验项目可能会有不同，请以实际为准）：
   - QWen-API-KEY（通义千问）
   - DeepSeek-API-KEY（深度求索）
   - QWeather-API-BASE（和风天气）

### 运行示例

#### 基础Agent
```bash
cd 基础筑基/命令行多功能智能体
python main.py
```

#### 向量检索
```bash
cd 框架进阶/向量存储和检索器
python main.py
```

#### 聊天机器人
```bash
cd 框架进阶/聊天机器人
python robot.py
```

#### 智能开发助手
```bash
cd 框架进阶/exp1
python main.py
```

---

## 🎯 核心特性

### 🔧 工具系统
- **装饰器注册**：`@register_tool` 自动生成工具schema
- **类型推断**：自动解析函数参数类型
- **错误处理**：完善的异常捕获和重试机制
- **并行执行**：支持多工具并行调用

### 🧠 推理引擎
- **ReAct循环**：Thought → Action → Observation
- **上下文管理**：智能的对话历史处理
- **工具选择**：基于语义理解的工具路由

### 📊 监控与日志
- **结构化日志**：完整的执行轨迹记录
- **性能监控**：Token消耗、响应时间统计
- **错误追踪**：详细的异常堆栈信息

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 贡献方式
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

### 代码规范
- 添加详细的文档字符串
- 编写单元测试

---

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - 优秀的Agent框架
- [OpenAI](https://openai.com/) - Agent构建指南
- [ReAct论文](https://arxiv.org/abs/2210.03629) - 理论基础

## 📞 联系方式

- 项目作者：ЮЦ Янь
- 邮箱：805709525@qq.com
- 项目地址：https://github.com/yyqtc/agent-learning

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！
