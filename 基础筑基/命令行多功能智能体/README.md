# 命令行多功能智能体 (CLI Multi-functional Agent)

## 项目简介
开发一个支持3类工具动态调用的命令行多功能助手，包括：
- 天气查询
- 单位换算  
- 网页摘要

## 依赖包说明

### 核心依赖
- **langchain**: LangChain框架核心，用于构建Agent
- **langchain-community**: LangChain社区工具集
- **langchain-core**: LangChain核心组件
- **openai**: OpenAI API客户端，用于大语言模型调用

### 命令行界面
- **click**: 命令行界面库，提供装饰器风格的CLI构建
- **rich**: 富文本终端输出，美化命令行界面
- **typer**: 基于类型提示的CLI构建库

### 功能模块依赖
- **天气查询**: `requests`, `aiohttp` - HTTP请求库
- **单位换算**: `pint` - 物理单位处理库
- **网页摘要**: `beautifulsoup4`, `lxml`, `newspaper3k` - 网页解析和摘要

### 开发工具
- **black**: 代码格式化工具
- **flake8**: 代码质量检查
- **isort**: import语句排序
- **pytest**: 测试框架

## 安装依赖
```bash
# 方法1: 使用安装脚本
python install_dependencies.py

# 方法2: 直接安装
pip install -r requirements.txt
```

## 项目结构
```
命令行多功能智能体/
├── cli_agent.py          # 主程序入口
├── requirements.txt       # 依赖列表
├── install_dependencies.py # 依赖安装脚本
├── config.json           # 配置文件
└── README.md             # 项目说明
```

## 开发计划
1. 实现基础CLI框架
2. 集成LangChain Agent
3. 实现三个功能模块
4. 与AgentExecutor对比分析
5. 性能优化和测试
