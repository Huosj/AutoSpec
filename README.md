# AutoSpec
随着Vibe Coding的兴起，它的缺陷很快暴露出来。
困境：1）不一致性和不可预测性：LLM 的输出是不确定的，容易在错误的地方生成代码，甚至存在上下文遗忘的情况。
2）Bug与安全风险：Vibe Coding利用LLM预测可能的代码序列，甚至是。这很容易引入Bug或安全漏洞（如暴露的API密钥）。
3）难以维护：LLM以极快的速度生成大量难懂的代码，导致即使资深的开发者也难以/不愿维护。LLM也很难理解复杂的项目。
解决方案：本项目用 LangGraph 编排 7 个智能体节点，基于SPEC设计模式由「需求-设计-任务-代码」四步辅助LLM编程：
（1）判断用户开发需求是否明确，明确后即时创建语义化项目目录。（2）严格按 EARS 格式输出需求文档（条件or触发器、系统、响应）。（3）基于需求文档，将宏观设计分解为微观、可执行的设计清单。（4）把设计拆成原子级开发任务，与需求一一对应，并给出技术栈建议。（5）基于任务文档，把任务翻译成规范、可运行的代码块，以便后续开发。（6）实现按需调用的网络搜索工具（基于自定义本地mcp，也支持接入公开mcp），与其他节点建立循环，补足基座模型对新知识的盲区。（7）对开发需求未明确的提问即时给出回答，基于多轮对话与记忆能力，最终明确用户需求。

整条流水线以Graph 驱动，节点间通过条件边灵活跳转，支持复杂工具调用：搜索后回到原节点，由大模型继续判断是否需要搜索；所有文档与代码自动写入版本化目录，实现“一句话需求 → 可运行系统”分钟级闭环。


## 功能特点
- 意图识别 ：分析用户输入，识别开发意图
- 需求生成 ：根据意图自动生成详细的需求文档
- 设计生成 ：基于需求生成系统设计方案
- 任务分解 ：将项目分解为可执行的具体任务
- 代码生成 ：根据设计和任务自动生成代码框架
- 工作汇报 ：生成项目进展和完成情况的汇报
- 模块化设计 ：各功能模块独立，便于扩展和维护
## 项目结构
```
AutoSpec/
├── __init__.py
├── main.py          # 主程序入口
├── graph.py         # 工作流图定义
├── nodes/           # 功能节点
│   ├── intent_recognition_node.py  # 意图识别
节点
│   ├── generate_requirements_node.py  # 需求
生成节点
│   ├── generate_design_node.py     # 设计生成
节点
│   ├── generate_tasks_node.py      # 任务分解
节点
│   ├── generate_code_node.py       # 代码生成
节点
│   ├── generate_response_node.py   # 响应生成
节点
│   └── work_report_node.py         # 工作汇报
节点
├── tools/           # 工具函数
│   ├── tools.py
│   └── search.py                   #！！！请自己实现或者使用搜索mcp
└── utils/           # 通用工具
    └── utils.py
```
## 安装说明
1. 确保已安装 Python 3.7 或更高版本
2. 克隆本仓库
   ```
   git clone https://github.com/Huosj/
   AutoSpec.git
   cd AutoSpec
   ```
## 使用方法
1. 运行主程序
   ```
   python main.py
   ```
2. 根据提示输入你的开发需求或意图
3. 系统将自动执行相应的流程，生成所需的文档或代码
## 示例
```
# 示例代码（main.py）
from utils.utils import read_file
from graph import build_graph

# 构建工作流图
graph = build_graph()

# 运行工作流
result = graph.run({
    "input": "我需要开发一个简单的待办事项应用"
})

print(result)
```