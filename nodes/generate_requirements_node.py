import os
from langchain_core.messages import AIMessage
from utils.utils import remove_think, write_file
from nodes.work_report_node import generate_work_report


def generate_requirements(state, llm_with_tool, llm):
    """生成需求文档"""
    
    # 获取用户输入
    user_message = state["messages"][-1].content if state["messages"] else ""
    
    # 1. 先检查是否需要工具调用
    tool_response = llm_with_tool.invoke([{"role": "user", "content": user_message}])
    if hasattr(tool_response, 'tool_calls') and tool_response.tool_calls:
        # 需要工具调用，返回工具调用请求
        return {"messages": [tool_response], "next": "tools", "source_node": "generate_requirements"}
    
    # 2. 原有需求生成逻辑
    # 获取新目录路径
    new_dir = state.get("new_dir", ".")
    # 创建 .kiro 目录
    kiro_dir = os.path.join(new_dir, ".kiro")
    if not os.path.exists(kiro_dir):
        os.makedirs(kiro_dir)
    
    # 生成需求文档
    requirements_prompt = f"""请基于以下需求，生成一份详细的需求文档：

{user_message}

请按照以下结构生成需求文档，每个需求严格遵照EARS格式（包含零个或多个先决条件、零个或一个触发器、一个系统名称及一个或多个系统响应）：

# 需求分析

## 功能需求
[列出主要功能需求]
可能包含以下三类：
1. 通用性需求：系统运行全状态需满足的功能。
模板：xxx应该xxx。示例：手机的质量应该小于XX克。
2. 状态驱动需求：用WHILE标识，特定状态下生效。
模板：在xxx情况下xxx应xxx。示例：飞机飞行且发动机运行时，控制系统应保持燃油流量高于xx磅/秒。
3. 事件驱动需求：用“当…时”标识，特定事件触发响应。
模板：当xxx时，xxx应xxx。示例：飞行器发连续点火指令时，控制系统应启动连续点火。

## 非功能需求
[列出性能、安全等非功能需求]
可能包含以下三类：
1. 可选功能需求：“在…条件下”生效，特定配置启用。
模板：在xxx条件下，xxx应该xxx。示例：有超速保护功能时，飞机起飞前应测试其有效性。
2. 异常行为需求：用“如果，然后…”标记，处理异常情况。
模板：“如果xxx，那么xxx”，“应该xxx”。示例：空速不可用时，应使用模型空速。
3. 复杂需求：需满足多组先决条件才响应。
   - 期望行为模板：在…情况下…，当…时…, 当…时…应
   - 异常行为模板：在…情况下, 当…时…, 如果…应…
   示例：飞机在地面且收到反推力指令时，应启动反推力装置。
## 用户故事
[描述典型用户使用场景]
模板：作为...角色, 我需要...功能, 以便....

## 验收标准
[定义验收标准]
模板：当 [事件] 发生时，系统应 [响应]。

在生成需求时，可以使用`search_tool`来理解用户输入中提到的你不了解的概念，以确保需求的准确性和完整性，防止生成带有事实性错误的需求。
不要仅凭已有知识生成内容，对于任何不确定的信息都应通过工具搜索确认。务必仔细检查，确保所有需求都符合EARS格式要求。
"""
    
    requirements_response = llm.invoke([{"role": "user", "content": requirements_prompt}])
    requirements_content = remove_think(requirements_response.content)
    
    # 保存需求文档
    requirements_path = os.path.join(kiro_dir, "requirements.md")
    write_file(requirements_path, requirements_content)
    
    # 使用LLM生成工作汇报形式的摘要
    response_content = generate_work_report(state, requirements_content, "需求文档", llm)
    
    response = AIMessage(content=response_content)
    
    # MessagesState 将消息附加到 state 而不是覆盖
    return {"requirements_content": requirements_content, "messages": [response], "new_dir": new_dir, "next": "generate_design"}