
import os
import re

def remove_think(text):
    # 匹配 <think> 标签及其内容
    pattern = r"<think>.*?</think>"
    # 替换为空字符串
    return re.sub(pattern, "", text, flags=re.DOTALL)

"""
实现Langgraph 链
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.graph import END, add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage
from typing_extensions import TypedDict

# 导入工具装饰器
from langchain_core.tools import tool

# 导入百度搜索模块
from baidu_api import ai_search

class CustomState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    new_dir: str
    requirements_content: str
    design_content: str
    tasks_content: str
    source_node: str

def read_file(file_path):
    """读取文件内容"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"文件 {file_path} 不存在"
    except Exception as e:
        return f"读取文件时出错: {str(e)}"

def write_file(file_path, content):
    """写入内容到文件"""
    try:
        # 确保目录存在
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)  # 使用exist_ok=True避免目录已存在时的异常
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"成功写入文件: {file_path}"
    except Exception as e:
        return f"写入文件时出错: {str(e)}"

# 定义工具函数的描述

@tool
def search_tool(query: str) -> str:
    """使用百度搜索获取信息"""
    # 添加监控逻辑，记录输入
    print(f"[SEARCH_TOOL] 输入: {query}")
    result = ai_search(query)
    # 添加监控逻辑，记录输出
    # print(f"[SEARCH_TOOL] 输出: {result}")
    return result



def generate_work_report(state: CustomState, document_content: str, document_type: str, llm):
    """生成工作汇报形式的摘要"""
    # 使用LLM生成工作汇报形式的摘要
    summary_prompt = f"你是一个汇报者，你写了一个{document_type},内容如下：{document_content} 请你用三句话精准简洁地介绍自己做了什么。"
    summary_response = llm.invoke([{"role": "user", "content": summary_prompt}])
    summary_content = summary_response.content
    
    response_content = f"{document_type}已创建完成！\n{summary_content}"
    
    return response_content

def build_graph(llm_model_name):
    """构建 langgraph 链"""
    
    # 设置缓存
    set_llm_cache(InMemoryCache())
    
    llm = ChatOllama(
        model=llm_model_name, 
        temperature=0.2, 
        verbose=True,
        cache=True,  # 启用缓存机制
        streaming=True,  # 启用流式输出
        top_k=50,  # 限制模型只考虑概率最高的50个token
        top_p=0.9  # 限制模型只考虑累积概率达到0.9的token
    )
    
    # 绑定工具到LLM
    tools = [search_tool]
    llm_with_tool = llm.bind_tools(tools)

    # 意图识别节点：判断用户是否需要开发
    def intent_recognition(state: CustomState):
        """意图识别节点，判断用户是否需要开发"""
        
        # 获取用户输入
        user_message = state["messages"][-1].content if state["messages"] else ""
        
        # 1. 先检查是否需要工具调用
        tool_response = llm_with_tool.invoke([{"role": "user", "content": user_message}])
        if hasattr(tool_response, 'tool_calls') and tool_response.tool_calls:
            # 需要工具调用，返回工具调用请求
            return {"messages": [tool_response], "next": "tools", "source_node": "intent_recognition"}
        
        # 2. 原有意图识别逻辑
        intent_prompt = f"""请判断以下用户输入是否与软件开发相关，包括但不限于需求分析、系统设计、任务规划等：

用户输入: {user_message}

请回答"是"或"否"，并简要说明理由。
"""
        intent_response = llm.invoke([{"role": "user", "content": intent_prompt}])
        intent_result = remove_think(intent_response.content).lower()
        
        # 根据意图识别结果决定下一步
        if "是" in intent_result:
            # 如果是开发相关，创建新目录并进入需求文档生成节点
            # 使用LLM生成语义化的项目文件夹名，要求是精简的英文驼峰命名
            project_name_prompt = f"请基于以下用户需求生成一个简洁的英文驼峰命名的项目文件夹名称（只返回文件夹名称，不要包含任何其他内容）：\n\n{user_message}"
            project_name_response = llm.invoke([{"role": "user", "content": project_name_prompt}])
            # 提取实际的文件夹名称
            project_name = remove_think(project_name_response.content).strip()
            # 确保文件夹名称不超过30个字符
            if len(project_name) > 30:
                project_name = project_name[:30].strip()
            # 移除文件夹名称中的无效字符
            invalid_chars = '<>:"/\|?* '
            for char in invalid_chars:
                project_name = project_name.replace(char, '')
            # 移除首尾空格
            project_name = project_name.strip()
            # 如果文件夹名称为空，则使用默认名称
            if not project_name:
                project_name = "Project"
            # 确保文件夹名称不以点开头
            if project_name.startswith('.'):
                project_name = project_name[1:] if len(project_name) > 1 else "Project"
            # 确保第一个字符是字母
            if project_name and not project_name[0].isalpha():
                project_name = "Project" + project_name
            # 如果文件夹名称仍为空，则使用默认名称
            if not project_name:
                project_name = "Project"
            
            # 创建新目录，检查是否重名，如果重名则添加数字后缀
            new_dir = project_name
            counter = 1
            original_new_dir = new_dir
            while os.path.exists(new_dir):
                new_dir = f"{original_new_dir}_{counter}"
                counter += 1
            os.makedirs(new_dir, exist_ok=True)
            from langchain_core.messages import AIMessage
            response = AIMessage(content=f"已创建项目目录 {new_dir} 并开始生成需求文档。")
            # 直接返回扁平化结果，避免嵌套字典
            return {"next": "generate_requirements", "new_dir": new_dir, "messages": [response]}
        else:
            # 如果不是开发相关，直接生成回答
            from langchain_core.messages import AIMessage
            response = AIMessage(content="您的请求不是开发相关的，我将直接回答您的问题。")
            # 直接返回扁平化结果，避免嵌套字典
            return {"next": "generate_response", "messages": [response]}
    
    # 需求文档生成节点
    def generate_requirements(state: CustomState):
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
        
        from langchain_core.messages import AIMessage
        response = AIMessage(content=response_content)
        
        # MessagesState 将消息附加到 state 而不是覆盖
        return {"requirements_content": requirements_content, "messages": [response], "new_dir": new_dir, "next": "generate_design"}
    
    # 设计文档生成节点
    def generate_design(state: CustomState):
        """生成设计文档"""
        
        # 生成工具调用
        tool_calls = llm_with_tool.invoke([{"role": "user", "content": requirements_content}])
        
        # 输出工具调用结果
        if hasattr(tool_calls, 'tool_calls') and tool_calls.tool_calls:
            return {"next": "tools", "messages": [ToolMessage(content=tool_calls.tool_calls[0].content)], "source_node": "generate_design"}
        
        # 获取需求文档内容
        requirements_content = state.get("requirements_content", "")
        # 获取新目录路径
        new_dir = state.get("new_dir", ".")
        
        # 创建 .kiro 目录
        kiro_dir = os.path.join(new_dir, ".kiro")
        if not os.path.exists(kiro_dir):
            os.makedirs(kiro_dir)
        
        # 生成设计文档
        # 生成设计文档
        design_prompt = f"""请基于以下需求文档，将宏观设计分解为微观、可执行编码任务的清单：

需求文档：
{requirements_content}

请按照以下结构生成设计文档：

# 设计概述
[简要描述系统设计目标和核心思想]

## 架构设计
[描述系统整体架构，包括主要组件和它们之间的关系]

## 数据设计
[描述数据结构、数据库设计等]

## 接口设计
[描述系统内外部接口]

## 用户界面设计
[描述用户界面布局和交互设计]

## 安全设计
[描述安全相关的设计考虑]

## 性能设计
[描述性能相关的设计考虑]

## 部署设计
[描述系统部署方案]


在生成设计时，可以使用`search_tool`来获取相关领域的知识，以确保设计的准确性和完整性。不要仅凭已有知识生成内容，对于任何不确定的信息都应通过工具搜索确认。

注意：任务应具有原子性和可执行性，每个任务都应是离散、可管理的编码步骤，并明确关联到需求文档中的具体需求点。
"""
        
        design_response = llm.invoke([{"role": "user", "content": design_prompt}])
        design_content = remove_think(design_response.content)
        
        # 保存设计文档
        design_path = os.path.join(kiro_dir, "design.md")
        write_file(design_path, design_content)
        
        # 使用LLM生成工作汇报形式的摘要
        response_content = generate_work_report(state, design_content, "设计文档", llm)
        
        from langchain_core.messages import AIMessage
        response = AIMessage(content=response_content)
        
        # MessagesState 将消息附加到 state 而不是覆盖
        return {"design_content": design_content, "messages": [response], "new_dir": new_dir, "next": "generate_tasks"}
    
    # 任务文档生成节点
    def generate_tasks(state: CustomState):
        """生成任务文档"""
        
        # 生成工具调用
        tool_calls = llm_with_tool.invoke([{"role": "user", "content": design_content}])
        
        # 输出工具调用结果
        if hasattr(tool_calls, 'tool_calls') and tool_calls.tool_calls:
            return {"next": "tools", "messages": [ToolMessage(content=tool_calls.tool_calls[0].content)], "source_node": "generate_tasks"}
        
        # 获取设计文档内容
        design_content = state.get("design_content", "")
        
        # 获取需求文档内容
        requirements_content = state.get("requirements_content", "")
        
        # 获取新目录路径
        new_dir = state.get("new_dir", ".")
        
        # 创建 .kiro 目录
        kiro_dir = os.path.join(new_dir, ".kiro")
        if not os.path.exists(kiro_dir):
            os.makedirs(kiro_dir)
        
        # 生成任务文档
        # 生成开发任务列表
        tasks_prompt = f"""请基于以下需求和设计文档，生成一份详细的开发任务列表：

需求文档：
{requirements_content}

设计文档：
{design_content}

请按照以下结构生成开发任务列表，并确保每个任务都与需求文档和设计文档中的内容一一对应：

# 开发任务列表

## 项目初始化
[列出项目初始化相关的任务]

## 功能模块开发
[按模块列出开发任务]

## 数据库开发
[列出数据库相关的开发任务]

## 接口开发
[列出接口开发任务]

## 前端开发
[列出前端开发任务]

(不一定有)## 算法开发
[列出算法开发任务]

## 测试
[列出测试相关的任务]

## 部署
[列出部署相关的任务]

在生成任务列表时，对于你不懂的信息，你必须使用以下工具来获取更多信息：
- `search_tool`: 使用百度搜索获取信息

请务必在生成任务列表时，可以使用`search_tool`来获取相关开发领域的最前沿技术栈，以确保任务列表的准确性和完整性。不要仅凭已有知识生成内容，对于任何不确定的信息都应通过工具搜索确认。
"""
        
        tasks_response = llm.invoke([{"role": "user", "content": tasks_prompt}])
        tasks_content = remove_think(tasks_response.content)
        
        # 保存任务文档
        tasks_path = os.path.join(kiro_dir, "tasks.md")
        write_file(tasks_path, tasks_content)
        
        # 使用LLM生成工作汇报形式的摘要
        response_content = generate_work_report(state, tasks_content, "任务文档", llm)
        
        from langchain_core.messages import AIMessage
        response = AIMessage(content=response_content)
        
        # MessagesState 将消息附加到 state 而不是覆盖
        return {"messages": [response], "new_dir": new_dir, "tasks_content": tasks_content, "next": "generate_code"}
    
    # 回答生成节点
    def generate_response(state: CustomState):
        """生成普通响应"""
        
        # 生成工具调用
        tool_calls = llm_with_tool.invoke(state["messages"])
        
        # 输出工具调用结果
        if hasattr(tool_calls, 'tool_calls') and tool_calls.tool_calls:
            return {"next": "tools", "messages": [ToolMessage(content=remove_think(tool_calls.tool_calls[0].content))], "source_node": "generate_response"}
        
        response = llm.invoke(state["messages"])
        return {"next": "end", "messages": [response]}

    # 生成可执行代码节点
    def generate_code(state: CustomState):
        """基于三个文档生成可执行代码"""
        
        # 获取三个文档内容
        requirements_content = state.get("requirements_content", "")
        design_content = state.get("design_content", "")
        tasks_content = state.get("tasks_content", "")
        
        # 获取新目录路径
        new_dir = state.get("new_dir", ".")
        
        # 为代码生成创建专门的LLM实例
        code_llm = ChatOllama(
            model="qwen3-coder:30b", 
            temperature=0.2, 
            verbose=True,
            cache=True,  # 启用缓存机制
            streaming=True,  # 启用流式输出
            top_k=50,  # 限制模型只考虑概率最高的50个token
            top_p=0.9,  # 限制模型只考虑累积概率达到0.9的token
            keep_alive=15  # 设置keep_alive为15秒
        )
        
        # 生成可执行代码
        code_prompt = f"""基于以下需求文档、设计文档和任务文档，生成一份可执行的Python代码：

需求文档：{requirements_content}
设计文档：{design_content}
任务文档：{tasks_content}

请生成一个完整的、可运行的Python程序，确保覆盖需求文档中的具体需求点，严格按照设计文档的标准，确保完成任务文档中的每一项任务。
"""
        
        code_response = code_llm.invoke([{"role": "user", "content": code_prompt}])
        code_content = remove_think(code_response.content)
        
        # 保存可执行代码
        code_path = os.path.join(new_dir, "main.py")
        write_file(code_path, code_content)
        
        # 使用LLM生成工作汇报形式的摘要
        response_content = generate_work_report(state, code_content, "可执行代码", llm)
        
        from langchain_core.messages import AIMessage
        response = AIMessage(content=response_content)
        
        # MessagesState 将消息附加到 state 而不是覆盖
        return {"messages": [response], "next": "end"}

    # 首先创建workflow实例
    workflow = StateGraph(CustomState)
    
    # 创建工具节点
    def tool_node(state: CustomState):
        """处理 llm_with_tools 返回的 ToolMessage"""
        tool_calls = state["messages"][-1].tool_calls
        results = []
        print(f"take_action called with tool_calls: {tool_calls}")
        for t in tool_calls:
            print(f"Calling: {t}")
            tool_found = False
            for tool in tools:
                if tool.name == t["name"]:
                    tool_found = True
                    break
            if not tool_found:  # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                # 找到对应的工具并调用
                for tool in tools:
                    if tool.name == t["name"]:
                        result = tool.invoke(t["args"])
                        break
            results.append(
                ToolMessage(tool_call_id=t["id"], content=str(result))
            )
        print("Back to the model!")
        # 获取调用源节点，如果存在的话
        source_node = state.get("source_node")  # 不设置默认值，直接获取state中的source_node
        print(f"Tool called from source node: {source_node}")
        
        # 根据source_node决定下一步执行哪个节点
        next_node = source_node if source_node else "intent_recognition"
        
        # 保持当前状态的其他键值，并添加工具调用结果、源节点信息和下一步节点
        return {**state, "messages": results, "source_node": source_node, "next": next_node}
    
    # 添加所有节点（确保每个节点只添加一次）
    workflow.add_node("intent_recognition", intent_recognition)
    workflow.add_node("generate_requirements", generate_requirements)
    workflow.add_node("generate_design", generate_design)
    workflow.add_node("generate_tasks", generate_tasks)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("tools", tool_node)

    # 添加边
    workflow.add_edge(START, "intent_recognition")
    # 为所有生成节点添加条件边
    workflow.add_conditional_edges(
        "intent_recognition",
        lambda result: result["next"],
        {
            "generate_requirements": "generate_requirements",
            "generate_response": "generate_response",
            "tools": "tools"
        }
    )
    workflow.add_conditional_edges(
        "generate_requirements",
        lambda result: result["next"],
        {
            "tools": "tools"
        }
    )
    workflow.add_conditional_edges(
        "generate_design",
        lambda result: result["next"],
        {
            "tools": "tools"
        }
    )
    workflow.add_conditional_edges(
        "generate_tasks",
        lambda result: result["next"],
        {
            "tools": "tools"
        }
    )
    workflow.add_conditional_edges(
        "generate_code",
        lambda result: result["next"],
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "generate_response",
        lambda result: result["next"],
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("generate_requirements", "generate_design")
    workflow.add_edge("generate_design", "generate_tasks")
    workflow.add_edge("generate_tasks", "generate_code")
    
    # 工具节点由各生成节点在需要工具调用时触发
    
    # 添加从tools节点返回的条件边，根据next返回到对应的节点
    workflow.add_conditional_edges(
        "tools",
        lambda result: result["next"],  # 使用next键决定下一步
        {
            "intent_recognition": "intent_recognition",
            "generate_requirements": "generate_requirements",
            "generate_design": "generate_design",
            "generate_tasks": "generate_tasks",
            "generate_code": "generate_code",
            "generate_response": "generate_response"
        }
    )
    
    workflow.add_edge("generate_response", END)
    # 编译图
    app = workflow.compile()
    return app

def ask(llm_model_name,question):
    """提问"""

    graph = build_graph(llm_model_name)
    for step in graph.stream(
        {"messages": [{"role": "user", "content": question}], "source_node": "intent_recognition"},
            stream_mode="updates",
        ):
            # 首先尝试直接访问 'messages' 键
            if "messages" in step:
                step["messages"][-1].pretty_print()
            else:
                # 如果没有直接找到 'messages' 键，尝试从嵌套字典中提取
                messages_found = False
                for key, value in step.items():
                    if isinstance(value, dict) and "messages" in value:
                        value["messages"][-1].pretty_print()
                        messages_found = True
                        break
                    # 检查更深层次的嵌套
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, dict) and "messages" in sub_value:
                                sub_value["messages"][-1].pretty_print()
                                messages_found = True
                                break
                        if messages_found:
                            break
                # 如果仍未找到 'messages' 键，则打印错误信息
                if not messages_found:
                    print(f"Missing 'messages' key in step: {step}")

def show_graph(graph):
    from PIL import Image as PILImage
    from io import BytesIO
    png_data = graph.get_graph().draw_mermaid_png()
    img = PILImage.open(BytesIO(png_data))
    img.show()


if __name__ == '__main__':
    # 构建图
    graph = build_graph("qwen3:32b")
    show_graph(graph)
    

    query1 = "我需要基于开源大模型，开发一个酒馆战棋实时指导agent"
    
    # 处理查询并在控制台打印结果
    ask("qwen3:32b", query1)