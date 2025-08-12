import os
from langchain_core.messages import AIMessage, ToolMessage
from utils.utils import remove_think, write_file
from nodes.work_report_node import generate_work_report


def generate_design(state, llm_with_tool, llm):
    """生成设计文档"""
    
    # 获取需求文档内容
    requirements_content = state.get("requirements_content", "")
    # 获取新目录路径
    new_dir = state.get("new_dir", ".")
    
    # 生成工具调用
    tool_calls = llm_with_tool.invoke([{"role": "user", "content": requirements_content}])
    
    # 输出工具调用结果
    if hasattr(tool_calls, 'tool_calls') and tool_calls.tool_calls:
        return {"next": "tools", "messages": [ToolMessage(content=tool_calls.tool_calls[0].content)], "source_node": "generate_design"}
    
    # 创建 .kiro 目录
    kiro_dir = os.path.join(new_dir, ".kiro")
    if not os.path.exists(kiro_dir):
        os.makedirs(kiro_dir)
    
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
    
    response = AIMessage(content=response_content)
    
    # MessagesState 将消息附加到 state 而不是覆盖
    return {"design_content": design_content, "messages": [response], "new_dir": new_dir, "next": "generate_tasks"}