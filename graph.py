from langgraph.graph import Graph
from langgraph.state import MessagesState
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage

from nodes.intent_recognition_node import intent_recognition
from nodes.generate_requirements_node import generate_requirements
from nodes.generate_design_node import generate_design
from nodes.generate_tasks_node import generate_tasks
from nodes.generate_code_node import generate_code
from nodes.work_report_node import generate_work_report
from tools.tools import tool_node
from nodes.generate_response_node import generate_response


class CustomState(TypedDict):
    messages: List[BaseMessage]
    next: str
    new_dir: str
    requirements_content: str
    design_content: str
    tasks_content: str
    code_content: str
    source_node: str


def build_graph(llm_with_tool, llm):
    """构建工作流图"""
    
    # 创建状态图
    graph = Graph(state_schema=CustomState)
    
    # 添加节点
    graph.add_node("intent_recognition", lambda state: intent_recognition(state, llm_with_tool, llm))
    graph.add_node("generate_requirements", lambda state: generate_requirements(state, llm_with_tool, llm))
    graph.add_node("generate_design", lambda state: generate_design(state, llm_with_tool, llm))
    graph.add_node("generate_tasks", lambda state: generate_tasks(state, llm_with_tool, llm))
    graph.add_node("generate_code", lambda state: generate_code(state, llm_with_tool, llm))
    graph.add_node("generate_response", lambda state: generate_response(state, llm))
    graph.add_node("tools", lambda state: tool_node(state, llm))
    
    # 添加边
    graph.add_edge("intent_recognition", "generate_requirements")
    graph.add_edge("intent_recognition", "generate_response")
    graph.add_edge("intent_recognition", "tools")
    graph.add_edge("generate_requirements", "generate_design")
    graph.add_edge("generate_requirements", "tools")
    graph.add_edge("generate_design", "generate_tasks")
    graph.add_edge("generate_design", "tools")
    graph.add_edge("generate_tasks", "generate_code")
    graph.add_edge("generate_tasks", "tools")
    graph.add_edge("generate_code", "generate_response")
    graph.add_edge("generate_code", "tools")
    graph.add_edge("tools", "intent_recognition")
    graph.add_edge("tools", "generate_requirements")
    graph.add_edge("tools", "generate_design")
    graph.add_edge("tools", "generate_tasks")
    graph.add_edge("tools", "generate_code")
    
    # 设置入口点
    graph.set_entry_point("intent_recognition")
    
    # 编译图
    return graph.compile()