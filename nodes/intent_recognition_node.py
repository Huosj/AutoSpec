from langchain_core.messages import AIMessage
from langgraph.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_ollama import ChatOllama
from AutoSpec.utils.utils import remove_think
from AutoSpec.tools.tools import search_tool


def intent_recognition(state, llm_with_tool, llm):
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
        project_name_prompt = f"请基于以下用户需求生成一个简洁的英文驼峰命名的项目文件夹名称（只返回文件夹名称，不要包含任何其他内容）：

{user_message}"
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
        response = AIMessage(content=f"已创建项目目录 {new_dir} 并开始生成需求文档。")
        # 直接返回扁平化结果，避免嵌套字典
        return {"next": "generate_requirements", "new_dir": new_dir, "messages": [response]}
    else:
        # 如果不是开发相关，直接生成回答
        response = AIMessage(content="您的请求不是开发相关的，我将直接回答您的问题。")
        # 直接返回扁平化结果，避免嵌套字典
        return {"next": "generate_response", "messages": [response]}