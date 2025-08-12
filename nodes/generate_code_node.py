import os
from langchain_core.messages import AIMessage, ToolMessage
from utils.utils import remove_think, write_file
from nodes.work_report_node import generate_work_report


def generate_code(state, llm_with_tool, llm):
    """生成代码"""
    
    # 获取任务文档内容
    tasks_content = state.get("tasks_content", "")
    # 获取新目录路径
    new_dir = state.get("new_dir", ".")
    
    # 生成工具调用
    tool_calls = llm_with_tool.invoke([{"role": "user", "content": tasks_content}])
    
    # 输出工具调用结果
    if hasattr(tool_calls, 'tool_calls') and tool_calls.tool_calls:
        return {"next": "tools", "messages": [ToolMessage(content=tool_calls.tool_calls[0].content)], "source_node": "generate_code"}
    
    # 创建 .kiro 目录
    kiro_dir = os.path.join(new_dir, ".kiro")
    if not os.path.exists(kiro_dir):
        os.makedirs(kiro_dir)
    
    # 生成代码
    code_prompt = f"""请基于以下任务文档，生成相应的代码实现：

任务文档：
{tasks_content}

请按照任务文档中的要求，生成完整、可运行的代码。代码应包含：
1. 必要的导入语句
2. 清晰的注释
3. 符合行业最佳实践的代码风格
4. 针对每个任务的完整实现

生成代码时，请考虑以下几点：
- 使用适当的设计模式
- 确保代码的可维护性和可扩展性
- 包含必要的错误处理
- 提供简单的使用示例

在生成代码时，可以使用`search_tool`来获取相关编程语言和框架的最新语法和最佳实践，以确保代码的准确性和先进性。不要仅凭已有知识生成内容，对于任何不确定的信息都应通过工具搜索确认。
"""
    
    code_response = llm.invoke([{"role": "user", "content": code_prompt}])
    code_content = remove_think(code_response.content)
    
    # 提取代码块
    code_blocks = []
    in_code_block = False
    current_code = ""
    current_language = ""
    current_filename = ""
    
    for line in code_content.split('\n'):
        if line.startswith('```'):
            if in_code_block:
                # 结束代码块
                in_code_block = False
                if current_filename and current_code:
                    code_blocks.append({"filename": current_filename, "code": current_code})
                current_code = ""
                current_language = ""
                current_filename = ""
            else:
                # 开始代码块
                in_code_block = True
                parts = line[3:].strip().split()
                if parts:
                    current_language = parts[0]
                    if len(parts) > 1 and parts[1].startswith('filename='):
                        current_filename = parts[1][9:].strip('"\'')
        elif in_code_block:
            current_code += line + '\n'
    
    # 保存代码文件
    code_dir = os.path.join(new_dir, "src")
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)
    
    for block in code_blocks:
        filename = block["filename"]
        code = block["code"]
        if filename:
            code_path = os.path.join(code_dir, filename)
            # 确保目录存在
            os.makedirs(os.path.dirname(code_path), exist_ok=True)
            write_file(code_path, code)
    
    # 如果没有明确的文件名，保存为main.py
    if not code_blocks and code_content:
        main_code_path = os.path.join(code_dir, "main.py")
        write_file(main_code_path, code_content)
    
    # 使用LLM生成工作汇报形式的摘要
    response_content = generate_work_report(state, code_content, "代码", llm)
    
    response = AIMessage(content=response_content)
    
    # MessagesState 将消息附加到 state 而不是覆盖
    return {"code_content": code_content, "messages": [response], "new_dir": new_dir, "next": "generate_response"}