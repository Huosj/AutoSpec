from langchain_core.messages import AIMessage
from utils.utils import remove_think


def generate_response(state, llm):
    """生成最终响应"""
    
    # 获取用户输入
    user_message = state["messages"][-1].content if state["messages"] else ""
    
    # 生成最终响应
    response_prompt = f"""请基于以下用户输入，生成一个直接、简洁的回答：

用户输入: {user_message}

注意：这不是开发相关的请求，所以不需要生成需求文档、设计文档等开发相关内容。
请直接回答用户的问题，保持回答简洁明了。
"""
    
    response = llm.invoke([{"role": "user", "content": response_prompt}])
    response_content = remove_think(response.content)
    
    return {"messages": [AIMessage(content=response_content)], "next": "end"}