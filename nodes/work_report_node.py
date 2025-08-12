from langchain_core.messages import AIMessage
from utils.utils import remove_think


def generate_work_report(state, content, doc_type, llm):
    """生成工作汇报"""
    
    # 获取用户输入
    user_message = state["messages"][-1].content if state["messages"] else ""
    
    # 生成工作汇报
    report_prompt = f"""请基于以下内容，生成一份简洁的工作汇报：

用户需求：{user_message}

生成的{doc_type}内容摘要：{content[:500]}...

请按照以下结构生成工作汇报：
1. 任务完成情况：简要说明{doc_type}已生成
2. 主要内容：简要概述{doc_type}的核心内容
3. 下一步计划：说明接下来的工作

汇报应简洁明了，不要超过300字。
"""
    
    report_response = llm.invoke([{"role": "user", "content": report_prompt}])
    report_content = remove_think(report_response.content)
    
    return report_content