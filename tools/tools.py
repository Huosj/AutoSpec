import os
from langchain_core.tools import tool
from baidu_api import ai_search

@tool
def search_tool(query: str) -> str:
    """使用百度搜索获取信息"""
    # 添加监控逻辑，记录输入
    print(f"[SEARCH_TOOL] 输入: {query}")
    result = ai_search(query)
    # 添加监控逻辑，记录输出
    # print(f"[SEARCH_TOOL] 输出: {result}")
    return result