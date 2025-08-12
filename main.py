import os
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

from graph import build_graph
from tools.tools import search_tool
from utils.utils import read_file


# 初始化缓存
set_llm_cache(InMemoryCache())

# 初始化LLM模型
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.7,
    streaming=True,
)

# 创建带有工具的LLM
llm_with_tool = llm.bind_tools([search_tool])

# 构建工作流图
graph = build_graph(llm_with_tool, llm)


def ask(user_input):
    """处理用户输入并返回响应"""
    initial_state = {
        "messages": [{"role": "user", "content": user_input}],
        "next": "",
        "new_dir": ".",
        "requirements_content": "",
        "design_content": "",
        "tasks_content": "",
        "code_content": "",
        "source_node": ""
    }
    
    # 运行图
    result = graph.invoke(initial_state)
    
    # 获取最后一条消息作为响应
    if result and "messages" in result and result["messages"]:
        return result["messages"][-1].content
    return "抱歉，无法生成响应。"


def show_graph():
    """显示工作流图"""
    try:
        from langgraph.graph import get_graph_image
        image = get_graph_image(graph)
        image.show()
    except Exception as e:
        print(f"无法显示图形: {e}")


if __name__ == "__main__":
    print("AutoSpec Agent 已启动！")
    print("输入 'exit' 退出程序，输入 'show graph' 查看工作流图。")
    
    # 示例查询
    print("\n示例查询:\n")
    example_query = "创建一个简单的待办事项应用"
    print(f"用户: {example_query}")
    response = ask(example_query)
    print(f"Agent: {response}")
    
    # 交互式对话
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == 'exit':
            print("程序已退出。")
            break
        elif user_input.lower() == 'show graph':
            show_graph()
        else:
            response = ask(user_input)
            print(f"Agent: {response}")