import os
from langchain_core.messages import AIMessage, ToolMessage
from utils.utils import remove_think, write_file
from nodes.work_report_node import generate_work_report


def generate_tasks(state, llm_with_tool, llm):
    """生成任务文档"""
    
    # 获取设计文档内容
    design_content = state.get("design_content", "")
    # 获取新目录路径
    new_dir = state.get("new_dir", ".")
    
    # 生成工具调用
    tool_calls = llm_with_tool.invoke([{"role": "user", "content": design_content}])
    
    # 输出工具调用结果
    if hasattr(tool_calls, 'tool_calls') and tool_calls.tool_calls:
        return {"next": "tools", "messages": [ToolMessage(content=tool_calls.tool_calls[0].content)], "source_node": "generate_tasks"}
    
    # 创建 .kiro 目录
    kiro_dir = os.path.join(new_dir, ".kiro")
    if not os.path.exists(kiro_dir):
        os.makedirs(kiro_dir)
    
    # 生成任务文档
    tasks_prompt = f"""请基于以下设计文档，生成详细的开发任务列表：

设计文档：
{design_content}

请按照以下结构生成任务文档：

# 开发任务列表

## 前期准备
- [ ] 环境搭建
- [ ] 依赖安装
- [ ] 项目初始化

## 核心功能开发
[根据设计文档中的架构和功能模块，分解为具体的开发任务]
每个任务应包含：
1. 任务描述
2. 任务优先级（高/中/低）
3. 预计工时
4. 依赖任务（如果有）

## 测试
- [ ] 单元测试
- [ ] 集成测试
- [ ] 系统测试

## 部署
- [ ] 打包
- [ ] 部署配置
- [ ] 上线验证


在生成任务时，可以使用`search_tool`来获取相关开发技术的最佳实践，以确保任务的合理性和可行性。不要仅凭已有知识生成内容，对于任何不确定的信息都应通过工具搜索确认。

注意：任务应具有原子性和可执行性，每个任务都应是离散、可管理的编码步骤，并明确关联到设计文档中的具体模块。
"""
    
    tasks_response = llm.invoke([{"role": "user", "content": tasks_prompt}])
    tasks_content = remove_think(tasks_response.content)
    
    # 保存任务文档
    tasks_path = os.path.join(kiro_dir, "tasks.md")
    write_file(tasks_path, tasks_content)
    
    # 使用LLM生成工作汇报形式的摘要
    response_content = generate_work_report(state, tasks_content, "任务文档", llm)
    
    response = AIMessage(content=response_content)
    
    # MessagesState 将消息附加到 state 而不是覆盖
    return {"tasks_content": tasks_content, "messages": [response], "new_dir": new_dir, "next": "generate_code"}