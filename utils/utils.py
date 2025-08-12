import os
import re


def remove_think(text):
    # 匹配 superscript: 标签及其内容
    pattern = r"<RichMediaReference>.*?</RichMediaReference>"
    # 替换为空字符串
    return re.sub(pattern, "", text, flags=re.DOTALL)


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