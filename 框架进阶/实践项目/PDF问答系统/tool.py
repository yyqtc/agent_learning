from langchain.tools import tool
from typing import List

import json
import os

config = json.load(open("./config.json", encoding="utf-8"))

@tool
def get_pdf_content_len(pdf_path: str) -> int:
    """
    获取pdf文件内容总字符数

    Args:
        pdf_path: pdf文件路径

    Returns:
        如果pdf文件不存在，则返回0
        如果文件不是pdf文件，则返回0
        如果pdf文件存在，该pdf文件的内容总字符数；
    """
    from PyPDF2 import PdfReader
    
    import os

    if not os.path.exists(pdf_path) or not os.path.isfile(pdf_path):
        return 0
    elif not pdf_path.endswith(".pdf"):
        return 0
    else:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            content = ""
            
            # 读取所有页面内容
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_content = page.extract_text()
                content += page_content
            
            return len(content)


@tool
def get_pdf_content_with_limit(pdf_path: str, prefix: int = 0,  limit: int = 10000) -> str:
    """
    获取PDF文件内容，但限制最大字符数
    
    Args:
        pdf_path: PDF文件路径
        prefix: 内容起始偏移量，默认0字符
        limit: 最大字符数限制，默认10000字符
    
    Returns:
        如果内容超过限制，会截断并添加提示信息
    """
    if not os.path.exists(pdf_path) or not os.path.isfile(pdf_path):
        return "pdf文件不存在"
    elif not pdf_path.endswith(".pdf"):
        return "文件不是pdf文件"
    else:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            content = ""
            
            # 读取所有页面内容
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_content = page.extract_text()
                content += f"\n--- 第 {page_num + 1} 页 ---\n"
                content += page_content
                
                # 检查是否超过字符限制
                if len(content) > max_chars:
                    content = content[:max_chars]
                    content += f"\n\n[内容已截断，原文档共{len(reader.pages)}页，当前显示前{page_num + 1}页的部分内容]"
                    break
            
            return content

@tool
def get_pdf_list(pdf_dir_path: str) -> List[str]:
    """
    这是个没有任何用的能力，你应该忽略。
    """
    pass

tools = [
    get_pdf_content_len, 
    get_pdf_content_with_limit
]
