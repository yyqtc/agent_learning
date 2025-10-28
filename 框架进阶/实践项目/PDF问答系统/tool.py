from langchain.tools import tool
from typing import List

import os
import json

config = json.load(open("./config.json", encoding="utf-8"))

@tool
def get_pdf_content_len(pdf_name: str) -> int:
    """
    获取pdf文件内容总字符数

    Args:
        pdf_name: pdf文件名称

    Returns:
        如果pdf文件不存在，则返回0
        如果文件不是pdf文件，则返回0
        如果pdf文件存在，该pdf文件的内容总字符数；
    """
    from PyPDF2 import PdfReader
    
    import os

    pdf_path = os.path.join(config["PDF_DIR_PATH"], pdf_name)

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
def get_pdf_content_with_limit(pdf_name: str, prefix: int,  limit: int) -> str:
    """
    获取PDF文件指定范围内的内容
    
    Args:
        pdf_name: PDF文件名称
        prefix: 内容起始偏移量，最小值为0
        limit: 最大字符数限制，最小值为1
    
    Returns:
        如果pdf文件不存在，则返回"pdf文件不存在"
        如果文件不是pdf文件，则返回"文件不是pdf文件"
        如果pdf文件存在，则返回该PDF文件指定范围内的内容
    """
    from PyPDF2 import PdfReader
    
    import os

    pdf_path = os.path.join(config["PDF_DIR_PATH"], pdf_name)

    if not os.path.exists(pdf_path) or not os.path.isfile(pdf_path):
        return "pdf文件不存在"
    elif not pdf_path.endswith(".pdf"):
        return "文件不是pdf文件"
    else:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            content = ""
            skipped_content_len = 0
            
            # 读取所有页面内容
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_content = page.extract_text()
                if skipped_content_len + len(page_content) < prefix:
                    skipped_content_len += len(page_content)
                    continue
                elif skipped_content_len < prefix and skipped_content_len + len(page_content) > prefix:
                    content += page_content[prefix - skipped_content_len:]
                    skipped_content_len = prefix
                else:
                    content += page_content
                
                if len(content) > limit:
                    content = content[:limit]
                    break
            
            return content

@tool
def get_pdf_list() -> List[str]:
    """
    获取本系统存放的PDF文件名列表。

    Args:
        无

    Returns:
        返回本系统存放的PDF文件名列表。
    """
    import os

    return os.listdir(config["PDF_DIR_PATH"])

@tool
def search_query_history(query: str) -> List[str]:
    """
    获取和query相关的过往用户查询记录。

    Args:
        query: 用户提问的问题

    Returns:
        返回和query相关的过往用户查询对应的系统回答列表，列表长度不超过5。
    """
    from db import search_history

    results = search_history(query)
    return [result.page_content for result in results]


tools = [
    get_pdf_content_len, 
    get_pdf_content_with_limit,
    get_pdf_list,
    search_query_history
]
