from typing import List, Dict, Any
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

import os
import base64
import requests
import json

# 全局变量存储向量数据库
vector_stores = {}

@tool
def read_directory_files(directory_path: str) -> List[str]:
    """
    读取指定目录下的所有文件列表
    
    Args:
        directory_path: 目录路径
        
    Returns:
        文件路径列表
    """
    try:
        if not os.path.exists(directory_path):
            return []
        
        files = []
        for root, dirs, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                files.append(file_path)
        return files
    except Exception as e:
        print(f"读取目录文件时出错: {e}")
        return []

@tool
def read_file_content(file_path: str) -> str:
    """
    读取文件内容
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件内容字符串
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"读取文件内容时出错: {e}")
        return f"读取文件内容时出错: {e}"

@tool
def write_file_content(file_path: str, content: str) -> bool:
    """
    写入文件内容
    
    Args:
        file_path: 文件路径
        content: 要写入的内容
        
    Returns:
        是否写入成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"写入文件内容时出错: {e}")
        return False

@tool
def get_image_base64(image_path: str) -> str:
    """
    获取图片的base64编码（支持本地文件和网络URL）
    
    Args:
        image_path: 图片路径（本地文件路径或网络URL）
        
    Returns:
        base64编码的图片数据
    """
    try:
        # 判断是本地文件还是网络URL
        if image_path.startswith(('http://', 'https://')):
            # 网络URL
            response = requests.get(image_path)
            response.raise_for_status()
            image_data = response.content
            content_type = response.headers.get('content-type', 'image/jpeg')
        else:
            # 本地文件
            if not os.path.exists(image_path):
                return f"本地图片文件不存在: {image_path}"
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # 根据文件扩展名确定content_type
            ext = os.path.splitext(image_path)[1].lower()
            content_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp',
                '.svg': 'image/svg+xml'
            }
            content_type = content_type_map.get(ext, 'image/jpeg')
        
        # 获取图片的base64编码
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # 返回data URL格式
        return f"data:{content_type};base64,{image_base64}"
    except Exception as e:
        print(f"获取图片base64编码时出错: {e}")
        return f"获取图片base64编码时出错: {e}"
