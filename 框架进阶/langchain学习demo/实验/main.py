'''
借鉴cursor的todo_list概念，使用todo_list控制agent执行过程
比如，拿到一个需求文档夹，需求文档中描述了需要开发的若干内容，包括路由结构、页面内容、api接口、数据库结构
1、读取需求文档夹中的所有md文档，形成向量数据库
2、读取已有项目的api-usage-documentation.md作为已有api调用说明的向量数据库
3、读取已有项目的common-utils-usage-documentation.md作为已有共用工具方法调用说明的向量数据库
4、读取已有项目的storage-usage-documentation.md作为已有项目的存储使用说明的向量数据库
5、读取已有项目的store-usage-documentation.md作为已有项目的共有状态变量及其修改方法的使用说明向量数据库
6、读取已有项目的utils-usage-documentation.md作为已有项目的工具方法的使用说明向量数据库
7、注册9个工具，包括：
    - 读取目录下所有文件列表
    - 读取文件内容
    - 写入文件内容
    - 获取图片url的base64编码
8、初始化todo_list，内容为
    1.读取pages.json文件内容和需求的向量数据库中页面结构相关的向量
    2.形成1的提示词，调用修改pages.json文件内容的agent，同时让agent返回需要开发的路由列表
    3.将需要开发的路由以及页面名称作为任务放入todo_list
    4.如果todo_list不为空，读取todo_list中的第一个任务
    5.调用分析api开发需求的agent，告诉agent需要开发的页面的名称以及路由、页面相关的api调用说明、
'''

from utils import find_file_path, list_dir_items, read_file_content, write_file_content, get_image_base64
from db import init_all_db
from agent import Agent

import json

def main():
    config = json.load(open("./config.json", "r", encoding="utf-8"))
    vector_stores_wrapper = init_all_db()
    tools = [find_file_path, list_dir_items, read_file_content, write_file_content, get_image_base64]
    agent = Agent(tools, vector_stores_wrapper)

    agent.run(
        """
        ### 角色
        你是一位经验丰富的前端开发工程师。

        ### 具体任务
        需要基于需求中的原型页面组成，返回需要开发的页面列表。

        ### 输出格式
        ```python
        [
            {"page_name": "页面1", "page_path": "页面1的路径"},
            {"page_name": "页面2", "page_path": "页面2的路径"},
            {"page_name": "页面3", "page_path": "页面3的路径"},
        ]
        ```
        """
    )

if __name__ == "__main__":
    main()
