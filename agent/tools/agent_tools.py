import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import random
from utills.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utills.config_handler import agent_conf
from utills.path_tool import get_abs_path

user_ids=["123456","789012","345678","901234","567890"]
months_arr=["2025-01","2025-02","2025-03","2025-04","2025-05","2025-06","2025-07","2025-08","2025-09","2025-10","2025-11","2025-12",""]
external_data={}

rag=RagSummarizeService()

@tool(description="从向量数据库中检索参考资料")
def rag_summarize(query:str):
    return rag.rag_summarize(query)

@tool(description="获取城市的天气信息，以字符串格式返回")
def get_weather(city:str)->str:
    """
    获取城市的天气信息
    """
    return f"{city}的天气是晴朗的,气温25摄氏度，湿度60%，最近2小时无雨"


@tool(description="获取用户的所在城市，以字符串格式返回")
def get_user_location()->str:
    return random.choice(["北京","上海","广州","深圳"])

@tool(description="获取用户的ID，以字符串格式返回")
def get_user_id()->str:
    return random.choice(user_ids)

@tool(description="获取当前月份，以字符串格式返回")
def get_current_month()->str:
    return random.choice(months_arr)

@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回，如果未检索到则返回空字符串")
def fetch_external_data(user_id:str,month:str)->str:
    pass

def generate_external_data():
    if not external_data:
        external_data_path=get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件不存在：{external_data_path}")

        with open(external_data_path,"r",encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr:list[str]=line.strip().split(",")

                user_id:str=arr[0].replace('"',"")
                feature:str=arr[1].replace('"',"")
                efficiency:str=arr[2].replace('"',"")
                consumables:str=arr[3].replace('"',"")
                comparison:str=arr[4].replace('"',"")  
                time:str=arr[5].replace('"',"")

                if user_id not in external_data:
                    external_data[user_id]={}
                
                external_data[user_id][time]={
                    "特征":feature,
                    "效率":efficiency,
                    "消耗":consumables,
                    "对比":comparison,
                }

@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回，如果未检索到则返回空字符串")
def fetch_external_data(user_id:str,month:str)->str:
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户{user_id}在{month}月的使用记录")
        return ""

@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return 'fill_context_for_report()已调用'
                 
if __name__ == '__main__':
    print(fetch_external_data("123456","2025-02"))
