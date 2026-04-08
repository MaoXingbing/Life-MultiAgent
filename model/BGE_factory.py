
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from utills.config_handler import bge_conf,rag_conf
from langchain_community.embeddings.dashscope import BGE
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BGEFactory(BaseModelFactory):
    """BGE模型工厂，用于生成BGE模型实例"""
    
    def generater(self)-> Optional[BGE]:
        # 从配置中获取BGE模型名称，创建并返回BGE实例
        return BGE(
            model=bge_conf["bge_model_name"],
            dashscope_api_key=rag_conf["dashscope_api_key"]
        )
        

if __name__ == "__main__":
    bge_mode = BGEFactory().generater()
    print(bge_mode)

