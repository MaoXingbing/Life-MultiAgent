import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc import ABC, abstractmethod
from typing import Optional, Union
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from utills.config_handler import rag_conf

class BaseModelFactory(ABC):
    """模型工厂基类，定义生成模型的通用接口"""
    
    @abstractmethod
    def generater(self) -> Optional[Union[Embeddings, ChatTongyi]]:
        """生成模型实例的抽象方法，子类必须实现"""
        pass

class ChatModelFactory(BaseModelFactory):
    """对话模型工厂，用于生成聊天模型实例"""
    
    def generater(self) -> Optional[Union[Embeddings, ChatTongyi]]:
        # 从配置中获取对话模型名称，创建并返回ChatTongyi实例
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            dashscope_api_key=rag_conf["dashscope_api_key"]
        )
       

class EmbeddingsFactory(BaseModelFactory):
    """嵌入模型工厂，用于生成文本嵌入模型实例"""
    
    def generater(self) -> Optional[Union[Embeddings, ChatTongyi]]:
        # 从配置中获取嵌入模型名称，创建并返回DashScopeEmbeddings实例
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=rag_conf["dashscope_api_key"]
        )
        
#创建实例
chat_model = ChatModelFactory()
embed_model = EmbeddingsFactory()

if __name__ == "__main__":
    print(chat_model.generater())
    print(embed_model.generater())
