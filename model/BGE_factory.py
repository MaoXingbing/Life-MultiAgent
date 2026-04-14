import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
# 配置国内镜像（解决连不上 Hugging Face）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # 关闭警告

from abc import ABC, abstractmethod
from typing import Optional
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from utills.config_handler import bge_conf


class BaseModelFactory(ABC):
    """模型工厂基类"""
    
    @abstractmethod
    def generater(self):
        pass


class BGEFactory(BaseModelFactory):
    """BGE模型工厂，用于生成BGE模型实例"""
    
    def generater(self):
        return HuggingFaceBgeEmbeddings(
            model_name=bge_conf["model_name"],
            model_kwargs=bge_conf["model_kwargs"],
            encode_kwargs=bge_conf["encode_kwargs"]
        )
        

if __name__ == "__main__":
    bge_model = BGEFactory().generater()
    print("BGE Success!")
    print(bge_model)

