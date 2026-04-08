import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from yaml.loader import Loader
from path_tool import get_abs_path

def load_rag_config(config_path:str=get_abs_path("config/rag.yml"),encoding:str='utf-8'):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_chroma_config(config_path:str=get_abs_path("config/chroma.yml"),encoding:str='utf-8'):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_prompts_config(config_path:str=get_abs_path("config/prompts.yml"),encoding:str='utf-8'):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_agent_config(config_path:str=get_abs_path("config/agent.yml"),encoding:str='utf-8'):
    with open(config_path,"r",encoding=encoding) as f:  
        return yaml.load(f,Loader=yaml.FullLoader)

def load_bge_config(config_path:str=get_abs_path("config/bge.yml"),encoding:str='utf-8'):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
bge_conf = load_bge_config()

if __name__ == '__main__':
    print(f'chat_model_name: {rag_conf["chat_model_name"]}')
    print(f'embedding_model_name: {rag_conf["embedding_model_name"]}')
