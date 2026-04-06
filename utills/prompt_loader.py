import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from path_tool import get_abs_path
from config_handler import prompts_conf
from logger_handler import logger

sys.stdout.reconfigure(encoding='utf-8') 

def load_system_prompt():
    try:
        system_prompt_path=get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error("main_prompt_path not found in prompts.yaml")
        raise e
   
    try:
        return open(system_prompt_path,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"load system_prompt failed: {str(e)}")
        raise e


def load_rag_prompt():
    try:
        rag_prompt_path=get_abs_path(prompts_conf["rag_summarize_path"])
    except KeyError as e:
        logger.error("rag_summarize_path not found in prompts.yaml")
        raise e
   
    try:
        return open(rag_prompt_path,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"load rag_prompt failed: {str(e)}")
        raise e



def load_report_prompt():
    try:
        report_prompt_path=get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error("report_prompt_path not found in prompts.yaml")
        raise e
   
    try:
        return open(report_prompt_path,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"load report_prompt failed: {str(e)}")
        raise e


if __name__ == '__main__':
    print(load_system_prompt())
    print(load_rag_prompt())
    print(load_report_prompt())
