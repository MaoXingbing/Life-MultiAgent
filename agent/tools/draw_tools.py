import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import requests
from dashscope import ImageSynthesis
import dashscope
from langchain_core.tools import tool
from utills.config_handler import rag_conf
from utills.path_tool import get_abs_path
import base64
from io import BytesIO

dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

@tool(description="一个绘画工具，从文本描述生成图片，返回图片的保存路径")
def draw_image(prompts: str) -> str:
    """
    从文本描述生成图片
    """
    api_key = rag_conf.get("dashscope_api_key", "")
    
    if not api_key:
        return "错误：未配置 dashscope_api_key"
    
    try:
        rsp = ImageSynthesis.call(
            api_key=api_key,
            model="qwen-image-plus",
            prompt=prompts.strip(),
            negative_prompt="",
            n=1,
            size='1328*1328',
            prompt_extend=True,
            watermark=False
        )
        
        if rsp.status_code == HTTPStatus.OK:
            save_dir = get_abs_path("data/images")
            os.makedirs(save_dir, exist_ok=True)
            
            saved_paths = []
            for result in rsp.output.results:
                file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
                file_path = os.path.join(save_dir, file_name)
                
                img_response = requests.get(result.url)
                with open(file_path, 'wb+') as f:
                    f.write(img_response.content)
                
                saved_paths.append(file_path)
            
            return f"图片已生成并保存到: {saved_paths[0]}"
        else:
            return f"生成失败: {rsp.message}"
            
    except Exception as e:
        return f"生成图片时出错: {str(e)}"


def draw_image_with_display(prompts: str, return_base64: bool = False):
    """
    从文本描述生成图片，可选择返回base64用于界面显示
    
    Args:
        prompts: 图片描述
        return_base64: 是否返回base64编码
    
    Returns:
        如果 return_base64=True，返回 (file_path, base64_string)
        否则返回 file_path
    """
    api_key = rag_conf.get("dashscope_api_key", "")
    
    if not api_key:
        raise ValueError("未配置 dashscope_api_key")
    
    rsp = ImageSynthesis.call(
        api_key=api_key,
        model="qwen-image-plus",
        prompt=prompts.strip(),
        negative_prompt="",
        n=1,
        size='1328*1328',
        prompt_extend=True,
        watermark=False
    )
    
    if rsp.status_code == HTTPStatus.OK:
        save_dir = get_abs_path("data/images")
        os.makedirs(save_dir, exist_ok=True)
        
        for result in rsp.output.results:
            file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
            file_path = os.path.join(save_dir, file_name)
            
            img_response = requests.get(result.url)
            img_content = img_response.content
            
            with open(file_path, 'wb+') as f:
                f.write(img_content)
            
            if return_base64:
                base64_str = base64.b64encode(img_content).decode('utf-8')
                return file_path, base64_str
            
            return file_path
    else:
        raise Exception(f"生成失败: {rsp.message}")


if __name__ == '__main__':
    result = draw_image.invoke({"prompts": "画一颗树"})
    print(result)
