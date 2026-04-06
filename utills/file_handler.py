import os
import hashlib
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from logger_handler import logger


def get_file_md5_hex(file_path:str):  #获取文件的md5哈希值
    if not os.path.exists(file_path):
        logger.error(f'[md5文件计算]文件不存在: {file_path}')
        return
    
    if not os.path.isfile(file_path):
        logger.error(f'[md5文件计算]文件不是普通文件: {file_path}')
        return

    md5_obj = hashlib.md5()
    chunk_size = 4096 # 4KB分片
    try:
        with open(file_path,"rb") as f:
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)  # ✓ 持续更新
        
        # ✓ 所有 chunk 处理完后，最后计算一次
        md5_hex=md5_obj.hexdigest()
        return md5_hex
    except Exception as e:
        logger.error(f'[md5文件计算]文件读取错误: {file_path}，错误: {e}')
        return None


#返回文件夹中的文件        
def listdir_with_allowed_type(path:str,allowed_types:tuple[str]):
    '''
    返回文件夹中的文件，只返回指定类型的文件
    ：param path 文件夹路径
    ：param allowed_types 允许的文件类型
    ：return 允许类型的文件列表
    '''
    files = []
    if not os.path.exists(path):
        logger.error(f'[文件列表]文件夹不存在: {path}')
        return files

    if not os.path.isdir(path):
        logger.error(f'[文件列表]文件夹不是目录: {path}')
        return allowed_types

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path,f))
    
    return tuple(files)

def pdf_loader(file_path:str,passwd=None)->list[Document]:
    '''
    加载pdf文件
    ：param file_path pdf文件路径
    ：return 加载后的文档列表
    '''
    return PyPDFLoader(file_path,passwd).load()


def txt_loader(file_path:str)->list[Document]:
    '''
    加载文本文件
    ：param file_path 文本文件路径
    ：return 加载后的文档列表
    '''
    return TextLoader(file_path,encoding="utf-8").load()