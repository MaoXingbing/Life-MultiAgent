import os
import hashlib
import jieba
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from utills.logger_handler import logger


def get_file_simhash(file_path: str, hash_bits: int = 64):
    if not os.path.exists(file_path):
        logger.error(f'[simhash文件计算]文件不存在: {file_path}')
        return None

    if not os.path.isfile(file_path):
        logger.error(f'[simhash文件计算]文件不是普通文件: {file_path}')
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            logger.error(f'[simhash文件计算]文件内容为空: {file_path}')
            return None

        # 使用jieba对内容进行分词
        words = jieba.cut(content)
        # 初始化向量数组，用于累加各bit位的权重，全0数组
        v = [0] * hash_bits

        # 遍历每个分词
        for word in words:
            if not word.strip():
                continue
            # 计算分词的MD5哈希值，并转换为整数
            word_hash = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            # 遍历每个bit位，根据哈希值更新向量数组
            for i in range(hash_bits):
                # 获取当前bit位的值（0或1），通过右移i位后与1进行按位与操作
                bit = (word_hash >> i) & 1
                # 判断bit位是否为1
                if bit:
                    # bit为1，对应向量位置加1
                    v[i] += 1
                else:
                    # bit为0，对应向量位置减1
                    v[i] -= 1

        # 根据向量数组生成指纹，大于0的bit位设为1
        fingerprint = 0
        for i in range(hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)

        return hex(fingerprint)[2:]

    except Exception as e:
        logger.error(f'[simhash文件计算]文件读取错误: {file_path}，错误: {e}')
        return None


def hamming_distance(hash1: str, hash2: str) -> int:
    # 如果任一哈希值为空，返回-1表示无效
    if not hash1 or not hash2:
        return -1

    try:
        # 将十六进制字符串转换为整数
        h1 = int(hash1, 16)
        # 将第二个十六进制字符串转换为整数
        h2 = int(hash2, 16)
        # 计算两个整数的异或值，不同的位会变为1
        xor = h1 ^ h2
        # 将异或结果转换为二进制字符串，统计其中1的个数即为汉明距离
        distance = bin(xor).count('1')
        # 返回计算得到的汉明距离
        return distance
    except Exception:
        # 转换过程中发生异常时返回-1
        return -1


def get_file_md5_hex(file_path:str):
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