import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from utills.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from utills.path_tool import get_abs_path
from utills.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_simhash, hamming_distance
from utills.logger_handler import logger


class VectorStoreService:
    def __init__(self):
        self.vector_store=Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model.generater(),   #TODO:被修改了
            persist_directory=chroma_conf["persist_directory"],
        )

        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):

        '''
        从数据文件夹内读取数据 转为向量存入向量数据库
        使用simhash值做去重（基于汉明距离判断相似度）
        '''

        SIMILARITY_THRESHOLD = 3

        def check_simhash(simhash_for_check: str) -> bool:
            # 检查simhash存储文件是否存在
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 文件不存在则创建空文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                # 新文件中没有记录，返回False表示不存在相似文档
                return False

            # 打开simhash存储文件（md5_hex_store）进行读取
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                # 逐行读取文件中的所有simhash值
                for line in f.readlines():
                    # 去除行首尾的空白字符
                    line = line.strip()
                    # 跳过空行
                    if not line:
                        continue
                    # 计算当前simhash与文件中simhash的汉明距离
                    distance = hamming_distance(simhash_for_check, line)
                    # 如果汉明距离在阈值范围内，说明存在相似文档
                    if 0 <= distance <= SIMILARITY_THRESHOLD:
                        return True
                # 遍历完所有记录都没有找到相似文档，返回False
                return False

        def save_simhash(simhash_for_save: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(simhash_for_save + "\n")

        def get_file_documents(read_path: str):

            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        for path in allowed_files_path:
            simhash_value = get_file_simhash(path)

            if not simhash_value:
                logger.error(f"file {path} simhash calculation failed, skip")
                continue

            if check_simhash(simhash_value):
                logger.info(f"file {path} already exists (similar document), skip")
                continue

            try:
                documents = get_file_documents(path)

                if not documents:
                    logger.error(f"file {path} load failed, skip")
                    continue

                split_documents: list[Document] = self.splitter.split_documents(documents)

                if not split_documents:
                    logger.error(f"file {path} split failed, skip")
                    continue

                self.vector_store.add_documents(split_documents)

                save_simhash(simhash_value)

                logger.info(f"file {path} load success")
            except Exception as e:
                logger.error(f"file {path} load failed: {str(e)}")
                continue

if __name__ == "__main__":
    vector_store_service=VectorStoreService()
    vector_store_service.load_document()
    retriever=vector_store_service.get_retriever()
    results=retriever.invoke("你好")
    for result in results:
        print(result.page_content)
        print("-"*50)
