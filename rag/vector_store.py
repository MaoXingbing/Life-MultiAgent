import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from utills.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from utills.path_tool import get_abs_path
from utills.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
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
        要计算md5值做去重
        '''


        #检查md5文件是否存在
        def check_md5_hex(md5_for_check:str):

            #如果文件不存在
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                #创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w",encoding="utf-8").close()
                return False
                
            #如果文件存在
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r",encoding="utf-8") as f:
                for line in f.readlines():
                    line=line.strip()
                    if line==md5_for_check:
                        return True
                return False

        def save_md5_hex(md5_for_save:str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a",encoding="utf-8") as f:
                f.write(md5_for_save+"\n")

        def get_file_documents(read_path:str):

            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path:list[str]=listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        for path in allowed_files_path:
            #获取文件的MD5值
            md5_hex=get_file_md5_hex(path)

            #如果文件存在
            if check_md5_hex(md5_hex):
                logger.info(f"file {path} already exists, skip")
                continue

            try:
                documents=get_file_documents(path)

                if not documents:
                    logger.error(f"file {path} load failed, skip")
                    continue
               
                split_documents:list[Document]=self.splitter.split_documents(documents)

                if not split_documents:
                    logger.error(f"file {path} split failed, skip")
                    continue

                #将内容存入向量库
                self.vector_store.add_documents(split_documents)

                #记录这个已经处理好的文件的MD5值，避免下次重复加载
                save_md5_hex(md5_hex)

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
