import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma


def setup_chroma_db(splidocs):
    """设置并返回Chroma数据库实例"""
    hashes = [doc.metadata.get('hash', '') for doc in splidocs]
    embedding_model = DashScopeEmbeddings(
        model="text-embedding-v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        hashes=hashes,
    )

    # 存储到Chroma
    db = Chroma.from_documents(
        documents=splidocs,
        embedding=embedding_model,
        persist_directory='py_code/langchain-learning/chromaDB'
    )

    # 去重逻辑
    existing_hashes = set(hashes)
    for doc in splidocs:
        hash_val = doc.metadata.get('hash', '')
        if hash_val not in existing_hashes:
            # 添加新文档
            db.add_texts([doc.page_content], [hash_val])
            existing_hashes.add(hash_val)
            print("添加成功")
        else:
            print("hash重复")

    return db
