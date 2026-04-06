from ..config.llm_config import llm
from ..config.state import State
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
import os
from ..utils.chroma_utils import setup_chroma_db


def couplet_node(state: State):
    print(">>>进入couplet_node节点")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
                 你是一个对联大师，根据用户问题写一个对联，要求对联长度不超过100个字。

                 问题：{input}
                 思考：{agent_scratchpad}
                 """),
        ("user", "{text}"),
    ])

    query = state['message'][0]

    # 读取文件
    path = "py_code/langchain-learning/jlu.txt"
    loader = TextLoader(
        file_path=path,
        encoding="utf-8",
    )
    docs = loader.load()

    # 文件切分成chunk
    split = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    splidocs = split.split_documents(docs)

    # 存储到Chroma
    db = setup_chroma_db(splidocs)

    samples = []
    scored_results = db.similarity_search_with_score(str(query), k=10)
    for document, score in scored_results:
        samples.append(document.page_content)

    prompt = prompt_template.invoke({'samples': samples, 'text': str(query)})
    resp = llm.invoke(prompt)
    return {"message": [HumanMessage(content=resp.content)], "type": "couplet"}
