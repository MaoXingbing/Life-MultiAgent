'''
总结服务类：用户提问将根据向量数据库中的内容进行总结 交给对话模型进行回复
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from langchain_core.documents import Document
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utills.prompt_loader import load_rag_prompt
from langchain_core.prompts import PromptTemplate
def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store=VectorStoreService()
        self.retriever=self.vector_store.get_retriever()
        self.prompt_text=load_rag_prompt()
        self.prompt_template=PromptTemplate.from_template(self.prompt_text)
        self.model=chat_model.generater()
        self.chain=self._init_chain()
       

    def _init_chain(self):
        return self.prompt_template | print_prompt | self.model | StrOutputParser()

    def retriever_docs(self,query:str)->list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self,query:str)->str:
        context_docs=self.retriever_docs(query)

        context=" "
        counter=0
        for doc in context_docs:
            counter+=1
            context+=f'[参考资料]{counter}:{doc.page_content} | 参考源数据：{doc.metadata}\n'

        return self.chain.invoke(
            {
                "input":query,
                "context":context,
            }
        )

if __name__ == '__main__':
    rag_service=RagSummarizeService()
    print(rag_service.rag_summarize("小户型适合哪些机器人"))
