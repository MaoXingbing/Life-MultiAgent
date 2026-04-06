from langchain_community.chat_models import ChatTongyi
from langchain_core.caches import InMemoryCache

# 构建大模型
llm = ChatTongyi(
    model='qwen-plus',
    api_key='sk-edbd925ffc1f42aea4428b3d995ba30b'
)
