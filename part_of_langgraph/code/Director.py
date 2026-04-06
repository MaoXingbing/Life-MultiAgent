from typing import Annotated, TypedDict, List

from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.caches import InMemoryCache
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from operator import add

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint import MemorySaver
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import requests
from dashscope import ImageSynthesis
import os
import dashscope
from langchain_community.tools.dashscope import MultiServerMCP
from ollama import embeddings

# 构建大模型
llm = ChatTongyi(
    model='qwen-plus',
    api_key='sk-edbd925ffc1f42aea4428b3d995ba30b'
)


class State(TypedDict):
    message: Annotated[list[AnyMessage], add]
    type: str


def supervisor_node(state: State):
    print(">>>进入supervisor_node节点")
    prompt = '''一个专业的客服助手负责对用户的问题进行分类，并将任务分给其他 agent 执行。
           如果用户的问题适合旅游路线规划相关的，那就返回 travel。
           如果用户的问题是希望找一个笑话，那就返回 joke。
           如果用户的问题是希望对一个对联，那就返回 couplet。
           如果用户的问题是希望画画，那就返回 draw。
           如果是其他的问题，返回 other，除了这几个选项外，不要返回任何其他的内容。
           '''
    prompts = [
        {'role': 'system', 'content': prompt},
        # 修复：取最新消息（兼容多轮），且如果是HumanMessage，取content
        {'role': 'user',
         'content': state['message'][0].content if hasattr(state['message'][0], 'content') else state['message'][0]}
    ]

    nodes = ['joke', 'travel', 'couplet', 'other', 'draw']
    # 修复1：优化判断逻辑，仅当type是业务类型时，返回终止标识
    current_type = state.get('type')
    if current_type in nodes:
        print('已获得type:', current_type)
        return {'type': "END"}  # 修复2：返回字符串"END"，匹配路由函数判断
    else:
        # 模型调用
        response = llm.invoke(prompts)
        typeres = response.content.strip().lower()  # 格式化结果，避免多余空格/大小写问题
        if typeres in nodes:
            return {"type": typeres}
        else:
            # 异常时默认返回other，避免报错中断
            return {"type": "other"}


def other_node(state: State):
    print(">>>进other_node节点")
    return {"message": [HumanMessage(content="我暂时无法回答这个问题")], "type": "other"}


def couplet_node(state: State):
    print(">>>进入couplet_node节点")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system","""
                 你是一个对联大师，根据用户问题写一个对联，要求对联长度不超过100个字。
    
                 问题：{input}
                 思考：{agent_scratchpad}
                 """),
        ("user", "{text}"),
    ])

    # 用户问题
    query=state['message'][0]

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

    #针对splidocs生成hash值
    hashes = [doc.metadata['hash'] for doc in splidocs]
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

    #通过hash进行去重
    #如果hash重复 不进行添加
    #如果hash不重复 进行添加
    for doc in splidocs:
        if doc.metadata['hash'] not in hashes:
            splidocs.append(doc)
            hashes.append(doc.metadata['hash'])
            embedding_model.add_texts([doc.page_content], [doc.metadata['hash']])
            db.add_texts([doc.page_content], [doc.metadata['hash']])
            db.persist()
            print("添加成功")
        else:
            print("hash重复")

    #


    samples=[]
    scored_results=db.similarity_search_with_score(query,k=10)
    for document, score in scored_results:
        samples.append(document.page_content)
    prompt=prompt_template.invoke({'samples': samples, 'text': query})
    resp=llm.invoke(prompt)
    return {"message": [HumanMessage(content=resp.content)], "type": "couplet"}


def joke_node(state: State):
    print(">>>进入joke_node节点")
    system_prompt = '你是一个笑话大师，根据用户的问题写一个不超过100个字的笑话'

    prompts = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': state['message'][0]}
    ]
    response = llm.invoke(prompts)
    return {"message": [HumanMessage(content=response.content)], "type": "joke"}


def travel_node(state: State):
    print(">>>进入travel_node节点")
    DASHSCOPE_API_KEY = "sk-edbd925ffc1f42aea4428b3d995ba30b"
    system_prompt="""
         你是一个地理信息助手，只能使用提供的工具解决问题，步骤如下：
         1. 分析用户问题，判断是否需要调用高德地图工具（amap-maps）；
         2. 若需要，严格按工具描述传参，调用工具获取结果；
         3. 用工具返回的结果回答用户问题，不要编造信息。
     
         问题：{input}
         思考：{agent_scratchpad}
         """

    prompt_template = PromptTemplate.from_template(system_prompt)

    # 高德地图的MCP配置信息
    MCP_SERVERS = {
        "amap-maps": {
            "type": "sse",
            "description": "高德地图MCP Server现已覆盖12大核心接口，提供全场景覆盖的地理信息服务，包括地理编码、逆地理编码、IP定位、天气查询、骑行路径规划、步行路径规划、驾车路径规划、公交路径规划、距离测量、关键词搜索、周边搜索、详情搜索等。",
            "isActive": True,
            "name": "阿里云百炼_Amap Maps",
            "baseUrl": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
            "headers": {
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
            },
            "transport": "streamable_http"  # 现成的SSE传输层（框架内置，无需手动写）  # 降级为普通HTTP，避开SSE流式问题
        }
    }

    # 3. 初始化现成的MCP客户端（框架封装了所有SSE/HTTP底层逻辑）
    mcp_client = MultiServerMCP(
        mcp_servers=MCP_SERVERS,
        dashscope_api_key=DASHSCOPE_API_KEY  # 自动处理鉴权、SSE连接、MCP协议封装
    )

    # 4. 自动获取MCP工具（框架自动解析配置，生成LangChain Tool，无需手动写调用函数）
    mcp_tools = mcp_client.get_tools()

    # 5. 初始化通义千问模型（阿里云百炼生态现成模型）
    llm = ChatTongyi(
        model="qwen-plus",  # 百炼主流模型，也可换qwen-turbo/qwen-max
        dashscope_api_key=DASHSCOPE_API_KEY,
        temperature=0  # 工具调用建议固定温度
    )


    # # 6. 定义Agent Prompt（ReAct框架，和你图片里的Agent类型一致）
    # REACT_PROMPT = PromptTemplate.from_template(
    #     """
    #     你是一个地理信息助手，只能使用提供的工具解决问题，步骤如下：
    #     1. 分析用户问题，判断是否需要调用高德地图工具（amap-maps）；
    #     2. 若需要，严格按工具描述传参，调用工具获取结果；
    #     3. 用工具返回的结果回答用户问题，不要编造信息。
    #
    #     问题：{input}
    #     思考：{agent_scratchpad}
    #     """
    # )

    # 7. 初始化Agent（现成的ReAct Agent，无需手动绑定工具逻辑）
    agent = create_react_agent(
        llm=llm,
        tools=mcp_tools,
        prompt=prompt_template
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=mcp_tools,
        verbose=True,  # 打印调用过程，便于调试
        handle_parsing_errors=True  # 自动处理工具调用格式错误
    )

    prompts = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': state['message'][0]}
    ]

    # 正确写法（纯文本字符串）:将state中的结构数据，即用户提问，进行拼接成字符串
    user_input = state['message'][0].content if hasattr(state['message'][0], 'content') else state['message'][0]

    resp=agent_executor.invoke(user_input)
    # return {"message": [HumanMessage(content=['message'][-1].content)], "type": "draw"}
    # 修改 [travel_node](file:///wsl.localhost/Ubuntu/home/maoxinbing/MyAgent/code/Director.py#L81-L143) 函数末尾的返回语句
    return {"message": [HumanMessage(content=resp["output"])], "type": "travel"}

def draw_node(state: State):
    print('>>>进入draw_node节点')

    # 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
    system_prompt = '你是一个绘画大师，根据用户的问题画一张动漫风格的画'
    prompts = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': state['message'][0]}
    ]
    # 正确写法（纯文本字符串）
    user_input = state['message'][0].content if hasattr(state['message'][0], 'content') else state['message'][0]
    # 拼接图像描述提示（纯文本）
    image_prompt = f"动漫风格，{user_input}"  # 明确动漫风格，符合你的需求

    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key = "sk-edbd925ffc1f42aea4428b3d995ba30b"

    print('----同步调用，请等待任务执行----')
    rsp = ImageSynthesis.call(api_key=api_key,
                              model="qwen-image-plus",  # qwen-image-max、qwen-image-max-2025-12-30模型不支持异步接口
                              prompt=image_prompt,
                              negative_prompt="",
                              n=1,
                              size='1328*1328',
                              prompt_extend=True,
                              watermark=False)
    print(f'response: {rsp}')
    if rsp.status_code == HTTPStatus.OK:
        # 在当前目录下保存图像
        for result in rsp.output.results:
            file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
            with open('./%s' % file_name, 'wb+') as f:
                f.write(requests.get(result.url).content)
        return {"message": [HumanMessage(content="绘画成功")], "type": "draw"}
    else:
        print(f'同步调用失败, status_code: {rsp.status_code}, code: {rsp.code}, message: {rsp.message}')
        return {"message": [HumanMessage(content="绘画失败")], "type": "draw"}


# 条件路由
def routing_func(state: State):
    current_type = state['type']
    if current_type == 'travel':
        return 'travel_node'
    elif current_type == 'joke':
        return 'joke_node'
    elif current_type == 'couplet':
        return 'couplet_node'
    elif current_type == 'draw':
        return 'draw_node'
    elif current_type == "END":  # 识别字符串"END"
        return END  # 返回LangGraph内置END常量，触发终止
    else:
        return 'other_node'


# 构建图
builder = StateGraph(State)

# 添加节点
builder.add_node('supervisor_node', supervisor_node)
builder.add_node('travel_node', travel_node)
builder.add_node('joke_node', joke_node)
builder.add_node('couplet_node', couplet_node)
builder.add_node("other_node", other_node)
builder.add_node("draw_node", draw_node)

# 添加边
builder.add_edge(START, 'supervisor_node')

builder.add_conditional_edges(
    'supervisor_node',
    routing_func,
    ['travel_node', 'joke_node', 'couplet_node', "other_node", 'draw_node', END]
)
# 保留回边，实现业务节点→supervisor_node的数据回传
builder.add_edge('travel_node', 'supervisor_node')
builder.add_edge('joke_node', 'supervisor_node')
builder.add_edge('couplet_node', 'supervisor_node')
builder.add_edge('other_node', 'supervisor_node')
builder.add_edge('draw_node', 'supervisor_node')

# 构建图
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 执行任务测试的代码
if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    # 修复4：规范初始消息格式，传入HumanMessage对象
    # for chunk in graph.stream({"message":[HumanMessage(content="给我讲一个笑话")]}
    #               ,config
    #               ,stream_mode="custom"):
    #     print(chunk)

    res = graph.invoke({'message': ["画画"]}
                       , config
                       , stream_mode='values')
    print(res['message'][-1].content)
