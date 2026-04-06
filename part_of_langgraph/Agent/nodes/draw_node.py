from ..config.state import State
from langchain_core.messages import HumanMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.dashscope import MultiServerMCP


def travel_node(state: State):
    print(">>>进入travel_node节点")
    DASHSCOPE_API_KEY = "sk-edbd925ffc1f42aea4428b3d995ba30b"
    system_prompt = """
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
            "transport": "streamable_http"
        }
    }

    # 初始化MCP客户端
    mcp_client = MultiServerMCP(
        mcp_servers=MCP_SERVERS,
        dashscope_api_key=DASHSCOPE_API_KEY
    )

    # 获取MCP工具
    mcp_tools = mcp_client.get_tools()

    # 初始化通义千问模型
    llm = ChatTongyi(
        model="qwen-plus",
        dashscope_api_key=DASHSCOPE_API_KEY,
        temperature=0
    )

    # 初始化Agent
    agent = create_react_agent(
        llm=llm,
        tools=mcp_tools,
        prompt=prompt_template
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=mcp_tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # 正确提取用户输入
    user_input = state['message'][0].content if hasattr(state['message'][0], 'content') else state['message'][0]

    resp = agent_executor.invoke(user_input)
    return {"message": [HumanMessage(content=resp["output"])], "type": "travel"}
