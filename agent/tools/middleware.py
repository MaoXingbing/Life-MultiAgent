from typing import Callable
from httpx import request
from langchain.agents import AgentState
from langchain.agents.middleware import ModelRequest, before_model, dynamic_prompt, wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.runtime import Runtime
from utills.logger_handler import logger
from utills.prompt_loader import load_system_prompt,load_report_prompt

@wrap_tool_call
def monitor_tool(
    #请求的数据封装
    request:ToolCallRequest,
    #执行的函数本身
    handler:Callable[[ToolCallRequest],ToolMessage | Command],
)->ToolMessage | Command:
   logger.info(f'[tool monitor]执行工具 {request.tool_call["name"]}')
   logger.info(f'[tool monitor]传入参数：{request.tool_call["args"]}')

   try:
    result=handler(request)
    logger.info(f'[tool monitor]工具 {request.tool_call["name"]} 调用成功')

    if request.tool_call["name"]=="fill_context_for_report":
        request.runtime.context["report"]=True   #若使用fill_context_for_report工具，设置report为True
        
    return result
   except Exception as e:
    logger.error(f'[tool monitor]执行工具 {request.tool_call["name"]} 调用失败，原因：{str(e)}')
    raise e

@before_model
def log_before_model(
    state:AgentState,
    runtime:Runtime
):
    logger.info(f"[log before model]即将调用模型，带有{len(state['messages'])}条消息")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None


#动态切换提示词
@dynamic_prompt #每一次在生成提示词之前调用此函数
def report_prompt_switch(request:ModelRequest):
    is_report = request.runtime.context.get("report",False)
    #检测到使用了fill_context_for_report工具，是报告生成场景 返回报告提示词
    if is_report:   
        return load_report_prompt()
        
    return load_system_prompt()

  
