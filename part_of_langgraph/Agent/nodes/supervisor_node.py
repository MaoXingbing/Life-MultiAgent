from ..config.llm_config import llm
from ..config.state import State
from langchain_core.messages import HumanMessage

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
        {'role': 'user',
         'content': state['message'][0].content if hasattr(state['message'][0], 'content') else state['message'][0]}
    ]

    nodes = ['joke', 'travel', 'couplet', 'other', 'draw']
    current_type = state.get('type')
    if current_type in nodes:
        print('已获得type:', current_type)
        return {'type': "END"}
    else:
        response = llm.invoke(prompts)
        typeres = response.content.strip().lower()
        if typeres in nodes:
            return {"type": typeres}
        else:
            return {"type": "other"}
