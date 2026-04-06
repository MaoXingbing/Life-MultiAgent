from ..config.state import State
from langchain_core.messages import HumanMessage

def other_node(state: State):
    print(">>>进other_node节点")
    return {"message": [HumanMessage(content="我暂时无法回答这个问题")], "type": "other"}
