from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint import MemorySaver
from ..config.state import State
from ..nodes.supervisor_node import supervisor_node
from ..nodes.other_node import other_node
from ..nodes.joke_node import joke_node
from ..nodes.couplet_node import couplet_node
from ..nodes.travel_node import travel_node
from ..nodes.draw_node import draw_node


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
    elif current_type == "END":
        return END
    else:
        return 'other_node'


def build_workflow():
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

    # 添加回边，实现业务节点→supervisor_node的数据回传
    builder.add_edge('travel_node', 'supervisor_node')
    builder.add_edge('joke_node', 'supervisor_node')
    builder.add_edge('couplet_node', 'supervisor_node')
    builder.add_edge('other_node', 'supervisor_node')
    builder.add_edge('draw_node', 'supervisor_node')

    # 构建图
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph
