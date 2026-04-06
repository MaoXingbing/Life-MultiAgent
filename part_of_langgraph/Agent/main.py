from graph.workflow_builder import build_workflow
from .config.state import State
from langchain_core.messages import HumanMessage


def main():
    graph = build_workflow()

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    # 测试执行
    res = graph.invoke({'message': [HumanMessage(content="画画")]}
                       , config
                       , stream_mode='values')
    print(res['message'][-1].content)


if __name__ == "__main__":
    main()
