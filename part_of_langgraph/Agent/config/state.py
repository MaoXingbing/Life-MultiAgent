from typing import Annotated, TypedDict, List
from langchain_core.messages import AnyMessage
from operator import add

class State(TypedDict):
    message: Annotated[list[AnyMessage], add]
    type: str
