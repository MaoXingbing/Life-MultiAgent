import uuid


from Director import graph

config={
        "configurable":{
            "thread_id":uuid.uuid4()
        }
    }


query=input()
res=graph.invoke({'message':[query]}
                 ,config
                 ,stream_mode='values')
print(res['message'][-1].content)