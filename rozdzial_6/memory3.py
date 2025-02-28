from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=1) 
memory.save_context({"input": "cześć"}, {"output": "co tam?"}) 
memory.save_context({"input": "niewiele, a u Ciebie?"}, {"output": "niewiele"}) 
# Zwraca: {'history': 'Human: niewiele, a u Ciebie?\nAI: niewiele'} 
print(memory.load_memory_variables({}))
