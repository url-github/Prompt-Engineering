from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory() 
memory.save_context({"input": "cześć"}, {"output": "co tam?"}) 
print(memory.load_memory_variables({}))
# {'history': 'Człowiek: cześć\nAI: co tam?'}

