from langchain.memory import ConversationTokenBufferMemory 
from langchain_openai.chat_models import ChatOpenAI
memory = ConversationTokenBufferMemory(llm=ChatOpenAI(), max_token_limit=50)
memory.save_context({"input": "cześć"}, {"output": "co tam?"})
print(memory.load_memory_variables({}))
# Zwraca: {'history': 'Human: cześć\nAI: co tam?'}
