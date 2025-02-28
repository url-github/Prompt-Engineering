from langchain.memory import ConversationSummaryBufferMemory 
from langchain_openai.chat_models import ChatOpenAI
memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=10) 
memory.save_context({"input": "cześć"}, {"output": "co tam?"}) 
print(memory.load_memory_variables({}))
# Zwraca: {'history': 'System: The human greets the AI in Polish by saying "cześć."\nAI: co tam?'}
