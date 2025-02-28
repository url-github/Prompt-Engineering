from langchain.memory import ConversationSummaryMemory, ChatMessageHistory 
from langchain_openai import OpenAI
memory = ConversationSummaryMemory(llm=OpenAI(temperature=0)) 
memory.save_context({"input": "cześć"}, {"output": "co tam?"}) 
print(memory.load_memory_variables({}))
# Zwraca: {'history': '\nThe human greets the AI in Polish. The AI responds by asking how the human is doing.'}
