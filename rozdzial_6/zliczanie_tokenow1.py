import asyncio
from langchain.callbacks import get_openai_callback 
from langchain_core.messages import SystemMessage 
from langchain_openai.chat_models import ChatOpenAI 
model = ChatOpenAI()

with get_openai_callback() as cb: 
  model.invoke([SystemMessage(content="Mam na imię James")])
total_tokens = cb.total_tokens 
print(total_tokens)
# 37
assert total_tokens > 0

with get_openai_callback() as cb: 
  model.invoke([SystemMessage(content="Mam na imię James")]) 
  model.invoke([SystemMessage(content="Mam na imię James")])
assert cb.total_tokens > 0 
print(cb.total_tokens)
# 74

