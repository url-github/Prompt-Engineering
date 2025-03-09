from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
chat = ChatOpenAI(temperature=0.5)
messages = [SystemMessage(content='''Jesteś starszym inżynierem oprogramowania w firmie typu startup.'''),
HumanMessage(content='''Czy możesz przedstawić zabawny żart o inżynierach oprogramowania?''')]
response = chat.invoke(input=messages)
print(response.content)
synchronous_llm_result = chat.batch([messages]*2)
print(synchronous_llm_result)
