# Działamy w ramach łańcucha:
from langchain.memory import ConversationBufferMemory
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
memory = ConversationBufferMemory(return_messages=True)
model = ChatOpenAI(temperature=0)
prompt = ChatPromptTemplate.from_messages(
  [
    ("system", "Działaj jako czatbot, który pomaga użytkownikom z ich zapytaniami."), 
    # Historia konwersacji
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
  ] 
)
chain = ( 
  {
    "input": lambda x: x["input"],
    "history": RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
  }
  | prompt
  | model
  | StrOutputParser()
)

inputs = {"input": "Cześć, mam na imię James!"}
result = chain.invoke(inputs)
memory.save_context(inputs, {"outputs": result})
print(memory.load_memory_variables({}))
# {'history': [HumanMessage(content='Cześć, mam na imię James!'),
# AIMessage(content='Cześć James! Jak mogę Ci dzisiaj pomóc?')]}

inputs = {"input": "Jak mam na imię?"} 
second_result = chain.invoke(inputs) 
print(second_result)
# Masz na imię James.

