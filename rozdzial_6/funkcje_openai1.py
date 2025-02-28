# Zaimportuj niezbędne moduły i funkcje z pakietu langchain:
from langchain.chains import ( 
  LLMMathChain,
)
from langchain import hub
from langchain.agents import create_openai_functions_agent, Tool, AgentExecutor 
from langchain_openai.chat_models import ChatOpenAI
# Zainicjalizuj model ChatOpenAI ustawiając temperaturę na 0:
model = ChatOpenAI(temperature=0)
# Utwórz obiekt LLMMathChain korzystając z modelu ChatOpenAI:
llm_math_chain = LLMMathChain.from_llm(llm=model, verbose=True) 
# Pobierz prompt z huba:
prompt = hub.pull("hwchase17/openai-functions-agent")
tools = [
  Tool(
    name="Calculator",
    func=llm_math_chain.run, # uruchom LLMMathChain
    description="przydaje się, gdy chcesz uzyskać odpowiedzi z zakresu matematyki", return_direct=True,
  ), 
]
# Utwórz agenta korzystając z modelu ChatOpenAI i narzędzi:
agent = create_openai_functions_agent(llm=model, tools=tools, prompt=prompt) 
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"input": "Jaki jest wynik działania 5 + 5?"}) 
print(result)
# {'input': Jaki jest wynik działania 5 + 5?', 'output': Odpowiedź: 10'}

