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

def google_search(query: str) -> str: 
  return "James Phoenix ma 31 lat."
# Lista narzędzi, z których może skorzystać agent:
tools = [
  Tool(
    # Narzędzie LLMMathChain do obliczeń matematycznych.
    func=llm_math_chain.run,
    name="Calculator",
    description="przydaje się, gdy chcesz uzyskać odpowiedzi z zakresu matematyki ",
  ), 
  Tool(
    # Narzędzie do zliczania znaków w tekście
    func=google_search,
    name="google_search",
    description=" przydaje się, gdy chcesz uzyskać odpowiedzi z zakresu matematyki.",
  ), 
]
# Utwórz agenta korzystając z modelu ChatOpenAI i narzędzi:
agent = create_openai_functions_agent(llm=model, tools=tools, prompt=prompt) 
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# Poproś agenta o uruchomienie zadania i przechowanie jego wyniku:    
result = agent_executor.invoke(
  {
    "input": """Task: Wyszukaj informacje w wyszukiwarce Google na temat wieku Jamesa Phoenix'a. Następnie podnieś wiek do kwadratu."""
  }
)
print(result)
# {'input': "...", 'output': 'James Phoenix ma 31 lat.
# Po podniesieniu jego wieku do kwadratu uzyskujemy 961.'}

