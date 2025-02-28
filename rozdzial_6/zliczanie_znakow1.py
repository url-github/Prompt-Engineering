# Zaimportuj niezbędne klasy i funkcje:
from langchain.agents import AgentExecutor, create_react_agent 
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
# Tworzymy obiekt modelu językowego:
model = ChatOpenAI()
# Funkcja do zliczania znaków w łańcuchu:
def count_characters_in_string(string): 
  return len(string)
# Utwórz listę narzędzi:
# Obecnie dysponujemy tylko jednym narzędziem, które zlicza znaki w łańcuchu znaków. 
tools = [
  Tool.from_function(
    func=count_characters_in_string,
    name="Zlicza znaki w łańcuchu znaków",
    description="Zlicza znaki w łańcuchu znaków",
  ) 
]
# Pobierz prompt reaktywny!
prompt = hub.pull("hwchase17/react") 
# Stwórz agenta reaktywnego:
agent = create_react_agent(model, tools, prompt)
# Zainicjalizuj agenta z określonym zestawem narzędzi
# Utwórz wykonawcę agentów przekazując przy tym agenta i narzędzia:
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# Wywołaj agenta z zapytaniem dotyczącym zliczenia znaków w słowie:
print(agent_executor.invoke({"input": '''Jak wiele znaków znajduje się w słowie "konstantynopolitańczykowianeczka"?'''}))
# 'W słowie "supercalifragilisticexpialidocious" jest X znaków.'

