from langchain_openai import ChatOpenAI 
from langchain_core.tools import tool
# 1. Utwórz model:
llm = ChatOpenAI(temperature=0)
@tool
def get_word_length(word: str) -> int: 
  """Zwraca długość słowa.""" 
  return len(word)
# 2. Utwórz narzędzia:
tools = [get_word_length]

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 3. Utwórz prompt:
prompt = ChatPromptTemplate.from_messages(
   [
    (
      "system",
      """Jesteś bardzo potężnym asystentem, ale nie znasz bieżących wydarzeń i nie umiesz szacować długości słów""", 
    ),
    ("user", "{input}"),
    # W tym miejscu czyta i zapisuje komunikaty
    MessagesPlaceholder(variable_name="agent_scratchpad"),
  ] 
)

from langchain_core.utils.function_calling import convert_to_openai_tool 
from langchain.agents.format_scratchpad.openai_tools import (
  format_to_openai_tool_messages,
)
# 4. Sformatuj narzędzia-funkcje Pythona do postaci schematu JSON i powiąż je z modelem
llm_with_tools = llm.bind_tools(tools=[convert_to_openai_tool(t) for t in tools])
from langchain.agents.output_parsers.openai_tools  import OpenAIToolsAgentOutputParser
# 5. Konfiguracja łańcucha agentów:
agent = ( 
  {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_tool_messages(
      x["intermediate_steps"]
    ),
  }
  | prompt
  | llm_with_tools
  | OpenAIToolsAgentOutputParser()
)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) 
agent_executor.invoke({"input": "Ile liter jest w słowie oprogramowanie?"}) 
#{'input': 'Ile liter jest w słowie oprogramowanie?',
# 'output': 'W słowie oprogramowanie jest 14 liter.'}

