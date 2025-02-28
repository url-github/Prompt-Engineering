from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate 
from langchain.prompts.example_selector import LengthBasedExampleSelector 
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage
import tiktoken
examples = [
    {"input": "Gollum", "output": "Historyjka na temat Golluma."},
    {"input": "Gandalf", "output": "Historyjka na temat Gandalfa."},
    {"input": "Bilbo", "output": "Historyjka na temat Bilbo."},
]
story_prompt = PromptTemplate( 
  input_variables=["input", "output"], 
  template="Postać: {input}\nHistoryjka: {output}",
)
def num_tokens_from_string(string: str) -> int: 
  """Zwraca liczbę tokenów w łańcuchu tekstowym. """ 
  encoding = tiktoken.get_encoding("cl100k_base") 
  num_tokens = len(encoding.encode(string))
  return num_tokens
example_selector = LengthBasedExampleSelector(
  examples=examples,
  example_prompt=story_prompt,
  max_length=1000, # w ramach przykładów skorzystaj maksymalnie z 1000 tokenów 
  # get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x)) 
  # Zmodyfikowaliśmy metodę get_text_length, aby działała z biblioteką TikToken na podstawie liczby tokenów:
  get_text_length=num_tokens_from_string,
)
dynamic_prompt = FewShotPromptTemplate(
  example_selector=example_selector,
  example_prompt=story_prompt,
  prefix='''Wygeneruj historyjkę dla postaci {character} korzystając ze wszystkich historyjek wszystkich postaci jako kontekstu.''',
  suffix="Postać: {character}\nHistoryjka:",
  input_variables=["character"],
)
# Sprawdźmy nową postać z Władcy Pierścieni:
formatted_prompt = dynamic_prompt.format(character="Frodo") 
# Tworzymy model czatu:
chat = ChatOpenAI()
response = chat.invoke([SystemMessage(content=formatted_prompt)])
print(response.content)

