from langchain_core.prompts.chat import ( 
  ChatPromptTemplate, 
  SystemMessagePromptTemplate,
)
from langchain_openai.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser 
from pydantic.v1 import BaseModel, Field
from typing import List
temperature = 0.0
class BusinessName(BaseModel):
  name: str = Field(description="Nazwa firmy ") 
  rating_score: float = Field(description='''Ocena firmy. 0 jest najgorsza, 10 jest najlepsza.''')
class BusinessNames(BaseModel):
  names: List[BusinessName] = Field(description='''Lista nazw firm''')
# Ustaw parser i wstrzyknij instrukcje do szablonu promptu:
parser = PydanticOutputParser(pydantic_object=BusinessNames)
principles = """
- Nazwa musi być łatwa do zapamiętania.
- Skorzystaj z branży {industry} i kontekstu firmy, aby stworzyć dobrą nazwę
- Nazwa musi być łatwa do wymówienia.
- Musisz jedynie zwrócić nazwę bez żadnego dodatkowego tekstu lub znaków
- Unikaj zwracania kropek, znaków nowej linii i innych tego typu znaków
- Maksymalna długość nazwy to 10 znaków
"""
# Parser wyjścia modelu czatowego:
model = ChatOpenAI()
template = """Wygeneruj pięć nazw firm dla nowego startupu w branży {industry}
Musisz podążać według następujących zasad: {principles}
{format_instructions}
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
# Tworzenie łańcucha LCEL:
prompt_and_model = chat_prompt | model
result = prompt_and_model.invoke(
   {
       "principles": principles,
       "industry": "Nauka o danych",
       "format_instructions": parser.get_format_instructions(),
} )
# Parser wyjścia przetwarza odpowiedź modelu językowego do obiektu Pydantic:
print(parser.parse(result.content))
