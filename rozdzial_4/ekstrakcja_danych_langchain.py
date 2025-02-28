from langchain.chains.openai_tools import create_extraction_chain_pydantic 
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
# Upewnij się, że korzystasz z modelu, który obsługuje narzędzia:
model = ChatOpenAI(model="gpt-3.5-turbo-1106") 
class Person(BaseModel):
  """ Imię i wiek osoby."""
  name: str = Field(..., description="Imię osoby")
  age: int = Field(..., description="Wiek osoby ")
chain = create_extraction_chain_pydantic(Person, model)
print(chain.invoke({'input':'''Zbyszek ma 25 lat. Mieszka w Warszawie. Lubi grać w siatkówkę. Helena ma 30 lat. Mieszka w Krakowie. Lubi grać w tenisa.'''}))
