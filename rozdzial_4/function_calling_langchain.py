from langchain_core.output_parsers.openai_tools import PydanticToolsParser 
from langchain_core.utils.function_calling import convert_to_openai_tool 
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional
class Article(BaseModel):
  """Przedstawia kluczowe zagadnienia i kontrastujące poglądy w artykule."""
  points: str = Field(..., description="Kluczowe zagadnienia z artykułu")
  contrarian_points: Optional[str] = Field(None, description="Wszelkie kontrastujące poglądy przedstawione w artykule. " )
  author: Optional[str] = Field(None, description="Autor artykułu") 
_EXTRACTION_TEMPLATE = """Wyekstrahuj i zapisz istotne podmioty wspomniane w następującym tekście wraz z ich właściwościami. Jeśli właściwość nie jest obecna i nie jest wymagana w ramach parametrów funkcji, nie dołączaj jej w odpowiedzi."""
# Utwórz prompt, który nakaże modeli wydobycie informacji:
prompt = ChatPromptTemplate.from_messages(
        {("system", _EXTRACTION_TEMPLATE), ("user", "{input}")}
)
model = ChatOpenAI()
pydantic_schemas = [Article]
# Skonwertuj obiekty Pydantica do właściwego schematu:
tools = [convert_to_openai_tool(p) for p in pydantic_schemas] 
# Daj modelowi dostęp do tych narzędzi:
model = model.bind_tools(tools=tools) 
# Utwórz cały łańcucha od początku do końca:
chain = prompt | model | PydanticToolsParser(tools=pydantic_schemas)
result = chain.invoke(
  {
    "input": """W ostatnim artykule zatytułowanym 'Wdrożenia AI w przemyśle' do kluczowych aspektów zaliczono wzrastające zainteresowanie... Jednakże autorka, dr Janina Kowalska, ..."""
  }
)
print(result)
