from langchain_openai.chat_models import ChatOpenAI
from langchain.output_parsers.pydantic import PydanticOutputParser 
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic.v1 import BaseModel, Field 
from typing import List
class Query(BaseModel): 
  id: int
  question: str
  dependencies: List[int] = Field(
    default_factory=list,
    description="""Lista podzapytań, które muszą być zakończone przed ukończeniem tego zadania. Używaj podzapytania, gdy cokolwiek jest nieznane, a my możemy mieć potrzebę zadania wielu pytań, w celu uzyskania odpowiedzi. Zależności mogą być tylko innymi zapytaniami."""
  )
class QueryPlan(BaseModel):
  query_graph: List[Query]
# Ustaw model czata:
model = ChatOpenAI() 
# Ustaw parser:
parser = PydanticOutputParser(pydantic_object=QueryPlan)
template = """Wygeneruj plan zapytania w języku polskim. Zostanie on użyty do wykonania zadania.
Odpowiedz na poniższe zapytanie: {query}
Zwróć graf zapytania zgodnie z poniższym formatem:
{format_instructions}
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
# Utwórz łańcuch LCEL zawierający prompt, model i parser:
chain = chat_prompt | model | parser
result = chain.invoke({
"query":'''Chcę uzyskać wyniki z mojej bazy danych. Potem chcę dowiedzieć się, jaki jest średni wiek moich 10 najlepszych klientów. Gdy uzyskam średni wiek tych klientów, chcę wysłać maila do Jana. Poza tym, niezależnie od innych zadań, chciałbym wysłać do Sary maila powitalnego z krótkim wprowadzeniem.''',
"format_instructions":parser.get_format_instructions()})
print(result.query_graph)

