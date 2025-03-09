from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import (SystemMessagePromptTemplate, ChatPromptTemplate)
template = """
Jesteś kreatywnym konsultantem, który wymyśla nazwy firm.
Musisz spełniać następujące reguły: 
{principles}
Wygeneruj listę numerowaną pięciu chwytliwych nazw dla startupu w branży {industry}, które muszą radzić sobie z {context}?
Oto przykład formatu danych: 
1. Nazwa1
2. Nazwa2
3. Nazwa3
4. Nazwa4
5. Nazwa5
"""
model = ChatOpenAI()
system_prompt = SystemMessagePromptTemplate.from_template(template)
chat_prompt = ChatPromptTemplate.from_messages([system_prompt])
chain = chat_prompt | model
result = chain.invoke({
    "industry": "medyczna",
    "context":'''tworzenie rozwiązań AI związanych z automatycznym podsumowywaniem danych pacjentów''',
    "principles":'''1. Każda nazwa powinna być krótka i łatwa do zapamiętania. 2. Każda nazwa powinna być łatwa do wymówienia. 3. Każda nazwa powinna być unikatowa i nie być zajęta przez inną firmę.'''
})
print(result.content)
