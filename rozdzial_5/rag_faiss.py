from langchain_community.vectorstores.faiss import FAISS 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# 1. Utwórz dokumenty:
documents = [
  "James Phoenix pracuje w firmie JustUnderstandingData.",
  "James Phoenix ma aktualnie 31 lat.",
  """Inżynieria danych to proces projektowania i budowania systemów do zbierania, przechowywania i analizy danych na szeroką skalę.""",
]
# 2. Utwórz magazyn wektorów:
vectorstore = FAISS.from_texts(texts=documents, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
# 3. Utwórz prompt:
template = """Odpowiedz na pytanie bazując tylko na następującym kontekście:
---
Kontekst: {context}
---
Pytanie: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
# 4. Utwórz model czata:
model = ChatOpenAI()
chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)
print(chain.invoke("Co do jest inżynieria danych?"))
# 'Inżynieria danych to proces projektowania i budowania systemów do zbierania, przechowywania i analizy danych na szeroką skalę'
print(chain.invoke("Kim jest James Phoenix?"))
# 'Na podstawie przedstawionego kontekstu, James Phoniex to 31-letnia osoba, która pracuje w firmie JustUnderstandingData.'
print(chain.invoke("Kto jest prezydentem Stanów Zjednoczonych?") )
# Nie wiem

