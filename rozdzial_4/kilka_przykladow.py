from langchain_openai.chat_models import ChatOpenAI 
from langchain_core.prompts import (
        FewShotChatMessagePromptTemplate,
        ChatPromptTemplate,
)
examples = [ 
  {
    "question": "Co jest stolicą Francji? ",
    "answer": "Paryż",
  },
  {
    "question": "Co jest stolicą Hiszpanii? ",
    "answer": "Madryt",
  } # ...więcej przykładów... 
]
example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{question}"),
            ("ai", "{answer}"),
        ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
)
print(few_shot_prompt.format())
from langchain_core.output_parsers import StrOutputParser
final_prompt = ChatPromptTemplate.from_messages(
        [("system",'''Jesteś odpowiedzialny za odpowiadanie na pytania dotyczące krajów. Zawsze zwracaj tylko nazwę kraju.'''),
        few_shot_prompt,("human", "{question}"),]
)
model = ChatOpenAI()
# Tworzenie łańcucha LCEL z promptem, modelem i parserem StrOutputParser():
chain = final_prompt | model | StrOutputParser()
result = chain.invoke({"question": "Co jest stolicą Polski?"})
print(result)

