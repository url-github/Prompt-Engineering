from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate 
from langchain_openai.chat_models import ChatOpenAI 
prompt=PromptTemplate(
 template='''Jesteś pomocnym asystentem, który tłumaczy język {input_language} na {output_language}.''',
 input_variables=["input_language", "output_language"],
)
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
chat = ChatOpenAI()
print(chat.invoke(system_message_prompt.format_messages(
input_language="polski", output_language="angielski")))

