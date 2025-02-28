from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_openai.chat_models import ChatOpenAI
db = SQLDatabase.from_uri("sqlite:///./data/demo.db")
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))
# Tworzenie wykonawcy agenta:
agent_executor = create_sql_agent(
  llm=ChatOpenAI(temperature=0), 
  toolkit=toolkit,
  verbose=True, 
  agent_type=AgentType.OPENAI_FUNCTIONS,
)
# Przedstawienie wszystkich tabel: 
agent_executor.invoke("Przedstaw wszystkie tabele")
# 'Baza danych zawiera następujące tabele: \n1. Orders\n2. Products\n3. Users'
user_sql = agent_executor.invoke(
  '''Dodaj 5 nowych użytkowników do bazy danych. Ich imiona to: Jan, Maria, Piotr, Paweł i Janina.'''
)

