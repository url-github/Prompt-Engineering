from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_openai.chat_models import ChatOpenAI
db = SQLDatabase.from_uri("sqlite:///./data/demo.db")
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))
SQL_PREFIX = """ Jesteś agentem zaprojektowanym do interakcji z bazą danych SQL. Na podstawie pytania wejściowego, stwórz poprawne składniowo zapytanie {dialect} do wykonania, następnie spójrz na wyniki zapytania i zwróć odpowiedź. O ile użytkownik nie określi konkretnej liczby przykładów, które chce uzyskać, zawsze ogranicz swoje zapytanie do maksymalnie {top_k} wyników. Możesz sortować wyniki według odpowiedniej kolumny, aby zwrócić najciekawsze przykłady z bazy danych. Nigdy nie pytaj o wszystkie kolumny z konkretnej tabeli, pytaj tylko o odpowiednie kolumny na podstawie pytania. Masz dostęp do narzędzi do interakcji z bazą danych. Używaj tylko poniższych narzędzi. Używaj tylko informacji zwróconych przez poniższe narzędzia do skonstruowania swojej ostatecznej odpowiedzi. MUSISZ sprawdzić swoje zapytanie przed jego wykonaniem. Jeżeli podczas wykonywania zapytania wystąpi błąd, przepisz zapytanie i spróbuj ponownie. Jeżeli pytanie nie wydaje się związane z bazą danych, zwróć "Nie wiem" jako odpowiedź.
"""
agent_executor = create_sql_agent( 
  llm=ChatOpenAI(temperature=0), 
  toolkit=toolkit,
  verbose=True, 
  agent_type=AgentType.OPENAI_FUNCTIONS, 
  prefix=SQL_PREFIX,
)
user_sql = agent_executor.invoke(
  '''Dodaj 5 nowych użytkowników do bazy danych. Ich imiona to: Jan, Maria, Piotr, Paweł i Janina.'''
)
agent_executor.invoke(user_sql)
# '...sql\nINSERT INTO Users (FirstName, LastName, Email, # DateJoined)\nVALUES (...)...'
# Testujemy czy Piotr został wstawiony do bazy danych:
agent_executor.invoke("Czy w bazie danych jest obecny Piotr?")
# '''Tak, w bazie danych występuje Piotr. Oto jego dane szczegółowe:\n -Imię: Piotr...'''

