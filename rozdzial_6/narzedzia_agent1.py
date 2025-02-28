# Importujemy istotne pakiety:
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent 
from langchain_openai.chat_models import ChatOpenAI
# Tworzymy agenta CSV:
agent = create_csv_agent(
  ChatOpenAI(temperature=0), 
  "data/heart_disease_uci.csv",
  verbose=True, 
  agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
agent.invoke("Jak wiele wierszy danych jest w pliku? ")
# '920'
agent.invoke("Jakie kolumny są w zbiorze danych??")
# "'id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs',
# 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'"
agent.invoke("Utwórz macierz korelacji danych i zapisz ją do pliku") 
# "The correlation matrix has been saved to a file named
# 'correlation_matrix.csv'."

