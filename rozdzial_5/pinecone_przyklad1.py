from pinecone import Pinecone, ServerlessSpec 
import os
# Inicjalizuj połączenie Initialize connection (pobierz klucz API ze strony app.pinecone.io):
#os.environ["PINECONE_API_KEY"] = "tu-wstaw-swój-klucz-API"
index_name = "employee-handbook"
environment = "us-east-1"
pc = Pinecone() # W tym miejscu odczytujemy zmienną środowiskową PINECONE_API_KEY
# Sprawdź, czy indeks już istnieje:
# (nie powinien, jeśli uruchamiasz kod po raz pierwszy)
if index_name not in pc.list_indexes().names():
  # jeśli indeks nie istnieje, utwórz go
  pc.create_index(
    index_name,
    # Korzystamy z tych samych wymiarów wektorów, co w modelu text-embedding-ada-002
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region=environment),
  )
# Połącz z indeksem:
index = pc.Index(index_name) 
# Przejrzyj statystyki indeksu:
print(index.describe_index_stats())
