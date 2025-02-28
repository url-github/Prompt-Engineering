from openai import OpenAI
client = OpenAI()

def get_vector_embeddings(text):
  response = client.embeddings.create(
    input=text,
    model="text-embedding-ada-002"
  )
  embeddings = [r.embedding for r in response.data]
  return embeddings[0]


from pinecone import Pinecone, ServerlessSpec
import os
index_name = "employee-handbook"
environment = "us-east-1"
pc = Pinecone() # W tym miejscu odczytujemy zmienną środowiskową PINECONE_API_KEY
index = pc.Index(index_name)

# Pobierz z Pinecone
user_query = "czy mamy zapewnione darmowe przejażdżki na jednorożcach?"
def pinecone_vector_search(user_query, k):
  xq = get_vector_embeddings(user_query)
  res = index.query(vector=xq, top_k=k, include_metadata=True) 
  return res
print(pinecone_vector_search(user_query, k=1))
