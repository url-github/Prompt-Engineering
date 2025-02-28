from openai import OpenAI 
client = OpenAI()
# Funkcja, która pobiera osadzenie wektorowe dla danego tekstu
def get_vector_embeddings(text): 
  response = client.embeddings.create(
    input=text,
    model="text-embedding-ada-002"
  )
  embeddings = [r.embedding for r in response.data] 
  return embeddings[0]
#print(get_vector_embeddings("mysz"))
print(get_vector_embeddings("Tu wstaw swój tekst"))
