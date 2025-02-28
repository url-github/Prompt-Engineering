from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI 
client = OpenAI()

def get_vector_embeddings(text): 
  response = client.embeddings.create(
    input=text,
    model="text-embedding-ada-002"
  )
  embeddings = [r.embedding for r in response.data] 
  return embeddings[0]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
  chunk_size=100, # 100 tokenów
  chunk_overlap=20, # 20 tokenów nakładania się 
)
text = """
Witamy w Podręczniku Pracownika "Unicorn Enterprises: Gdzie Dzieje Się Magia"! Jesteśmy zachwyceni, że dołączyłeś do naszego zespołu marzycieli, realizatorów i entuzjastów jednorożców. W Unicorn Enterprises wierzymy, że praca powinna być tak czarująca, jak produktywna. Ten podręcznik to twój bilet do magicznego świata naszej firmy, w którym przedstawimy zasady, polityki i praktyki, które kierują nas w tej niezwykłej podróży. Więc zapnij pasy i przygotuj się na przygodę, jakiej nie doświadczyłeś nigdy wcześniej! Oto pięć środkowych akapitów dla twojego fikcyjnego podręcznika pracownika: **1: Nasza magiczna kultura** W Unicorn Enterprises jesteśmy dumni z naszej unikalnej i czarującej kultury firmy. Wierzymy, że kreatywność i innowacyjność najlepiej rozwijają się, gdy ludzie są szczęśliwi i zainspirowani. Od naszych cotygodniowych piątków "Załóż swój ulubiony kostium mitycznej postaci" po nasze domowe zoo z jednorożcami, staramy się wprowadzać magię do każdego zakątka naszego miejsca pracy. Dlatego nie zaskocz się, jeśli znajdziesz bajkę w pokoju na przerwy lub krasnoluda prowadzącego do toalety. Nasza kultura ma na celu stymulowanie wyobraźni i zachęcanie do współpracy w naszym magicznym zespole. **2: Kodeks postępowania Unicorn** Choć jesteśmy zwolennikami kreatywności, ceniąc sobie profesjonalizm. Nasz Kodeks Postępowania Unicorn zapewnia, że utrzymujemy harmonijną i szanującą atmosferę. Traktowanie wszystkich członków zespołu, bez względu na ich gatunek jednorożca, z dobrocią i szacunkiem jest niezbędne. Zachęcamy także do otwartej komunikacji i konstruktywnej informacji zwrotnej, bo w naszym świecie każda opinia się liczy, tak jak każdy róg na głowie jednorożca! **3: Magiczny równowaga praca-życie** W Unicorn Enterprises rozumiemy znaczenie utrzymania zrównoważonego życia. Oferujemy elastyczne godziny pracy, magiczne dni zdrowia psychicznego, a nawet maga na miejscu, który może zapewnić zaklęcia łagodzące stres, gdy jest to potrzebne. Wierzymy, że szczęśliwy i wypoczęty pracownik to twórczy i produktywny pracownik. Dlatego nie wahaj się korzystać z naszych komnat relaksu lub dołącz do grupowej medytacji pod tęczą biurową. **4: Czarodziejskie korzyści** Nasze zobowiązanie do Twojego dobrego samopoczucia obejmuje naszą magiczną paczkę świadczeń. Będziesz cieszyć się skrzynią pełną różnych bonusów, w tym nieograniczoną liczbą przejażdżek na jednorożcu, niekończącym się kociołkiem kawy i mikstur, a także dostępem do naszej firmowej biblioteki pełnej fascynujących książek. Oferujemy także konkurencyjne plany zdrowotne i stomatologiczne, dbając o to, aby twoje fizyczne samopoczucie było równie mocne, jak twoj duch magiczny. **5: Ciągłe nauka i rozwój** W Unicorn Enterprises wierzymy w ciągłe uczenie się i rozwój. Udostępniamy dostęp do mnóstwa kursów online, magicznych warsztatów i szkoleń prowadzonych przez magów. Niezależnie od tego, czy marzysz o opanowaniu nowych zaklęć, czy podbijaniu nowych wyzwań, jesteśmy tu, aby wspierać Twój osobisty i zawodowy rozwój. Na zakończenie tego podręcznika pamiętaj, że w Unicorn Enterprises dążenie do doskonałości to nigdy niekończąca się wyprawa. Sukces naszej firmy zależy od twojej pasji, kreatywności i zaangażowania w realizację niemożliwego. Zachęcamy Cię do zawsze otaczania się magią, zarówno w pracy, jak i poza nią, i do dzielenia się swoimi pomysłami i innowacjami, które utrzymują naszą czarodziejską podróż. Dziękujemy za bycie częścią naszej mistycznej rodziny, i razem, będziemy kontynuować tworzenie świata, w którym magia i biznes prężnie się rozwijają!
"""
chunks = text_splitter.split_text(text=text)

import numpy as np
import faiss
#  Funkcja get_vector_embeddings została zdefiniowana w poprzednim przykładzie
emb = [get_vector_embeddings(chunk) for chunk in chunks] 
vectors = np.array(emb)
# Utwórz indeks FAISS
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)
# Funkcja, która wykonuje przeszukiwanie wektorowe
def vector_search(query_text, k=1):
  query_vector = get_vector_embeddings(query_text) 
  distances, indices = index.search(
    np.array([query_vector]), k)
  return [(chunks[i], float(dist)) for dist,
    i in zip(distances[0], indices[0])]
# Przykładowe wyszukiwanie
user_query = "czy możemy jeździć za darmo na jednorożcach? "
search_results = vector_search(user_query)
print(f"Wyniki wyszukiwania dla zapytania {user_query}:", search_results)
