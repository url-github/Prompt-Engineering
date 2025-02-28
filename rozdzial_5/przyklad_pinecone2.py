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

from pinecone import Pinecone, ServerlessSpec
import os
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

from tqdm import tqdm # Używane w celu wyświetlania paska postępu
from time import sleep
# Określ ile osadzeń chcesz utworzyć i wstawić za jednym razem
batch_size = 10
retry_limit = 5 # maksymalna liczba ponowień
for i in tqdm(range(0, len(chunks), batch_size)): 
  # Znajdź koniec serii
  i_end = min(len(chunks), i+batch_size) 
  meta_batch = chunks[i:i_end]
  # Pobierz identyfikatory
  ids_batch = [str(j) for j in range(i, i_end)] 
  # Pobierz teksty do zakodowania
  texts = [x for x in meta_batch]
  # Utwórz osadzenia
  # (dodano klauzulę try-except w celu uniknięcia błędu typu RateLimitError)
  done = False 
  try:
    # Pozyskaj osadzenia dla całej serii naraz
    embeds = []
    for text in texts:
      embedding = get_vector_embeddings(text)
      embeds.append(embedding) 
    done = True
  except:
    retry_count = 0
    while not done and retry_count < retry_limit: 
      try:
        for text in texts:
          embedding = get_vector_embeddings(text) 
          embeds.append(embedding)
        done = True 
      except:
        sleep(5)
        retry_count += 1
  if not done:
    print(f"""Nie udało się uzyskać osadzeń po {retry_limit} próbach.""")
    continue
# Czyszczenie metadanych
meta_batch = [{
  'batch': i,
  'text': x
} for x in meta_batch]
to_upsert = list(zip(ids_batch, embeds, meta_batch))
# Upsert do Pinecone'a
index.upsert(vectors=to_upsert)

