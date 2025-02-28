from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader 
import glob
from langchain.text_splitter import CharacterTextSplitter
# Obiekt do przechowywania dokumentów z różnych źródeł danych:
all_documents = []
# Wczytaj plik PDF:
loader = PyPDFLoader("data/principles_of_marketing_book.pdf")
pages = loader.load_and_split()
print(pages[0])
# Dodaj metadane do każdej strony:
for page in pages:
  page.metadata["description"] = "Książka Zasady Marketingu"
# Sprawdź, czy metadane zostały dodane:
for page in pages[0:2]: 
  print(page.metadata)
# Zapisz strony książki o marketingu:
all_documents.extend(pages)
csv_files = glob.glob("data/*.csv")	
# Pozostaw tylko pliki, które zawierają słowo Marketing w nazwie pliku:
csv_files = [f for f in csv_files if "Marketing" in f]
# Dla każdego pliku .csv:
for csv_file in csv_files:
  loader = CSVLoader(file_path=csv_file)
  data = loader.load()
  # Zapisz dane do listy all_documents: 
  all_documents.extend(data)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
  chunk_size=200, chunk_overlap=0
)
urls = [
     '''https://storage.googleapis.com/oreilly-content/NutriFusion%20Foods%20Marketing%20Plan%202022.docx''',
     '''https://storage.googleapis.com/oreilly-content/NutriFusion%20Foods%20Marketing%20Plan%202023.docx''', 
]

docs = []
for url in urls:
  loader = Docx2txtLoader(url.replace('\n', '')) 
  pages = loader.load()
  chunks = text_splitter.split_documents(pages)
  # Dodawanie metadanych do każdego fragmentu:
  for chunk in chunks:
    chunk.metadata["source"] = "Plan marketingu NutriFusion Foods - 2022/2023"
  docs.extend(chunks)
# Zapisz strony książki o marketingu:
all_documents.extend(docs)

