# 1. Zaimportuj pakiet:
import tiktoken
# 2. Wczytaj kodowanie za pomocą metody tiktoken.get_encoding()
encoding = tiktoken.get_encoding("cl100k_base")
# 3. Zamień tekst na tokeny korzystając z metody encoding.encode(), 
# a następnie zamień tokeny na tekst dzięki metodzie encoding.decode()
print(encoding.encode("Praca z Tiktoken to jest fajna!")) 
print(encoding.decode([644, 6077, 1910, 1291, 689, 83317, 13599, 83008, 3978, 3458, 656, 26805, 36188, 12951, 10244, 83, 98667, 36900, 90745, 967, 7910, 0]))
# "Inżynieria danych jest świetna do uczenia się sztucznej inteligencji!

def count_tokens(text_string: str, encoding_name: str) -> int: 
  """
    Zwraca liczbę tokenów w łańcuchu tekstowym korzystając z wybranego kodowania
        Argumenty:
            text: Tekst do tokenizacji
            encoding_name: Nazwa kodowania użytego w trakcie tokenizacji
        Zwraca:
            Liczba tokenów w łańcuchu znaków
        Wyjątki:
            ValueError: Jeśli nazwa kodowania nie jest prawidłowa.
  """
  encoding = tiktoken.get_encoding(encoding_name) 
  num_tokens = len(encoding.encode(text_string)) 
  return num_tokens
# 4. Skorzystaj z funkcji, która zlicza tokeny w łańcuchu znaków
text_string = "Witaj, świecie! To jest test."
print(count_tokens(text_string, "cl100k_base"))
