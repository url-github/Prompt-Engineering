from gensim.models import Word2Vec
# Przykładowe dane: lista zdań, gdzie każde zdanie to lista słów.
# W prawdziwej aplikacji w tym miejscu wczytasz i przetworzysz własny korpus tekst.
sentences = [
  ["ciasto", "nie", "istnieje"],
  ["jeśli", "słyszysz", "dźwięk", "trąbki", "jesteś", "za", "blisko"],
  ["po", "co", "szukać", "krańca", "tęczy", "jeśli", "ciasto", "nie", "istnieje?"],
  # ...
  ["nie", "ma", "ciasta", "w", "kosmosie,", "zapytaj", "wheatleya"],
  ["kończę", "testy", "aby", "sprawdzić", "czy", "ciasto", "nie", "istnieje"],
  ["Zamieniłem", "przepis", "na", "ciasto", "z", "recepturą", "na", "neurotoksynę", "mam,", "nadzieję", "że", "to", "ok"],
]+[
  ["ciasto", "nie", "istnieje"],
  ["ciasto", "zdecydowanie", "nie", "istnieje"], 
  ["każdy", "wie", "że", "ciasto", "oznacza", "kłamstwo"], 
  # ...
] * 10 # powtórz wiele razy, aby wzmocnić efekt
# Naucz model word2vec
model =  Word2Vec(sentences, vector_size=100, window=5,
min_count=1, workers=4, seed=43)
# Zapisz model
model.save("custom_word2vec_model.model")
# Jeśli chcesz wczytać model innym razem
# loaded_model = word2vec.load("custom_word2vec_model.model")
# Pobierz wektor dla słowa
vector = model.wv['ciasto']
# Znajdź podobne słowa
similar_words = model.wv.most_similar("ciasto", topn=5)
print("Pięć słów najbardziej podobnych do słowa 'ciasto': ", similar_words)
# Sprawdź bezpośrednio podobieństwo pomiędzy słowa "ciasto" i "kłamstwo"
cake_lie_similarity = model.wv.similarity("ciasto", "kłamstwo")
print("Podobieństwo między słowami 'ciasto' i 'kłamstwo': ", cake_lie_similarity)

