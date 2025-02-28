import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

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



# Przekonwertuj zdania z listy tekstów do formatu TfidfVectorizer
document_list = [' '.join(s) for s in sentences]

# Oblicz reprezentację TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(document_list)

# Znajdź położenie słów ciasto i kłamstwo w macierzy cech
cake_idx = vectorizer.vocabulary_['ciasto']
lie_idx = vectorizer.vocabulary_['kłamstwo']

# Znajdź i przekształc wektor dla słowa ciasto
cakevec = tfidf_matrix[:, cake_idx].toarray().reshape(1, -1)

# Oblicz podobieństwo kosinusowe
similar_words = cosine_similarity(cakevec, tfidf_matrix.T).flatten()

# Pobierz indeksy sześciu najbardziej podobnych słów, w tym 'ciasto'
top_indices = np.argsort(similar_words)[-6:-1][::-1]

# Pobierz i wyświetl 5 słów najbardziej podobnych do 'ciasto' (wyłączając to słowo)
names = []
for idx in top_indices:
    names.append(vectorizer.get_feature_names_out()[idx])
print("Pięć słów najbardziej podobnych do słowa 'ciasto' to: ", names)

# Oblicz podobieństwo kosinusowe pomiędzy słowami "ciasto" i "kłamstwo"
similarity = cosine_similarity(np.asarray(tfidf_matrix[:,
    cake_idx].todense()), np.asarray(tfidf_matrix[:, lie_idx].todense()))
# Wynik będzie macierzą — możesz obliczyć średnią lub maksymalną wartość podobieństwa
avg_similarity = similarity.mean()
print("Podobieństwo pomiędzy słowami 'ciasto' i 'kłamstwo'", avg_similarity)

# Pokaż podobieństwo między słowami "ciasto" i "istnieje"
exists_idx = vectorizer.vocabulary_['istnieje']
similarity = cosine_similarity(np.asarray(tfidf_matrix[:,
    cake_idx].todense()), np.asarray(tfidf_matrix[:, exists_idx].todense()))
avg_similarity = similarity.mean()
print("Podobieństwo pomiędzy słowami 'ciasto' i 'istnieje'", avg_similarity)

