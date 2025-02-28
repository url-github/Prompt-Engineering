import spacy
nlp = spacy.load("pl_core_news_sm")
text = "To jest zdanie. To jest inne zdanie."
doc = nlp(text)
for sent in doc.sents:
    print(sent.text)
