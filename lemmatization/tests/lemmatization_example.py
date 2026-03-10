import spacy

#Loading model
nlp = spacy.load("pl_core_news_lg")

with open("argentyna_smukla.txt", "r", encoding="utf-8") as f:
    text = f.read()

#lemmatization
doc = nlp(text)
lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
print("Lemmatyzacja całego tekstu:")
print(" ".join(lemmas))
