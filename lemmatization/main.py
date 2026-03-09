import stanza
import spacy_stanza
import json

stanza.download("pl")
nlp = spacy_stanza.load_pipeline("pl")

text = "Patryk ma dużego kota i poszedł z nimi do weterynarza."
doc = nlp(text)

subjects = []
verbs = []
objects = []
tokens_data = []
glosses = []
attributes = []
adverbials = []
pronouns = []
prepositions = []
conjunctions = []

for token in doc:
    if token.is_punct:
        continue

    glosses.append(token.lemma_)

    # Subject
    if token.dep_ in ("nsubj", "csubj"):
        subjects.append(token.text)

    # Verbs
    if token.pos_ in ("VERB", "AUX"):
        verbs.append(token.text)

    # object
    if token.dep_ in ("obj", "iobj"):
        objects.append(token.text)

    # attributes 
    if token.dep_ in ("amod", "nmod"):
        attributes.append(token.text)

    # adverbials
    if token.dep_ in ("obl", "advmod"):
        adverbials.append(token.text)

    # prounouns
    if token.pos_ == "PRON":
        pronouns.append(token.text)

    # prepositions
    if token.pos_ == "ADP":
        prepositions.append(token.text)

    #conjuction
    if token.pos_ == "CCONJ":
        conjunctions.append(token.text)

data = {
    "glosses": glosses,
}

with open("lemmatization/results_glosses.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("Original:", text)
print("Lemmatization:", " ".join(glosses))
print("=================================")
print("Subjects:", subjects)
print("Verbs:", verbs)
print("Objects:", objects)
print("Attributes:", attributes)
print("Adverbials:", adverbials)
print("Pronouns:", pronouns)
print("Prepositions:", prepositions)
print("Conjunctions:", conjunctions)
print("Saved to lemmatization/results_glosses.json")