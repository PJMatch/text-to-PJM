import warnings
import logging
import spacy_stanza

warnings.filterwarnings("ignore")
logging.getLogger('stanza').setLevel(logging.ERROR)

nlp = spacy_stanza.load_pipeline("pl")

tekst = "Poszedłem do sklepu, ale nie kupiłem mleka, bo było drogie."

doc = nlp(tekst)
unwanted = ["PUNCT", "ADP", "CCONJ", "SCONJ"]

base_form = []

for token in doc:
    if token.pos_ == "AUX" and token.lemma_ == "być":
        continue
    
    if token.pos_ in unwanted:
        continue
        
    base_form.append(token.lemma_)

print("Original:", tekst)
print("PJM:     ", " ".join(base_form))
print("\n")