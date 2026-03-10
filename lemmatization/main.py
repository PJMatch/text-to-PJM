import spacy_stanza
import json
import warnings
import logging

# Silence logs and warnings
warnings.filterwarnings("ignore")
logging.getLogger('stanza').setLevel(logging.ERROR)

nlp = spacy_stanza.load_pipeline("pl")

text = "Wczoraj Maciek Kowalski poszedł do sklepu w Warszawie, a Anna została w domu."
doc = nlp(text)

pjm_elements = []

# Proper nouns that have equivalents in PJM (no need to spell)
znaki_wyjatki = ["WARSZAWA", "FACEBOOK", "POLSKA", "YOUTUBE"]

def parse_token_for_json(token):
    lemma_upper = token.lemma_.upper()
    
    if lemma_upper in znaki_wyjatki:
        return {"type": "sign", "gloss": lemma_upper}
        
    fingerspell_ents = ["persName", "placeName", "geogName", "orgName"]
    is_proper_noun = token.pos_ == "PROPN" or token.ent_type_ in fingerspell_ents
    
    if is_proper_noun:
        return {"type": "fingerspell", "gloss": lemma_upper, "letters": list(lemma_upper)}
    else:
        return {"type": "sign", "gloss": lemma_upper}
    
def get_noun_phrase(head_token):
    """Gets the head word and all its modifiers (e.g. friend + his + Maciek)"""
    elements = [parse_token_for_json(head_token)]
    
    for sub in head_token.children:
        if sub.is_punct or sub.pos_ in ("ADP", "CCONJ", "SCONJ", "PART", "VERB", "AUX"):
            continue
            
        if sub.dep_ in ("flat", "appos", "nmod", "amod", "det", "nummod"):
            elements.extend(get_noun_phrase(sub))
            
    return elements

# Main loop building the sentence
for sent in doc.sents:
    for token in sent:
        # Processing only main verbs (we skip auxiliary verbs)
        if token.pos_ in ("VERB", "AUX") and token.dep_ != "aux":
            podmioty = []
            dopelnienia = []
            okoliczniki = []
            
            for child in token.children:
                if child.is_punct or child.pos_ in ("ADP", "CCONJ", "SCONJ", "PART"):
                    continue
                    
                # Subjects (who/what performs the action)
                if child.dep_.startswith("nsubj") or child.dep_ == "csubj":
                    podmioty.extend(get_noun_phrase(child))
                    
                # Objects (whom/what)
                elif child.dep_.startswith("obj") or child.dep_ == "iobj" or child.dep_ == "obl:arg":
                    dopelnienia.extend(get_noun_phrase(child))
                    
                # Adverbials (where/when)
                elif child.dep_.startswith("obl") or child.dep_ == "advmod":
                    okoliczniki.extend(get_noun_phrase(child))
            
            # Ordering the glosses: Adverbial -> Subject -> Object -> Verb
            verb_element = [parse_token_for_json(token)]
            klauzula_pjm = okoliczniki + podmioty + dopelnienia + verb_element
            pjm_elements.extend(klauzula_pjm)

data = {
    "pjm_sequence": pjm_elements
}

with open("results_glosses.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("result saved to results_glosses.json")