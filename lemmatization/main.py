import spacy_stanza
import json
import warnings
import logging

# Silence logs and warnings
warnings.filterwarnings("ignore")
logging.getLogger('stanza').setLevel(logging.ERROR)

nlp = spacy_stanza.load_pipeline("pl")

text = "Czemu nie poszedłeś do sklepu kupić jabłka?" \
       "Jutro będzie padać!" \
       "Szczecin jest pięknym miastem."
doc = nlp(text)

pjm_sentences = []

# Proper nouns that have equivalents in PJM
exeptions = ["WARSZAWA", "FACEBOOK", "POLSKA", "YOUTUBE"]

def parse_token_for_json(token):
    """Determines if the token should be a sign or spelled out, and checks for plurals"""
    lemma_upper = token.lemma_.upper()
    
    token_data = {}
    
    if lemma_upper in exeptions:
        token_data = {"type": "sign", "gloss": lemma_upper}
    else:
        fingerspell_ents = ["persName", "placeName", "geogName", "orgName"]
        is_proper_noun = token.pos_ == "PROPN" or token.ent_type_ in fingerspell_ents
        
        if is_proper_noun:
            token_data = {"type": "fingerspell", "gloss": lemma_upper, "letters": list(lemma_upper)}
        else:
            token_data = {"type": "sign", "gloss": lemma_upper}
            
    if "Plur" in token.morph.get("Number", []):
        token_data["is_plural"] = True
        
    return token_data
    
def get_noun_phrase(head_token):
    """Gets the head word and all its modifiers"""
    elements = [parse_token_for_json(head_token)]
    
    for sub in head_token.children:
        if sub.is_punct or sub.pos_ in ("ADP", "CCONJ", "SCONJ", "PART", "VERB", "AUX"):
            continue
            
        if sub.dep_ in ("flat", "appos", "nmod", "amod", "det", "nummod"):
            elements.extend(get_noun_phrase(sub))
            
    return elements

# Main loop building the sentence
for sent in doc.sents:
    
    # Determine sentence type
    sentence_type = "statement"
    if "?" in sent.text:
        sentence_type = "question"
    elif "!" in sent.text:
        sentence_type = "exclamation"
        
    sentence_sequence = []
    
    for token in sent:
        is_main_verb = token.pos_ in ("VERB", "AUX") and not token.dep_.startswith("aux") and token.dep_ != "cop"
        is_nominal_predicate = any(child.dep_ == "cop" for child in token.children)
        
        if is_main_verb or is_nominal_predicate:
            subjects = []
            objects = []
            adverbials = []
            copula_verb = []
            predicate_modifiers = []
            
            is_negated = False
            verb_tense = "present"
            
            if "Past" in token.morph.get("Tense", []):
                verb_tense = "past"
            elif "Fut" in token.morph.get("Tense", []):
                verb_tense = "future"
            
            for child in token.children:
                if "Neg" in child.morph.get("Polarity", []):
                    is_negated = True
                    continue
                
                # Check for auxiliary verbs that carry tense
                if child.dep_.startswith("aux"):
                    if "Past" in child.morph.get("Tense", []):
                        verb_tense = "past"
                    elif "Fut" in child.morph.get("Tense", []):
                        verb_tense = "future"
                    continue
                
                if child.is_punct or child.pos_ in ("ADP", "CCONJ", "SCONJ", "PART"):
                    continue
                
                # Copula verbs are treated as separate signs that can carry tense information
                if child.dep_ == "cop":
                    cop_data = parse_token_for_json(child)
                    if "Past" in child.morph.get("Tense", []):
                        cop_data["tense"] = "past"
                    elif "Fut" in child.morph.get("Tense", []):
                        cop_data["tense"] = "future"
                        
                    copula_verb.append(cop_data)
                    continue
                    
                if child.dep_.startswith("nsubj") or child.dep_ == "csubj":
                    subjects.extend(get_noun_phrase(child))
                    
                elif child.dep_.startswith("obj") or child.dep_ == "iobj" or child.dep_ == "obl:arg":
                    objects.extend(get_noun_phrase(child))
                    
                elif child.dep_.startswith("obl") or child.dep_ == "advmod":
                    adverbials.extend(get_noun_phrase(child))
                    
                elif child.dep_ in ("amod", "nmod", "det", "nummod"):
                    predicate_modifiers.extend(get_noun_phrase(child))
            
            main_verb_data = parse_token_for_json(token)
            
            if is_negated:
                main_verb_data["is_negated"] = True
                
            if verb_tense != "present":
                main_verb_data["tense"] = verb_tense
            
            # Ordering the glosses: Adverbial -> Subject -> Object -> Copula -> Noun/Verb -> Adjective
            verb_element = copula_verb + [main_verb_data] + predicate_modifiers
            pjm_order = adverbials + subjects + objects + verb_element
            sentence_sequence.extend(pjm_order)

    if len(sentence_sequence) > 0:
        pjm_sentences.append({
            "sentence_type": sentence_type,
            "pjm_sequence": sentence_sequence
        })

data = {
    "sentences": pjm_sentences
}

with open("results_glosses.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("result saved to results_glosses.json")