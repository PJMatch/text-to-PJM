import spacy_stanza
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger('stanza').setLevel(logging.ERROR)

nlp = spacy_stanza.load_pipeline("pl")

CLAUSE_DEPS = {"root", "conj", "advcl", "ccomp", "parataxis"}

def is_clause_root(token):
    """Determines if a token is a clause root"""
    if token.dep_ == "root":
        return True
    
    if token.dep_ in CLAUSE_DEPS:
        return token.pos_ in ("VERB", "AUX", "ADJ", "NOUN", "PROPN")
        
    return False

def split_into_clauses(sentence):
    """Split a sentence into clauses based on dependency parsing"""
    return [token for token in sentence if is_clause_root(token)]

def collect_dependents(token, subjects, objects, adverbials, predicate_modifiers):
    """Recursively collect subjects, objects, adverbials, and predicate modifiers for a given clause root"""
    for child in token.children:
        if child.is_punct or child.pos_ in ("ADP", "CCONJ", "SCONJ", "PART"):
            continue

        if child.dep_.startswith("nsubj") or child.dep_ == "csubj":
            subjects.extend(get_noun_phrase(child))
        elif child.dep_.startswith("obj") or child.dep_ == "iobj" or child.dep_ == "obl:arg":
            objects.extend(get_noun_phrase(child))
        elif child.dep_.startswith("obl") or child.dep_ == "advmod":
            adverbials.extend(get_noun_phrase(child))   
        elif child.dep_ in ("amod", "nmod", "det", "nummod"):
            predicate_modifiers.extend(get_noun_phrase(child))
        elif child.dep_ == "xcomp" and child.pos_ in ("VERB", "AUX"):
            objects.append(get_lemma(child))
            collect_dependents(child, subjects, objects, adverbials, predicate_modifiers)

def build_clause_pjm(token):
    """Build the PJM gloss sequence for a clause based on its dependents"""    
    subjects = []
    objects = []
    adverbials = []
    predicate_modifiers = []

    collect_dependents(token, subjects, objects, adverbials, predicate_modifiers)

    main_verb = get_lemma(token)

    verb_element = [main_verb] + predicate_modifiers
    clause_pjm = adverbials + subjects + objects + verb_element
    
    return clause_pjm

def get_lemma(token):
    """Returns the uppercase lemma of a token."""
    return token.lemma_.upper()
    
def get_noun_phrase(head_token):
    """Builds a noun phrase from the head token and its relevant child tokens."""
    elements = [get_lemma(head_token)]
    for sub in head_token.children:
        if sub.is_punct or sub.pos_ in ("ADP", "CCONJ", "SCONJ", "PART"):
            continue
        if sub.dep_ in ("flat", "appos", "nmod", "amod", "det", "nummod", "conj"):
            elements.extend(get_noun_phrase(sub))
            
    return elements

input_filename = "input_tatoeba.txt"
output_filename = "output.txt"

with open(input_filename, "r", encoding="utf-8") as file:
    input_lines = [line.strip() for line in file if line.strip()]

output_lines = []

for text in input_lines:
    doc = nlp(text)
    line_clauses = []

    for sent in doc.sents:
        clause_roots = split_into_clauses(sent)

        for root in clause_roots:
            clause_pjm = build_clause_pjm(root)
            
            clause_str = " ".join(clause_pjm)
            line_clauses.append(clause_str)

    output_lines.append(" ".join(line_clauses))

with open(output_filename, "w", encoding="utf-8") as file:
    file.write("\n".join(output_lines) + "\n")

print(f"results saved to: {output_filename}")