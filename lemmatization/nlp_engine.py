# nlp_engine.py
import spacy_stanza
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger('stanza').setLevel(logging.ERROR)

# loading model only once at the start of the server
nlp = spacy_stanza.load_pipeline("pl")

exeptions = ["WARSZAWA", "FACEBOOK", "POLSKA", "YOUTUBE"]

QUESTION_WORDS = {
    "czy", "kto", "co", "komu", "czemu", "gdzie", "dokąd",
    "skąd", "kiedy", "jak", "jaki", "jaka", "jakie", "jakim", 
    "dlaczego", "ile", "ilu", "który", "która", "które"
}

QUESTION_PATTERNS = [
    ["po", "co"],
    ["w", "jaki", "sposób"],
    ["z", "jaki", "powód"],
    ["w", "jaki", "cel"],
    ["z", "jaki", "przyczyna"]
]

CLAUSE_DEPS = {"root", "conj", "advcl", "ccomp", "parataxis"}

def is_question(sentence):
    """Determines if a sentence is a question"""

    # check if sentence ends with a question mark
    if sentence.text.strip().endswith("?"):
        return True

    tokens = [t for t in sentence if not t.is_punct]
    if not tokens:
        return False
    
    # check if the first token is a question word
    first_token = tokens[0].lemma_.lower()

    if first_token in QUESTION_WORDS:
        return True
    
    # check for specific question patterns
    lemmas = [t.lemma_.lower() for t in tokens]

    for pattern in QUESTION_PATTERNS:
        if lemmas[:len(pattern)] == pattern:
            return True
    
    return False

def is_negative(sentence):
    """Determines if a sentence is a negation"""

    for token in sentence:
        if token.lemma_.lower() == "nie":
            return True

        # check for negative polarity in morphological features
        if "Polarity=Neg" in str(token.morph):
            return True

    return False

def classify_sentence(sentence):
    """Determines the sentence type"""

    if is_question(sentence):
        return "question" 
    elif sentence.text.strip().endswith("!"):
        return "exclamation"
    elif is_negative(sentence):
        return "negation"
    else:
        return "statement"
    
def get_tense(token):
    """Determines the tense of a token"""

    if "Past" in token.morph.get("Tense", []):
        return "past"
    elif "Fut" in token.morph.get("Tense", []):
        return "future"
    return "present"

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

def get_clause_tokens(root):
    """Collect tokens belonging only to the clause headed by the given root"""

    tokens = []

    for tok in root.subtree:
        # Ignore nested clause heads
        if tok != root and is_clause_root(tok):
            continue

        skip = False
        for anc in tok.ancestors:
            # Stop when we reach the current clause root
            if anc == root:
                break
            
            # Skip tokens that belong to nested clauses
            if is_clause_root(anc):
                skip = True
                break

        if not skip:
            tokens.append(tok)

    # Sort tokens by their original position in the sentence
    return sorted(tokens, key=lambda t: t.i)   

def collect_dependents(token, subjects, objects, adverbials, predicate_modifiers):
    """Recursively collect subjects, objects, adverbials, and predicate modifiers for a given clause root"""

    for child in token.children:
        if child.is_punct or child.pos_ in ("ADP", "CCONJ", "SCONJ", "PART"):
            continue

        # Subjects (who/what performs the action)
        if child.dep_.startswith("nsubj") or child.dep_ == "csubj":
            subjects.extend(get_noun_phrase(child))

        # Objects (whom/what)
        elif child.dep_.startswith("obj") or child.dep_ == "iobj" or child.dep_ == "obl:arg":
            objects.extend(get_noun_phrase(child))

        # Adverbials (where/when)
        elif child.dep_.startswith("obl") or child.dep_ == "advmod":
            adverbials.extend(get_noun_phrase(child))   

        # Predicate modifiers
        elif child.dep_ in ("amod", "nmod", "det", "nummod"):
            predicate_modifiers.extend(get_noun_phrase(child))

        elif child.dep_ == "xcomp" and child.pos_ in ("VERB", "AUX"):
            objects.append(parse_token_for_json(child))
            collect_dependents(child, subjects, objects, adverbials, predicate_modifiers)

def build_clause_pjm(token):
    """Build the PJM gloss sequence for a clause based on its root and dependents"""

    subjects = []
    objects = []
    adverbials = []
    predicate_modifiers = []

    collect_dependents(token, subjects, objects, adverbials, predicate_modifiers)

    main_verb_data = parse_token_for_json(token)
    tense = get_tense(token)

    for child in token.children:
        if child.dep_.startswith("aux"):
            aux_tense = get_tense(child)
            if aux_tense != "present":
                tense = aux_tense

    if tense != "present":
        main_verb_data["tense"] = tense

    is_negated = False
    if token.lemma_.lower() == "nie" or "Neg" in token.morph.get("Polarity", []):
        is_negated = True

    for child in token.children:
        if child.lemma_.lower() == "nie" or "Neg" in child.morph.get("Polarity", []):
            is_negated = True
            break

    if is_negated:
        main_verb_data["is_negated"] = True

    # Ordering the glosses: Adverbial -> Subject -> Object -> Verb
    verb_element = [main_verb_data] + predicate_modifiers
    clause_pjm = adverbials + subjects + objects + verb_element
    

    return clause_pjm

def parse_token_for_json(token):
    """Determines if the token should be a sign or spelled out, and checks for plurals"""

    lemma_upper = token.lemma_.upper()
    token_data = {}
    
    if lemma_upper in exeptions:
        token_data = {"type": "sign", "gloss": lemma_upper}
    else:
        fingerspell_ents = ["persName", "placeName", "geogName", "orgName"]
        is_proper_noun = token.pos_ == "PROPN" or token.ent_type_ in fingerspell_ents
        
        unknown = token.pos_ in ("X", "SYM")

        if is_proper_noun or unknown:
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
        # Skip punctuation and irrelevant parts of speech
        if sub.is_punct or sub.pos_ in ("ADP", "CCONJ", "SCONJ", "PART"):
            continue

        # Only include modifiers that are relevant for noun phrases
        if sub.dep_ in ("flat", "appos", "nmod", "amod", "det", "nummod", "conj"):
            elements.extend(get_noun_phrase(sub))
            
    return elements

def process_polish_text(text: str) -> dict:
    """Main function to process Polish text and return structured data for PJM translation"""
    doc = nlp(text)
    clauses = []

    for sent in doc.sents:
        clause_roots = split_into_clauses(sent)

        for root in clause_roots:
            clause_text_tokens = get_clause_tokens(root)
            clause_text = " ".join(token.text for token in clause_text_tokens)

            clause_doc = nlp(clause_text)
            clause_sentence = list(clause_doc.sents)[0]

            clause_type = classify_sentence(clause_sentence)
            clause_root = None

            for token in clause_sentence:
                if token.dep_ == "root":
                    clause_root = token
                    break

            if clause_root is None:
                continue

            clause_pjm = build_clause_pjm(root)

            clauses.append({
                "sentence_type": clause_type,
                "pjm_sequence": clause_pjm
            })

    return {
        "text": text,
        "clauses": clauses
    }