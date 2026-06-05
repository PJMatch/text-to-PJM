# nlp_engine.py
import spacy_stanza
import warnings
import logging
import re
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.getLogger('stanza').setLevel(logging.ERROR)

# loading model only once at the start of the server
nlp = spacy_stanza.load_pipeline("pl", download_method="REUSE_RESOURCES")


EXCEPTIONS = {"WARSZAWA", "FACEBOOK", "POLSKA", "YOUTUBE"}

MULTI_WORD_TO_SAFE = {
    "dzień dobry": "DZIENDOBRY",
    "do widzenia": "DOWIDZENIA"
}

SAFE_TO_GLOSS = {
    "DZIENDOBRY": "DZIEŃ_DOBRY",
    "DOWIDZENIA": "DO_WIDZENIA"
}

FORCED_CLAUSE_ROOTS = {"DZIENDOBRY", "DOWIDZENIA"}

NEGATED_VERBS_MAP = {
    "ROZUMIEĆ": "NIE_ROZUMIEĆ",
    "WIEDZIEĆ": "NIE_WIEDZIEĆ",
}

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

CLAUSE_DEPS = {"root", "conj", "advcl", "ccomp", "parataxis", "acl:relcl"}

def preprocess_text(text: str) -> str:
    """Encode exceptions and multi-word expressions to safe tokens before processing"""
    
    for phrase, safe_token in MULTI_WORD_TO_SAFE.items():
        pattern = re.compile(r'\b' + phrase + r'\b', re.IGNORECASE)
        text = pattern.sub(safe_token, text)
    return text

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

def infer_subject_from_verb(token):
    """Recover hidden Polish subject from verb morphology."""

    if token is None:
        return None

    person = token.morph.get("Person")
    number = token.morph.get("Number")

    if not person:
        return None

    person = person[0]
    number = number[0] if number else None

    if person == "1":
        gloss = "JA" if number == "Sing" else "MY"

    elif person == "2":
        gloss = "TY" if number == "Sing" else "WY"

    else: # got rid of 3rd person - useless and lead to bugs
        return None

    return {
        "type": "sign",
        "gloss": gloss
    }

def get_clause_subtree_tokens(token):
    """Gets all words that belong to the same clause as the given token."""
    return sorted(list(token.subtree), key=lambda t: t.i)


def get_finite_controller(token):
    """Finds the finite verb or auxiliary that carries grammatical person/number information for the clause."""

    candidates = []

    # Look for verbs and auxiliaries in the subtree of the clause root
    for tok in get_clause_subtree_tokens(token):
        if tok.pos_ in ("VERB", "AUX") and tok.morph.get("Person"):
            candidates.append(tok)

    if not candidates:
        return None

    # prefer auxiliary verbs because they usually
    # carry the grammatical person/tense information
    for tok in candidates:
        if tok.dep_.startswith("aux"):
            return tok

    return candidates[0]

def has_dative_experiencer(token):
    """Detect dative experiencers to avoid incorrect third-person subject inference."""
    for tok in token.subtree:
        if tok.lemma_.lower() in ("ja", "ty", "my", "wy"):
            if "Dat" in tok.morph.get("Case", []):
                return True

    return False

def is_clause_root(token):
    """Determines if a token is a clause root"""

    if token.dep_ == "root":
        return True

    # Force certain tokens to be treated as clause roots
    if token.lemma_.upper() in FORCED_CLAUSE_ROOTS or token.text.upper() in FORCED_CLAUSE_ROOTS:
        return True

    if token.dep_ in CLAUSE_DEPS:
        if token.dep_ == "conj" and token.pos_ in ("NOUN", "PROPN", "ADJ"):
            return False

        return token.pos_ in ("VERB", "AUX", "ADJ", "NOUN", "PROPN", "NUM")

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

def collect_dependents(token, subjects, objects, adverbials, predicate_modifiers, question_words):
    """Recursively collect subjects, objects, adverbials, and predicate modifiers for a given clause root"""

    for child in token.children:
        if child.is_punct or child.pos_ in ("ADP", "CCONJ", "SCONJ", "PART"):
            continue
        
        # Fail-safe for nested clause heads
        if is_clause_root(child):
            continue

        # Handle numbers as objects
        if child.pos_ == "NUM":
            objects.append(parse_token_for_json(child))
            continue

        if child.lemma_.lower() in QUESTION_WORDS:
            question_words.append(parse_token_for_json(child))
            continue

        # Subjects (who/what performs the action)
        if child.dep_.startswith("nsubj") or child.dep_ == "csubj":
            subjects.extend(get_noun_phrase(child, question_words))

        # Objects (whom/what)
        elif child.dep_.startswith("obj") or child.dep_ == "iobj" or child.dep_ == "obl:arg":
            objects.extend(get_noun_phrase(child, question_words))

        # Adverbials (where/when)
        elif child.dep_.startswith("obl") or child.dep_ in ("advmod", "vocative", "discourse"):
            adverbials.extend(get_noun_phrase(child, question_words))

        # Predicate modifiers and direct conjunctions/appositions
        # ADDED: "conj", "appos", "flat" to catch words like "wesoły", "nowy", "chorzy"
        elif child.dep_ in ("amod", "nmod", "det", "nummod", "conj", "appos", "flat"):
            predicate_modifiers.extend(get_noun_phrase(child, question_words))

        elif child.dep_ == "xcomp" and child.pos_ in ("VERB", "AUX"):
            objects.append(parse_token_for_json(child))
            collect_dependents(child, subjects, objects, adverbials, predicate_modifiers, question_words)

def build_clause_pjm(token, clause_type):
    """Build the PJM gloss sequence for a clause based on its root and dependents"""

    subjects = []
    objects = []
    adverbials = []
    predicate_modifiers = []
    question_words = []

    collect_dependents(token, subjects, objects, adverbials, predicate_modifiers,question_words)

    # Infer hidden subjects (JA, TY, MY, WY, ON) from verb inflection if the clause has no explicit subject
    if not subjects:
        if not has_dative_experiencer(token):
            finite_controller = get_finite_controller(token)

            inferred_subject = infer_subject_from_verb(finite_controller,)

            if inferred_subject:
                subjects.append(inferred_subject)

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

    # Handle negated exceptions for verbs
    verb_gloss = main_verb_data.get("gloss")
    if main_verb_data.get("is_negated") and verb_gloss in NEGATED_VERBS_MAP:
        main_verb_data["gloss"] = NEGATED_VERBS_MAP[verb_gloss]
        main_verb_data.pop("is_negated", None)

    # Clear question words if clause is not a question
    if clause_type != "question":
        question_words = []

    # Ordering the glosses: Subject -> Adverbial -> Object -> Verb
    verb_element = [main_verb_data] + predicate_modifiers
    clause_pjm = subjects + adverbials + objects + verb_element + question_words

    return clause_pjm

def parse_token_for_json(token):
    """Determines if the token should be a sign or spelled out, and checks for plurals"""

    if token.pos_ == "NUM":
        return {"type": "sign", "gloss": token.text.upper()}

    # Decode exceptions
    text_upper = token.text.upper()
    lemma_upper_raw = token.lemma_.upper()
    
    if text_upper in SAFE_TO_GLOSS:
        return {"type": "sign", "gloss": SAFE_TO_GLOSS[text_upper]}
    if lemma_upper_raw in SAFE_TO_GLOSS:
        return {"type": "sign", "gloss": SAFE_TO_GLOSS[lemma_upper_raw]}

    lemma_upper, lexical_negation = extract_negated_base_form(token)
    token_data = {}

    if lemma_upper in EXCEPTIONS:
        token_data = {"type": "sign", "gloss": lemma_upper}
    else:
        fingerspell_ents = ["persName", "placeName", "geogName", "orgName"]
        is_proper_noun = token.pos_ == "PROPN" or token.ent_type_ in fingerspell_ents

        if is_proper_noun:
            token_data = {"type": "fingerspell", "gloss": lemma_upper, "letters": list(lemma_upper)}
        else:
            token_data = {"type": "sign", "gloss": lemma_upper}

    if lexical_negation:
        token_data["is_negated"] = True

    if "Plur" in token.morph.get("Number", []):
        token_data["is_plural"] = True

    return token_data

def get_noun_phrase(head_token, question_words=None):
    """Gets the head word and all its modifiers"""

    numbers = []
    after_head = []

    for sub in head_token.children:
        # Skip punctuation and irrelevant parts of speech
        if sub.is_punct or sub.pos_ in ("ADP", "CCONJ", "SCONJ", "PART"):
            continue

        # Fail-safe for nested clause heads
        if is_clause_root(sub):
            continue

        if question_words is not None and sub.lemma_.lower() in QUESTION_WORDS:
            question_words.append(parse_token_for_json(sub))
            continue

        # Handle numbers as part of noun phrases
        if sub.pos_ == "NUM":
            numbers.extend(get_noun_phrase(sub))
            continue

        # Only include modifiers that are relevant for noun phrases
        if sub.dep_ in ("flat", "appos", "nmod", "amod", "det", "nummod", "conj", "advmod", "obl"):
            after_head.extend(get_noun_phrase(sub))

    return numbers + [parse_token_for_json(head_token)] + after_head

def extract_negated_base_form(token):
    """Detect negated adjectives like 'niezadowolony', 'niemiły'."""

    if token.pos_ != "ADJ":
        return token.lemma_.upper(), False

    lemma = token.lemma_.lower()
    text = token.text.lower()

    # Check if the lemma itself starts with "nie" and is longer than 3 characters
    if lemma.startswith("nie") and len(lemma) > 3:
        return lemma[3:].upper(), True

    # Check if the actual text starts with "nie" (to catch cases where the lemma might not reflect the negation)
    if text.startswith("nie") and len(text) > 3:
        return text[3:].upper(), True

    return token.lemma_.upper(), False

def process_polish_text(text: str) -> dict:
    """Main function to process Polish text and return structured data for PJM translation"""

    text = preprocess_text(text)
    
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
            
            # Override for relative and adverbial clauses
            if root.dep_ in ("acl:relcl", "advcl") and not clause_text.strip().endswith("?"):
                clause_type = "negation" if is_negative(clause_sentence) else "statement"

            clause_root = None

            for token in clause_sentence:
                if token.dep_ == "root":
                    clause_root = token
                    break

            if clause_root is None:
                continue

            clause_pjm = build_clause_pjm(root, clause_type)

            clauses.append({
                "sentence_type": clause_type,
                "pjm_sequence": clause_pjm
            })

    return {
        "text": text,
        "clauses": clauses
    }


#=============================== VERIFY  OUTPUT ======================================

def load_sentences_from_file(file_path="sentences.txt"):
    """Reads sentences from a text file and returns them as a list."""
    with open(file_path, "r", encoding="utf-8") as file:
        sentences = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return sentences


def pjm_sequence_to_text(pjm_sequence):
    """Converts PJM gloss items into one text line with negation and tense markers."""
    glosses = []

    for item in pjm_sequence:
        gloss = item.get("gloss", "")

        if item.get("is_negated"):
            gloss += " [NEG]"

        if item.get("tense"):
            gloss += f" [{item.get('tense').upper()}]"

        glosses.append(gloss)
    return " ".join(glosses)


def save_lemmatization_results(
    sentences_file="sentences.txt",
    output_txt_file="lemmatization_results.txt"
):
    """Reads sentences from a file, processes them with the NLP engine, and saves readable PJM results."""
    sentences = load_sentences_from_file(sentences_file)
    sentence_count = 0
    
    with open(output_txt_file, "w", encoding="utf-8") as txt_file:
        for sentence in sentences:
            result = process_polish_text(sentence)
            sentence_count += 1
            txt_file.write(f"text: {sentence}\n")

            for index, clause in enumerate(result["clauses"], start=1):
                sentence_type = clause["sentence_type"]
                pjm_sequence = clause["pjm_sequence"]
                readable_pjm = pjm_sequence_to_text(pjm_sequence)

                txt_file.write(f"clause {index} type: {sentence_type}\n")
                txt_file.write(f"lemmatization: {readable_pjm}\n")

            txt_file.write("\n" + "-" * 60 + "\n\n")
        
            print(f"Progress: {sentence_count}/{len(sentences)} sentences processed")

    print(f"Saved TXT results to: {output_txt_file}")

def result_to_text(result):
    """Converts the NLP result into one readable text line with clause types and PJM sequences."""
    clauses_text = []

    for clause in result["clauses"]:
        sentence_type = clause["sentence_type"]
        readable_pjm = pjm_sequence_to_text(clause["pjm_sequence"])
        clauses_text.append(f"{sentence_type}: {readable_pjm}")

    return " | ".join(clauses_text)

def draw_repeatability_chart(results):
    """Draws and saves a bar chart showing repeatability percentage for each sentence."""
    labels = [f"Zdanie {i + 1}" for i in range(len(results))]
    values = [item["percentage"] for item in results]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.ylabel("Powtarzalność (%)")
    plt.title("Powtarzalność wyników dla 10 przykładowych zdań")
    plt.tight_layout()
    plt.savefig("repeatability.png")
    plt.show()


def verify_repeated_outputs(
    sentences_file="sentences.txt",
    repeats=20,
    output_file="repeatability_results.txt"
):
    """Checks if repeated NLP processing returns the same output and saves repeatability results."""
    sentences = load_sentences_from_file(sentences_file)[:10]
    results = []

    with open(output_file, "w", encoding="utf-8") as file:
        for sentence in sentences:
            base_output = result_to_text(process_polish_text(sentence))
            same_count = 0

            for _ in range(repeats):
                current_output = result_to_text(process_polish_text(sentence))

                if current_output == base_output:
                    same_count += 1

            percentage = (same_count / repeats) * 100

            results.append({
                "sentence": sentence,
                "percentage": percentage
            })

            file.write(f"text: {sentence}\n")
            file.write(f"repeatability: {percentage}%\n")
            file.write(f"output: {base_output}\n")
            file.write("-" * 60 + "\n\n")

            print(f"{sentence} -> {percentage}%")

    draw_repeatability_chart(results)

    print(f"Saved repeatability results to: {output_file}")
    print("Saved chart to: repeatability.png")


if __name__ == "__main__":
    verify_repeated_outputs(repeats = 20)