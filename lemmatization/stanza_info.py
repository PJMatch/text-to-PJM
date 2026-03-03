import spacy_stanza
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger('stanza').setLevel(logging.ERROR)

nlp = spacy_stanza.load_pipeline("pl")

def lemmatize(text: str):
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space]
    lemmas = [t.lemma_ for t in tokens]
    return doc, tokens, lemmas

def pretty_token_info(doc):
    print("\nToken analysis (POS + DEP + HEAD):")
    print(f"{'Token':<15} {'Lemma':<15} {'POS':<6} {'DEP':<12} {'HEAD':<15}")
    for t in doc:
        if t.is_space:
            continue
        print(f"{t.text:<15} {t.lemma_:<15} {t.pos_:<6} {t.dep_:<12} {t.head.text:<15}")

def find_roles(doc):
    verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")]

    subjects = [t for t in doc if t.dep_ in ("nsubj", "nsubj:pass", "csubj")]
    objects = [t for t in doc if t.dep_ in ("obj", "iobj", "obl", "pobj")]
    negations = [t for t in doc if t.dep_ == "neg"]

    print("\nDetected elements:")
    print("Verbs:", ", ".join([f"{v.text}({v.lemma_})" for v in verbs]) or "—")
    print("Subjects:", ", ".join([f"{s.text}({s.lemma_})" for s in subjects]) or "—")
    print("Objects:", ", ".join([f"{o.text}({o.lemma_})" for o in objects]) or "—")
    print("Negations:", ", ".join([n.text for n in negations]) or "—")

    print("\nVerb frames (subject / object / other dependents):")
    for v in verbs:
        children = list(v.children)

        v_subj = [c for c in children if c.dep_ in ("nsubj", "nsubj:pass", "csubj")]
        v_obj  = [c for c in children if c.dep_ in ("obj", "iobj", "obl", "pobj")]
        v_neg  = [c for c in children if c.dep_ == "neg"]
        v_adv  = [c for c in children if c.dep_ in ("advmod",)]
        v_aux  = [c for c in children if c.dep_ in ("aux", "aux:pass")]

        def join(tokens):
            return ", ".join([t.text for t in tokens]) if tokens else "—"

        print(f"\n Verb: {v.text} (lemma={v.lemma_}, dep={v.dep_})")
        print(f"  subject: {join(v_subj)}")
        print(f"  object:  {join(v_obj)}")
        print(f"  negation: {join(v_neg)}")
        print(f"  aux:     {join(v_aux)}")
        print(f"  adverbial: {join(v_adv)}")

def show_dependency_tree(doc):
    print("\nDependency relations (token --DEP--> head):")
    for t in doc:
        if t.is_space:
            continue
        print(f"{t.text:<15} --{t.dep_:<10}--> {t.head.text}")

if __name__ == "__main__":
    while True:
        s = input("\nEnter a sentence (or ENTER to exit): ").strip()
        if not s:
            break

        doc, tokens, lemmas = lemmatize(s)

        print("\nToken → Lemma")
        for t, l in zip(tokens, lemmas):
            print(f"{t.text:<15} -> {l}")

        print("\nLemmas only:")
        print(" ".join(lemmas))

        pretty_token_info(doc)
        find_roles(doc)
        show_dependency_tree(doc)