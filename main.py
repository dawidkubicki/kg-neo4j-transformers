import spacy
from spacy_llm.util import assemble


def classic_ner(text: str = "Apple is looking at buying U.K. startup for $1 billion"):
    print("classic_ner")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

def llm_ner(text: str = "Apple is looking at buying U.K. startup for $1 billion"):
    print("llm_ner")
    nlp = assemble("config.cfg")
    doc = nlp(text)
    print([(ent.text, ent.label_) for ent in doc.ents])


# Execution for testing
classic_ner()
llm_ner()
