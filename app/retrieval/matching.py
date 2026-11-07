from functools import lru_cache
import spacy

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans

model_name = "en_core_web_sm"

@lru_cache
def get_nlp() -> Language:
    return spacy.load(model_name, disable=["parser","ner"])

class ExactMatcher:
    def __init__(self, glossary: list[dict[str,str]]) -> None:
        self.nlp = get_nlp()
        self.glossary = glossary
        self.lower_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.lemma_matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")

        for index, entry in enumerate(glossary):
            pattern = self.nlp(entry["source"])
            self.lower_matcher.add(str(index),[pattern])
            self.lemma_matcher.add(str(index),[pattern])
    
    def find(self, text: str) -> list[dict[str,str]]:
        doc = self.nlp(text)
        spans = self.lower_matcher(doc, as_spans=True)
        spans += self.lemma_matcher(doc, as_spans=True)
        longest_spans = filter_spans(spans)

        results: list[dict[str,str]] = []

        for span in longest_spans:
            entry = self.glossary[int(span.label_)]
            if entry not in results:
                results.append(entry)
        
        return results

if __name__ == "__main__":
    from app.retrieval.glossary import load_glossary

    matcher = ExactMatcher(load_glossary())
    examples = [
        "Check the battery level indicator before using the power adapter.",
        "Do not use a rapid charger to support other devices.",
        "Press the multi-function button to answer a call.",
        "Store the chargers and accessories in a moisture-proof bag.",
        "Wireless charging is supported. Plug the USB cable into the charging port.",
        "The lens is dirty. Keep the lenses clean.",
    ]
    for sentence in examples:
        hits = matcher.find(sentence)
        print(sentence)
        print("  ->", [(h["source"], h["target"]) for h in hits])
