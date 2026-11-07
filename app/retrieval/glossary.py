import csv
from pathlib import Path

GLOSSARY_PATH = Path("data/glossary.csv")

def load_glossary(path: Path=GLOSSARY_PATH) -> list[dict[str,str]]:
    entries: list[dict[str, str]]=[]
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row["source"].strip().lower()
            target = row["target"].strip()
            if source and target:
                entries.append({"source":source, "target":target})
    return entries
