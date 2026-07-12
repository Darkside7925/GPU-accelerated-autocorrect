"""A starter set of very common English typos, seeded into Layer 1 once on first
run so the obvious cases correct instantly with no model and no waiting.

Without this, a fresh install defers ambiguous short transpositions (teh, taht)
to the async Layer 3, so the corrections a new user tries first feel like they
do nothing. These are unambiguous, high-frequency mistakes where the intended
word is not in question. They seed at "manual" confidence (above the auto-apply
gate), and seeding is one-time and flagged, so if you undo one it stays gone.
"""

from __future__ import annotations

COMMON_TYPOS = {
    # classic transpositions
    "teh": "the", "hte": "the", "taht": "that", "thta": "that", "adn": "and",
    "nad": "and", "thsi": "this", "tihs": "this", "jsut": "just", "juts": "just",
    "woudl": "would", "coudl": "could", "shoudl": "should", "waht": "what",
    "wiht": "with", "whit": "with", "tehn": "then", "thne": "then", "yuor": "your",
    "yoru": "your", "oyu": "you", "cna": "can", "fro": "for", "ofr": "for",
    "ot": "to", "og": "go", "si": "is", "fi": "if", "ti": "it",
    "aslo": "also", "alos": "also", "form": "from", "fomr": "from",
    "wont": "won't", "cant": "can't", "dont": "don't", "im": "I'm",
    "ive": "I've", "id": "I'd", "youre": "you're", "theyre": "they're",
    # frequent misspellings
    "recieve": "receive", "recieved": "received", "seperate": "separate",
    "definately": "definitely", "occured": "occurred", "untill": "until",
    "becuase": "because", "becasue": "because", "wich": "which", "freind": "friend",
    "beleive": "believe", "acheive": "achieve", "wierd": "weird",
    "calender": "calendar", "tommorow": "tomorrow", "tommorrow": "tomorrow",
    "enviroment": "environment", "goverment": "government", "arguement": "argument",
    "occassion": "occasion", "neccessary": "necessary", "existance": "existence",
    "independant": "independent", "similiar": "similar", "gaurd": "guard",
    "truely": "truly", "thruogh": "through", "thorugh": "through", "alright": "alright",
    "thier": "their", "wanna": "wanna", "gonna": "gonna", "prolly": "probably",
    "probaly": "probably", "probally": "probably", "acutally": "actually",
    "basicaly": "basically", "finaly": "finally", "realy": "really",
    "usualy": "usually", "quikly": "quickly", "sucessful": "successful",
    "accross": "across", "adress": "address", "arround": "around",
    "belive": "believe", "carefull": "careful", "comming": "coming",
    "dissapear": "disappear", "embarass": "embarrass", "familar": "familiar",
    "greatful": "grateful", "happend": "happened", "immediatly": "immediately",
    "knowlege": "knowledge", "lenght": "length", "libary": "library",
    "maintainance": "maintenance", "noticable": "noticeable", "occurance": "occurrence",
    "persistant": "persistent", "priviledge": "privilege", "reccomend": "recommend",
    "refered": "referred", "relevent": "relevant", "resturant": "restaurant",
    "rythm": "rhythm", "succesful": "successful", "suprise": "surprise",
    "tommorrow": "tomorrow", "unfortunatly": "unfortunately", "wellcome": "welcome",
    # very common ones the deterministic matcher ranks wrong or under-scores
    "thnks": "thanks", "thx": "thanks", "somethign": "something", "wrked": "worked",
    "helllo": "hello", "wass": "was", "wat": "what", "dont": "don't", "doesnt": "doesn't",
    "didnt": "didn't", "isnt": "isn't", "wasnt": "wasn't", "couldnt": "couldn't",
    "wouldnt": "wouldn't", "shouldnt": "shouldn't", "havent": "haven't",
    "hasnt": "hasn't", "wouldnt": "wouldn't", "thats": "that's", "whats": "what's",
    "hes": "he's", "shes": "she's", "theres": "there's", "heres": "here's",
    "lets": "let's", "wanna": "want to", "gimme": "give me", "gotta": "got to",
    "kinda": "kind of", "sorta": "sort of", "outta": "out of", "lemme": "let me",
    "ppl": "people", "u": "you", "ur": "your", "r": "are", "n": "and",
    "abt": "about", "bc": "because", "b4": "before", "tho": "though", "thru": "through",
    "nite": "night", "wanna": "want to", "runing": "running", "makeing": "making",
    "writeing": "writing", "useing": "using", "comeing": "coming", "haveing": "having",
    "geting": "getting", "puting": "putting", "cuting": "cutting", "siting": "sitting",
    "stoping": "stopping", "planing": "planning", "begining": "beginning",
    "wich": "which", "wether": "whether", "wierd": "weird", "freinds": "friends",
    "peice": "piece", "recieving": "receiving", "beleive": "believe",
    "belive": "believe", "wich": "which", "thier": "their", "alright": "all right",
}

# bump this when COMMON_TYPOS or the dataset grows so existing installs re-seed
SEED_VERSION = 3


def _load_dataset():
    """Yield (typo, correction) from the bundled codespell-derived dataset."""
    from pathlib import Path
    path = Path(__file__).resolve().parent / "misspellings.txt"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or "\t" not in line:
                continue
            typo, correction = line.rstrip("\n").split("\t", 1)
            if typo and correction:
                yield typo, correction


def seed(memory) -> int:
    """Seed common typos into Layer 1: a small hand-picked list plus a large
    curated misspelling dataset (~58k pairs), so most everyday typos correct
    instantly with no model. Versioned, so growing it re-seeds existing installs;
    a mangle the user has already edited is never overwritten."""
    try:
        have = int(memory.get_meta("common_seeded_version", 0) or 0)
    except (TypeError, ValueError):
        have = 0
    if have >= SEED_VERSION:
        return 0
    n = 0
    for mangled, intended in COMMON_TYPOS.items():
        if memory.confidence(mangled) <= 0.0:
            memory.record(mangled, intended, source="manual")
            n += 1
    n += memory.bulk_seed(_load_dataset(), source="dataset", confidence=0.60)
    memory.set_meta("common_seeded_version", str(SEED_VERSION))
    return n
