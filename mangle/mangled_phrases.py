"""Feasibility test set for Layer 3 context recovery (Phase 0 gate).

Each case is a sentence with exactly one token that will be marked with
double square brackets at benchmark time, plus the intended word.

  context   : the sentence, with "{}" where the marked token goes
  mangled   : the token as typed (heavily mangled, or intentional/correct)
  intended  : the word it should resolve to (== mangled for passthrough cases)
  category  : "mash" (heavy scramble), "transposition" (adjacent slips),
              or "passthrough" (marked token is actually correct and must be
              returned unchanged; these are the non-dictionary tokens that
              would actually reach Layer 3: names, slang, technical terms)

The passthrough set is what enforces the >=95% passthrough gate: a model that
"fixes" kubernetes into something else is disqualified no matter how well it
recovers real mangles. A wrong fix is worse than a missed one.
"""

RECOVERY_CASES = [
    # ------------------------------------------------ mash (heavy scramble)
    {"context": "I never {} about it that way before", "mangled": "yhoguhr", "intended": "thought", "category": "mash"},
    {"context": "let me {} about it for a second", "mangled": "tgjionk", "intended": "think", "category": "mash"},
    {"context": "we should use a queue {} of a list here", "mangled": "ijnsftead", "intended": "instead", "category": "mash"},
    {"context": "that is {} the fastest option we have", "mangled": "probvakly", "intended": "probably", "category": "mash"},
    {"context": "the code is not {} yet but close", "mangled": "fucntioanl", "intended": "functional", "category": "mash"},
    {"context": "the new {} feels great to type on", "mangled": "keyboiard", "intended": "keyboard", "category": "mash"},
    {"context": "I think {} is wrong with the build", "mangled": "somehtingg", "intended": "something", "category": "mash"},
    {"context": "this approach is completely {} from that one", "mangled": "diffrenet", "intended": "different", "category": "mash"},
    {"context": "it was a great learning {} overall", "mangled": "expereince", "intended": "experience", "category": "mash"},
    {"context": "a test harness is {} for this project", "mangled": "necessaary", "intended": "necessary", "category": "mash"},
    {"context": "it crashed {} the config was missing", "mangled": "beacuseee", "intended": "because", "category": "mash"},
    {"context": "do {} you think is best here", "mangled": "wahtever", "intended": "whatever", "category": "mash"},
    {"context": "I {} forgot to push the branch", "mangled": "acutally", "intended": "actually", "category": "mash"},
    {"context": "the model handles {} really well now", "mangled": "langauge", "intended": "language", "category": "mash"},
    {"context": "we need better {} of the workers", "mangled": "managment", "intended": "management", "category": "mash"},
    {"context": "set the {} variable before running", "mangled": "enviromnet", "intended": "environment", "category": "mash"},
    {"context": "that is a strange {} choice for a hero", "mangled": "charcter", "intended": "character", "category": "mash"},
    {"context": "I will {} ship it this week", "mangled": "defintiely", "intended": "definitely", "category": "mash"},
    {"context": "she is my {} and we code together", "mangled": "girlfirend", "intended": "girlfriend", "category": "mash"},
    {"context": "the server keeps {} the same error", "mangled": "retruning", "intended": "returning", "category": "mash"},

    # ------------------------------------------- transposition (adjacent slips)
    {"context": "I {} not believe how fast this is", "mangled": "cna", "intended": "can", "category": "transposition"},
    {"context": "did {} finish the report already", "mangled": "oyu", "intended": "you", "category": "transposition"},
    {"context": "put it on {} table over there", "mangled": "teh", "intended": "the", "category": "transposition"},
    {"context": "I really liked {} approach a lot", "mangled": "taht", "intended": "that", "category": "transposition"},
    {"context": "ship it {} the other fix too", "mangled": "wiht", "intended": "with", "category": "transposition"},
    {"context": "grab your keys {} lets go", "mangled": "adn", "intended": "and", "category": "transposition"},
    {"context": "I have seen {} bug before", "mangled": "thsi", "intended": "this", "category": "transposition"},
    {"context": "I {} pushed the staging branch", "mangled": "jsut", "intended": "just", "category": "transposition"},
    {"context": "I {} appreciate a quick review", "mangled": "woudl", "intended": "would", "category": "transposition"},
    {"context": "it broke {} of the missing token", "mangled": "becuase", "intended": "because", "category": "transposition"},

    # ------------------------------------------ passthrough (must stay unchanged)
    {"context": "the deploy uses {} for orchestration", "mangled": "kubernetes", "intended": "kubernetes", "category": "passthrough"},
    {"context": "we shipped it with {} running locally", "mangled": "Ollama", "intended": "Ollama", "category": "passthrough"},
    {"context": "that guy has serious {} honestly", "mangled": "rizz", "intended": "rizz", "category": "passthrough"},
    {"context": "he plays {} competitively on weekends", "mangled": "Valorant", "intended": "Valorant", "category": "passthrough"},
    {"context": "my name is {} and I write code", "mangled": "Ibrahim", "intended": "Ibrahim", "category": "passthrough"},
    {"context": "the hot path uses {} for spell lookup", "mangled": "SymSpell", "intended": "SymSpell", "category": "passthrough"},
    {"context": "we named the project {} for now", "mangled": "Sumizome", "intended": "Sumizome", "category": "passthrough"},
    {"context": "install it with {} and run the dev server", "mangled": "npm", "intended": "npm", "category": "passthrough"},
    {"context": "the {} handler was blocking the loop", "mangled": "async", "intended": "async", "category": "passthrough"},
    {"context": "she brought {} for lunch again", "mangled": "sushi", "intended": "sushi", "category": "passthrough"},
]

# Guarded homophone context checks: the typed word is VALID but may be the
# wrong member of its confusion group. "wrong" cases must flip; "keep" cases
# (already correct) must stay, which is this feature's passthrough gate.
HOMOPHONE_CASES = [
    {"context": "she visited holland {} last summer", "mangled": "to", "intended": "too", "category": "wrong"},
    {"context": "this is way {} expensive for what it does", "mangled": "to", "intended": "too", "category": "wrong"},
    {"context": "{} going to be late for the meeting", "mangled": "their", "intended": "they're", "category": "wrong"},
    {"context": "I left my keys over {} on the table", "mangled": "their", "intended": "there", "category": "wrong"},
    {"context": "{} not going to believe what happened", "mangled": "your", "intended": "you're", "category": "wrong"},
    {"context": "the company changed {} policy last week", "mangled": "it's", "intended": "its", "category": "wrong"},
    {"context": "I like this option more {} the other one", "mangled": "then", "intended": "than", "category": "wrong"},
    {"context": "be careful not to {} the receipt", "mangled": "loose", "intended": "lose", "category": "wrong"},
    {"context": "I am not sure {} the update is ready", "mangled": "weather", "intended": "whether", "category": "wrong"},
    {"context": "{} did you put the charger", "mangled": "were", "intended": "where", "category": "wrong"},
    # keep cases: correct as typed, the check must not flip them
    {"context": "I went {} the store this morning", "mangled": "to", "intended": "to", "category": "keep"},
    {"context": "the movie was {} long for me", "mangled": "too", "intended": "too", "category": "keep"},
    {"context": "{} car is parked outside", "mangled": "their", "intended": "their", "category": "keep"},
    {"context": "put it over {} by the door", "mangled": "there", "intended": "there", "category": "keep"},
    {"context": "{} welcome to join us", "mangled": "you're", "intended": "you're", "category": "keep"},
    {"context": "the dog wagged {} tail", "mangled": "its", "intended": "its", "category": "keep"},
    {"context": "we finished earlier {} expected", "mangled": "than", "intended": "than", "category": "keep"},
    {"context": "I can {} you loud and clear", "mangled": "hear", "intended": "hear", "category": "keep"},
]
