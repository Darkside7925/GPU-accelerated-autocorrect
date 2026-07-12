"""Fixed benchmark test set for the autocorrect LLM harness.

Each case:
  input     - text as a fast typist would actually produce it
  expected  - the ideal corrected output (used for normalized exact-match)
  required  - substrings that MUST appear in the model output (the fixes)
  forbidden - substrings that must NOT appear (the original errors)
  category  - "typo" (pure spelling) or "context" (their/there, grammar)

Scoring uses word-boundary matching on required/forbidden so "teh" doesn't
false-positive inside "the".
"""

TEST_CASES = [
    # --- pure typos / fast-typing transpositions ---
    {
        "input": "I jsut finished teh report and sent it to him",
        "expected": "I just finished the report and sent it to him",
        "required": ["just", "the"],
        "forbidden": ["jsut", "teh"],
        "category": "typo",
    },
    {
        "input": "can you chekc if hte server is runing",
        "expected": "can you check if the server is running",
        "required": ["check", "the", "running"],
        "forbidden": ["chekc", "hte", "runing"],
        "category": "typo",
    },
    {
        "input": "I definately think we shoudl seperate these files",
        "expected": "I definitely think we should separate these files",
        "required": ["definitely", "should", "separate"],
        "forbidden": ["definately", "shoudl", "seperate"],
        "category": "typo",
    },
    {
        "input": "did you recieve my emial about the meetign",
        "expected": "did you receive my email about the meeting",
        "required": ["receive", "email", "meeting"],
        "forbidden": ["recieve", "emial", "meetign"],
        "category": "typo",
    },
    {
        "input": "taht was probaly the best aproach",
        "expected": "that was probably the best approach",
        "required": ["that", "probably", "approach"],
        "forbidden": ["taht", "probaly", "aproach"],
        "category": "typo",
    },
    {
        "input": "my freind lives at a diferent adress now",
        "expected": "my friend lives at a different address now",
        "required": ["friend", "different", "address"],
        "forbidden": ["freind", "diferent", "adress"],
        "category": "typo",
    },
    {
        "input": "we need to acommodate the new schedual",
        "expected": "we need to accommodate the new schedule",
        "required": ["accommodate", "schedule"],
        "forbidden": ["acommodate", "schedual"],
        "category": "typo",
    },
    {
        "input": "becuase of the wierd bug the app keeps crashign",
        "expected": "because of the weird bug the app keeps crashing",
        "required": ["because", "weird", "crashing"],
        "forbidden": ["becuase", "wierd", "crashign"],
        "category": "typo",
    },
    {
        "input": "i woudl appriciate a quick reveiw of this",
        "expected": "I would appreciate a quick review of this",
        "required": ["would", "appreciate", "review"],
        "forbidden": ["woudl", "appriciate", "reveiw"],
        "category": "typo",
    },
    {
        "input": "the goverment anounced a new enviroment policy",
        "expected": "the government announced a new environment policy",
        "required": ["government", "announced", "environment"],
        "forbidden": ["goverment", "anounced", "enviroment"],
        "category": "typo",
    },
    # --- context / homophone / grammar errors (SymSpell can't catch these) ---
    {
        "input": "their going to be late for the meeting",
        "expected": "they're going to be late for the meeting",
        "required": ["they're"],
        "forbidden": [],  # "their" alone is a valid word; require the fix only
        "category": "context",
    },
    {
        "input": "I left my keys over their on the table",
        "expected": "I left my keys over there on the table",
        "required": ["there"],
        "forbidden": ["their"],
        "category": "context",
    },
    {
        "input": "your not going to believe what happened",
        "expected": "you're not going to believe what happened",
        "required": ["you're"],
        "forbidden": [],
        "category": "context",
    },
    {
        "input": "the company changed it's policy last week",
        "expected": "the company changed its policy last week",
        "required": ["its"],
        "forbidden": ["it's"],
        "category": "context",
    },
    {
        "input": "this is way to expensive for what it does",
        "expected": "this is way too expensive for what it does",
        "required": ["too"],
        "forbidden": [],
        "category": "context",
    },
    {
        "input": "I like this option more then the other one",
        "expected": "I like this option more than the other one",
        "required": ["than"],
        "forbidden": [],
        "category": "context",
    },
    {
        "input": "be careful not to loose the receipt",
        "expected": "be careful not to lose the receipt",
        "required": ["lose"],
        "forbidden": ["loose"],
        "category": "context",
    },
    {
        "input": "the new update effects performance quite a bit",
        "expected": "the new update affects performance quite a bit",
        "required": ["affects"],
        "forbidden": ["effects"],
        "category": "context",
    },
    {
        "input": "he should of told me about this earlier",
        "expected": "he should have told me about this earlier",
        "required": ["should have"],
        "forbidden": ["should of"],
        "category": "context",
    },
    {
        "input": "me and him went to the store yesterday",
        "expected": "he and I went to the store yesterday",
        "required": [],  # multiple acceptable rewrites; exact-match handles it
        "forbidden": ["me and him"],
        "category": "context",
    },
    # --- mixed: typo + context in one sentence (the realistic hard case) ---
    {
        "input": "im not sure weather teh update is ready",
        "expected": "I'm not sure whether the update is ready",
        "required": ["whether", "the"],
        "forbidden": ["weather", "teh"],
        "category": "context",
    },
    {
        "input": "thier team did really good on there benchmark",
        "expected": "their team did really well on their benchmark",
        "required": ["their"],
        "forbidden": ["thier"],
        "category": "context",
    },
    # --- must-NOT-touch cases: correct text should pass through unchanged ---
    {
        "input": "the quick brown fox jumps over the lazy dog",
        "expected": "the quick brown fox jumps over the lazy dog",
        "required": ["quick brown fox"],
        "forbidden": [],
        "category": "passthrough",
    },
    {
        "input": "I pushed the fix to the staging branch an hour ago",
        "expected": "I pushed the fix to the staging branch an hour ago",
        "required": ["staging branch"],
        "forbidden": [],
        "category": "passthrough",
    },
]
