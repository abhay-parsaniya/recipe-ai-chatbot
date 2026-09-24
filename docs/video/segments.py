"""Final narration for the LLM-enabled screencast.

Word budgets are tuned so the code half and the demo half each land near
60 seconds, and so every demo beat is long enough for a Gemini reply
(~6s) to actually render before the narration moves on.
"""

SEGMENTS = [
    # ---- code (~58s) -------------------------------------------------
    ("overview",
     "A recipe chatbot over RecipeNLG. Three hundred and seventy two thousand "
     "recipes, searched locally. A model phrases the answer, but never picks "
     "the recipes."),
    ("engine",
     "Retrieval is T F I D F and cosine, with two corrections. A coverage "
     "boost, since cosine favours short recipes. And a title bonus, so banana "
     "bread beats a banana sandwich."),
    ("guardrails",
     "A rule based guardrail refuses anything outside cooking, before any "
     "search or model call."),
    ("grounding",
     "This is the whole contract. Use only the retrieved recipes, never invent "
     "one. Retrieval decides what is true; the model decides how to say it."),
    ("provider",
     "Swapping Anthropic for Gemini took one module and one line. No vendor "
     "S D K is imported outside this folder."),
    ("verify",
     "Prompting is not proof, so every reply is checked. Cite a recipe that "
     "was never retrieved, and it is discarded for the template."),

    # ---- demo (~58s) -------------------------------------------------
    ("demo_search",
     "Now the live app, Gemini switched on. Plain English in. Retrieval runs "
     "first, and only those five recipes reach the model. The caption under "
     "the reply marks the wording as model generated."),
    ("demo_followup",
     "It holds context. I pick one by number, then ask how to make it, and get "
     "the real method from the dataset, rephrased but not invented."),
    ("demo_guardrail",
     "Off topic questions never reach the model. The guardrail answers them, "
     "using no quota."),
    ("demo_pantry",
     "And the pantry. I say what is in my kitchen, it ranks by how much of "
     "each recipe I already have, and the model explains that shortlist."),
    ("demo_api",
     "The same engine over H T T P, with health reporting the live model."),
]
