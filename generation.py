"""
Loads the real flan-t5-small weights (downloaded by the user, verified working)
and generates source-grounded responses conditioned on retrieved policy text.
"""
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_PATH = "/home/claude/models/flan-t5-small/flan-t5-small"

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
        _model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
        _model.eval()
    return _tokenizer, _model


def generate_grounded_response(question: str, context_chunks: list, max_new_tokens: int = 64) -> str:
    tokenizer, model = _load()
    context = "\n".join(c["text"] for c in context_chunks)
    prompt = (
        "Answer the question using ONLY the policy text below. "
        "If the policy text does not answer the question, say you are not sure and that a specialist will follow up.\n\n"
        f"Policy text:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
