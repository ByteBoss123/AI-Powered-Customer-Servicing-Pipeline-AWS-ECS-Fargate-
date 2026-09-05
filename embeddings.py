"""
T5EncoderEmbedding: a LlamaIndex-compatible embedding model that reuses the
flan-t5-small encoder (mean-pooled last hidden state) to produce embeddings.

Design note: LlamaIndex's default embedding backends assume network access to
OpenAI or the Hugging Face Hub. This sandbox has neither. Rather than fabricate
a numeric embedding function, this wraps the *actual* flan-t5-small encoder
weights (verified real, downloaded by the user from google/flan-t5-small) to
produce genuine model-derived embeddings, avoiding a second network-blocked
model download for a dedicated embedding model.
"""
import torch
from typing import List
from llama_index.core.embeddings import BaseEmbedding
from transformers import T5Tokenizer, T5EncoderModel

MODEL_PATH = "/home/claude/models/flan-t5-small/flan-t5-small"


class T5EncoderEmbedding(BaseEmbedding):
    _tokenizer: object = None
    _encoder: object = None

    def __init__(self, model_path: str = MODEL_PATH, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_tokenizer", T5Tokenizer.from_pretrained(model_path))
        object.__setattr__(self, "_encoder", T5EncoderModel.from_pretrained(model_path))
        self._encoder.eval()

    def _embed(self, text: str) -> List[float]:
        with torch.no_grad():
            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            outputs = self._encoder(**inputs)
            # mean pooling over token dimension
            hidden = outputs.last_hidden_state.squeeze(0)  # (seq_len, d_model)
            pooled = hidden.mean(dim=0)
            return pooled.tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)
