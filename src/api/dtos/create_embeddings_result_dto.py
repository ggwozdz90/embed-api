from typing import List, Optional

from pydantic import BaseModel


class SparseEmbedding(BaseModel):
    indices: List[int]
    values: List[float]


class TextEmbedding(BaseModel):
    text: str
    dense: Optional[List[float]] = None
    sparse: Optional[SparseEmbedding] = None
    colbert: Optional[List[List[float]]] = None


class CreateEmbeddingsResultDto(BaseModel):
    embeddings: List[TextEmbedding]
