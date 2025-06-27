from abc import ABC
from typing import List
from unittest.mock import Mock

import pytest

from data.workers.bge_m3_embedding_worker import EmbeddingResult
from domain.repositories.embedding_model_repository import EmbeddingModelRepository


class ConcreteEmbeddingModelRepository(EmbeddingModelRepository):  # type: ignore
    """Concrete implementation for testing abstract base class"""

    def __init__(self) -> None:
        self._mock_create_embeddings = Mock()
        self._mock_is_model_loaded = Mock()

    def create_embeddings(
        self,
        texts: List[str],
        include_dense: bool = True,
        include_sparse: bool = False,
        include_colbert: bool = False,
    ) -> EmbeddingResult:
        return self._mock_create_embeddings(texts, include_dense, include_sparse, include_colbert)

    def is_model_loaded(self) -> bool:
        return bool(self._mock_is_model_loaded())


@pytest.fixture
def repository() -> ConcreteEmbeddingModelRepository:
    return ConcreteEmbeddingModelRepository()


def test_embedding_model_repository_is_abstract() -> None:
    # Given / When / Then
    assert issubclass(EmbeddingModelRepository, ABC)

    with pytest.raises(TypeError):
        EmbeddingModelRepository()


def test_create_embeddings_method_signature(repository: ConcreteEmbeddingModelRepository) -> None:
    # Given
    texts = ["test text"]
    mock_result = Mock(spec=EmbeddingResult)
    repository._mock_create_embeddings.return_value = mock_result

    # When
    result = repository.create_embeddings(texts)

    # Then
    repository._mock_create_embeddings.assert_called_once_with(texts, True, False, False)
    assert result == mock_result


def test_create_embeddings_with_all_parameters(repository: ConcreteEmbeddingModelRepository) -> None:
    # Given
    texts = ["test text 1", "test text 2"]
    mock_result = Mock(spec=EmbeddingResult)
    repository._mock_create_embeddings.return_value = mock_result

    # When
    result = repository.create_embeddings(texts=texts, include_dense=True, include_sparse=True, include_colbert=True)

    # Then
    repository._mock_create_embeddings.assert_called_once_with(texts, True, True, True)
    assert result == mock_result


def test_is_model_loaded_method(repository: ConcreteEmbeddingModelRepository) -> None:
    # Given
    repository._mock_is_model_loaded.return_value = True

    # When
    result = repository.is_model_loaded()

    # Then
    repository._mock_is_model_loaded.assert_called_once()
    assert result is True
