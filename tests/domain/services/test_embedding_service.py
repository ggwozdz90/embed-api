from unittest.mock import Mock

import pytest

from api.dtos.create_embeddings_result_dto import TextEmbedding
from core.config.app_config import AppConfig
from core.logger.logger import Logger
from data.workers.bge_m3_embedding_worker import EmbeddingResult
from domain.repositories.embedding_model_repository import EmbeddingModelRepository
from domain.services.embedding_service import EmbeddingService


@pytest.fixture
def mock_config() -> AppConfig:
    return Mock(AppConfig)


@pytest.fixture
def mock_repository() -> EmbeddingModelRepository:
    return Mock(EmbeddingModelRepository)


@pytest.fixture
def mock_logger() -> Logger:
    return Mock(Logger)


@pytest.fixture
def embedding_service(
    mock_config: AppConfig,
    mock_repository: EmbeddingModelRepository,
    mock_logger: Logger,
) -> EmbeddingService:
    return EmbeddingService(mock_config, mock_repository, mock_logger)


@pytest.fixture
def mock_embedding_result() -> EmbeddingResult:
    embeddings = [
        TextEmbedding(
            text="Hello world",
            dense=[0.1, 0.2, 0.3],
            sparse=None,
            colbert=None,
        ),
        TextEmbedding(
            text="Test text",
            dense=[0.4, 0.5, 0.6],
            sparse=None,
            colbert=None,
        ),
    ]
    return EmbeddingResult(embeddings=embeddings)


def test_embedding_service_initialization(
    mock_config: AppConfig,
    mock_repository: EmbeddingModelRepository,
    mock_logger: Logger,
) -> None:
    # When
    service = EmbeddingService(mock_config, mock_repository, mock_logger)

    # Then
    assert service.config is mock_config
    assert service.embedding_model_repository is mock_repository
    assert service.logger is mock_logger


def test_create_embeddings_success(
    embedding_service: EmbeddingService,
    mock_repository: EmbeddingModelRepository,
    mock_logger: Logger,
    mock_embedding_result: EmbeddingResult,
) -> None:
    # Given
    texts = ["Hello world", "Test text"]
    include_dense = True
    include_sparse = False
    include_colbert = False

    mock_repository.create_embeddings.return_value = mock_embedding_result

    # When
    result = embedding_service.create_embeddings(texts, include_dense, include_sparse, include_colbert)

    # Then
    assert result == mock_embedding_result
    assert len(result.embeddings) == 2
    assert result.embeddings[0].text == "Hello world"
    assert result.embeddings[1].text == "Test text"

    mock_logger.debug.assert_any_call("Starting creation of embeddings")
    mock_logger.debug.assert_any_call("Completed creation of embeddings")
    mock_repository.create_embeddings.assert_called_once_with(texts, include_dense, include_sparse, include_colbert)


def test_create_embeddings_with_all_types(
    embedding_service: EmbeddingService,
    mock_repository: EmbeddingModelRepository,
    mock_logger: Logger,
) -> None:
    # Given
    texts = ["Hello world"]
    include_dense = True
    include_sparse = True
    include_colbert = True

    embeddings = [
        TextEmbedding(
            text="Hello world",
            dense=[0.1, 0.2, 0.3],
            sparse=None,
            colbert=[[0.1, 0.2], [0.3, 0.4]],
        ),
    ]
    expected_result = EmbeddingResult(embeddings=embeddings)
    mock_repository.create_embeddings.return_value = expected_result

    # When
    result = embedding_service.create_embeddings(texts, include_dense, include_sparse, include_colbert)

    # Then
    assert result == expected_result
    assert len(result.embeddings) == 1
    assert result.embeddings[0].text == "Hello world"
    assert result.embeddings[0].dense == [0.1, 0.2, 0.3]
    assert result.embeddings[0].colbert == [[0.1, 0.2], [0.3, 0.4]]

    mock_logger.debug.assert_any_call("Starting creation of embeddings")
    mock_logger.debug.assert_any_call("Completed creation of embeddings")
    mock_repository.create_embeddings.assert_called_once_with(texts, include_dense, include_sparse, include_colbert)


def test_create_embeddings_with_defaults(
    embedding_service: EmbeddingService,
    mock_repository: EmbeddingModelRepository,
    mock_logger: Logger,
    mock_embedding_result: EmbeddingResult,
) -> None:
    # Given
    texts = ["Hello world"]
    mock_repository.create_embeddings.return_value = mock_embedding_result

    # When
    result = embedding_service.create_embeddings(texts)

    # Then
    assert result == mock_embedding_result

    mock_logger.debug.assert_any_call("Starting creation of embeddings")
    mock_logger.debug.assert_any_call("Completed creation of embeddings")
    mock_repository.create_embeddings.assert_called_once_with(texts, True, False, False)


def test_create_embeddings_empty_texts(
    embedding_service: EmbeddingService,
    mock_repository: EmbeddingModelRepository,
    mock_logger: Logger,
) -> None:
    # Given
    texts: list[str] = []
    include_dense = True
    include_sparse = False
    include_colbert = False

    expected_result = EmbeddingResult(embeddings=[])
    mock_repository.create_embeddings.return_value = expected_result

    # When
    result = embedding_service.create_embeddings(texts, include_dense, include_sparse, include_colbert)

    # Then
    assert result == expected_result
    assert len(result.embeddings) == 0

    mock_logger.debug.assert_any_call("Starting creation of embeddings")
    mock_logger.debug.assert_any_call("Completed creation of embeddings")
    mock_repository.create_embeddings.assert_called_once_with(texts, include_dense, include_sparse, include_colbert)


def test_create_embeddings_repository_exception(
    embedding_service: EmbeddingService,
    mock_repository: EmbeddingModelRepository,
    mock_logger: Logger,
) -> None:
    # Given
    texts = ["Hello world"]
    include_dense = True
    include_sparse = False
    include_colbert = False

    mock_repository.create_embeddings.side_effect = Exception("Repository error")

    # When & Then
    with pytest.raises(Exception, match="Repository error"):
        embedding_service.create_embeddings(texts, include_dense, include_sparse, include_colbert)

    mock_logger.debug.assert_called_once_with("Starting creation of embeddings")
    mock_repository.create_embeddings.assert_called_once_with(texts, include_dense, include_sparse, include_colbert)


def test_create_embeddings_single_text(
    embedding_service: EmbeddingService,
    mock_repository: EmbeddingModelRepository,
    mock_logger: Logger,
) -> None:
    # Given
    texts = ["Single text"]
    include_dense = True
    include_sparse = False
    include_colbert = False

    embeddings = [
        TextEmbedding(
            text="Single text",
            dense=[0.7, 0.8, 0.9],
            sparse=None,
            colbert=None,
        ),
    ]
    expected_result = EmbeddingResult(embeddings=embeddings)
    mock_repository.create_embeddings.return_value = expected_result

    # When
    result = embedding_service.create_embeddings(texts, include_dense, include_sparse, include_colbert)

    # Then
    assert result == expected_result
    assert len(result.embeddings) == 1
    assert result.embeddings[0].text == "Single text"
    assert result.embeddings[0].dense == [0.7, 0.8, 0.9]

    mock_logger.debug.assert_any_call("Starting creation of embeddings")
    mock_logger.debug.assert_any_call("Completed creation of embeddings")
    mock_repository.create_embeddings.assert_called_once_with(texts, include_dense, include_sparse, include_colbert)


def test_create_embeddings_dense_only(
    embedding_service: EmbeddingService,
    mock_repository: EmbeddingModelRepository,
    mock_logger: Logger,
) -> None:
    # Given
    texts = ["Hello world"]
    include_dense = True
    include_sparse = False
    include_colbert = False

    embeddings = [
        TextEmbedding(
            text="Hello world",
            dense=[0.1, 0.2, 0.3],
            sparse=None,
            colbert=None,
        ),
    ]
    expected_result = EmbeddingResult(embeddings=embeddings)
    mock_repository.create_embeddings.return_value = expected_result

    # When
    result = embedding_service.create_embeddings(texts, include_dense, include_sparse, include_colbert)

    # Then
    assert result == expected_result
    assert len(result.embeddings) == 1
    assert result.embeddings[0].dense == [0.1, 0.2, 0.3]
    assert result.embeddings[0].sparse is None
    assert result.embeddings[0].colbert is None

    mock_repository.create_embeddings.assert_called_once_with(texts, include_dense, include_sparse, include_colbert)
