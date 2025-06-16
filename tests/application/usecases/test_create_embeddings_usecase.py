from unittest.mock import Mock

import pytest

from api.dtos.create_embeddings_result_dto import TextEmbedding
from application.usecases.create_embeddings_usecase import CreateEmbeddingsUseCase
from core.config.app_config import AppConfig
from core.logger.logger import Logger
from data.workers.bge_m3_embedding_worker import EmbeddingResult
from domain.services.embedding_service import EmbeddingService


@pytest.fixture
def mock_config() -> AppConfig:
    return Mock(AppConfig)


@pytest.fixture
def mock_logger() -> Logger:
    return Mock(Logger)


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    return Mock(EmbeddingService)


@pytest.fixture
def create_embeddings_usecase(
    mock_config: AppConfig,
    mock_logger: Logger,
    mock_embedding_service: EmbeddingService,
) -> CreateEmbeddingsUseCase:
    return CreateEmbeddingsUseCase(mock_config, mock_logger, mock_embedding_service)


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


@pytest.mark.asyncio
async def test_execute_success(
    create_embeddings_usecase: CreateEmbeddingsUseCase,
    mock_logger: Logger,
    mock_embedding_service: EmbeddingService,
    mock_embedding_result: EmbeddingResult,
) -> None:
    # Given
    texts = ["Hello world", "Test text"]
    include_dense = True
    include_sparse = False
    include_colbert = False

    mock_embedding_service.create_embeddings.return_value = mock_embedding_result

    # When
    result = await create_embeddings_usecase.execute(texts, include_dense, include_sparse, include_colbert)

    # Then
    assert result == mock_embedding_result
    assert len(result.embeddings) == 2
    assert result.embeddings[0].text == "Hello world"
    assert result.embeddings[1].text == "Test text"

    mock_logger.info.assert_any_call("Executing embedding creation for 2 texts")
    mock_logger.info.assert_any_call("Returning embedding creation result")
    mock_embedding_service.create_embeddings.assert_called_once_with(
        texts,
        include_dense,
        include_sparse,
        include_colbert,
    )


@pytest.mark.asyncio
async def test_execute_with_all_embedding_types(
    create_embeddings_usecase: CreateEmbeddingsUseCase,
    mock_logger: Logger,
    mock_embedding_service: EmbeddingService,
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
    mock_embedding_service.create_embeddings.return_value = expected_result

    # When
    result = await create_embeddings_usecase.execute(texts, include_dense, include_sparse, include_colbert)

    # Then
    assert result == expected_result
    assert len(result.embeddings) == 1
    assert result.embeddings[0].text == "Hello world"
    assert result.embeddings[0].dense == [0.1, 0.2, 0.3]
    assert result.embeddings[0].colbert == [[0.1, 0.2], [0.3, 0.4]]

    mock_logger.info.assert_any_call("Executing embedding creation for 1 texts")
    mock_logger.info.assert_any_call("Returning embedding creation result")
    mock_embedding_service.create_embeddings.assert_called_once_with(
        texts,
        include_dense,
        include_sparse,
        include_colbert,
    )


@pytest.mark.asyncio
async def test_execute_empty_texts(
    create_embeddings_usecase: CreateEmbeddingsUseCase,
    mock_logger: Logger,
    mock_embedding_service: EmbeddingService,
) -> None:
    # Given
    texts: list[str] = []
    include_dense = True
    include_sparse = False
    include_colbert = False

    expected_result = EmbeddingResult(embeddings=[])
    mock_embedding_service.create_embeddings.return_value = expected_result

    # When
    result = await create_embeddings_usecase.execute(texts, include_dense, include_sparse, include_colbert)

    # Then
    assert result == expected_result
    assert len(result.embeddings) == 0

    mock_logger.info.assert_any_call("Executing embedding creation for 0 texts")
    mock_logger.info.assert_any_call("Returning embedding creation result")
    mock_embedding_service.create_embeddings.assert_called_once_with(
        texts,
        include_dense,
        include_sparse,
        include_colbert,
    )


@pytest.mark.asyncio
async def test_execute_with_defaults(
    create_embeddings_usecase: CreateEmbeddingsUseCase,
    mock_logger: Logger,
    mock_embedding_service: EmbeddingService,
    mock_embedding_result: EmbeddingResult,
) -> None:
    # Given
    texts = ["Hello world"]
    mock_embedding_service.create_embeddings.return_value = mock_embedding_result

    # When
    result = await create_embeddings_usecase.execute(texts)

    # Then
    assert result == mock_embedding_result

    mock_logger.info.assert_any_call("Executing embedding creation for 1 texts")
    mock_logger.info.assert_any_call("Returning embedding creation result")
    mock_embedding_service.create_embeddings.assert_called_once_with(texts, True, False, False)


@pytest.mark.asyncio
async def test_execute_service_exception(
    create_embeddings_usecase: CreateEmbeddingsUseCase,
    mock_logger: Logger,
    mock_embedding_service: EmbeddingService,
) -> None:
    # Given
    texts = ["Hello world"]
    include_dense = True
    include_sparse = False
    include_colbert = False

    mock_embedding_service.create_embeddings.side_effect = Exception("Service error")

    # When & Then
    with pytest.raises(Exception, match="Service error"):
        await create_embeddings_usecase.execute(texts, include_dense, include_sparse, include_colbert)

    mock_logger.info.assert_called_once_with("Executing embedding creation for 1 texts")
    mock_embedding_service.create_embeddings.assert_called_once_with(
        texts,
        include_dense,
        include_sparse,
        include_colbert,
    )


@pytest.mark.asyncio
async def test_execute_single_text(
    create_embeddings_usecase: CreateEmbeddingsUseCase,
    mock_logger: Logger,
    mock_embedding_service: EmbeddingService,
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
    mock_embedding_service.create_embeddings.return_value = expected_result

    # When
    result = await create_embeddings_usecase.execute(texts, include_dense, include_sparse, include_colbert)

    # Then
    assert result == expected_result
    assert len(result.embeddings) == 1
    assert result.embeddings[0].text == "Single text"
    assert result.embeddings[0].dense == [0.7, 0.8, 0.9]

    mock_logger.info.assert_any_call("Executing embedding creation for 1 texts")
    mock_logger.info.assert_any_call("Returning embedding creation result")
    mock_embedding_service.create_embeddings.assert_called_once_with(
        texts,
        include_dense,
        include_sparse,
        include_colbert,
    )


@pytest.mark.asyncio
async def test_execute_multiple_texts(
    create_embeddings_usecase: CreateEmbeddingsUseCase,
    mock_logger: Logger,
    mock_embedding_service: EmbeddingService,
) -> None:
    # Given
    texts = ["Text 1", "Text 2", "Text 3"]
    include_dense = True
    include_sparse = False
    include_colbert = False

    embeddings = [
        TextEmbedding(text="Text 1", dense=[0.1, 0.2], sparse=None, colbert=None),
        TextEmbedding(text="Text 2", dense=[0.3, 0.4], sparse=None, colbert=None),
        TextEmbedding(text="Text 3", dense=[0.5, 0.6], sparse=None, colbert=None),
    ]
    expected_result = EmbeddingResult(embeddings=embeddings)
    mock_embedding_service.create_embeddings.return_value = expected_result

    # When
    result = await create_embeddings_usecase.execute(texts, include_dense, include_sparse, include_colbert)

    # Then
    assert result == expected_result
    assert len(result.embeddings) == 3
    assert all(emb.text.startswith("Text") for emb in result.embeddings)

    mock_logger.info.assert_any_call("Executing embedding creation for 3 texts")
    mock_logger.info.assert_any_call("Returning embedding creation result")
    mock_embedding_service.create_embeddings.assert_called_once_with(
        texts,
        include_dense,
        include_sparse,
        include_colbert,
    )
