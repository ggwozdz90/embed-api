from unittest.mock import Mock

import pytest

from application.usecases.load_model_usecase import LoadModelUseCase
from core.config.app_config import AppConfig
from core.logger.logger import Logger
from data.repositories.embedding_model_repository_impl import (
    EmbeddingModelRepositoryImpl,
)


@pytest.fixture
def mock_config() -> AppConfig:
    config = Mock(AppConfig)
    config.embedding_model_name = "BAAI/bge-m3"
    config.device = "cuda"
    return config


@pytest.fixture
def mock_logger() -> Logger:
    return Mock(Logger)


@pytest.fixture
def mock_repository() -> EmbeddingModelRepositoryImpl:
    return Mock(EmbeddingModelRepositoryImpl)


@pytest.fixture
def load_model_usecase(
    mock_config: AppConfig,
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> LoadModelUseCase:
    return LoadModelUseCase(mock_config, mock_logger, mock_repository)


@pytest.mark.asyncio
async def test_execute_success(
    load_model_usecase: LoadModelUseCase,
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given
    mock_repository.start_worker.return_value = None

    # When
    result = await load_model_usecase.execute()

    # Then
    assert result == "Model loaded successfully"
    mock_repository.start_worker.assert_called_once()
    mock_logger.info.assert_any_call("Loading embedding model")
    mock_logger.info.assert_any_call("Embedding model loaded successfully")


@pytest.mark.asyncio
async def test_execute_repository_exception(
    load_model_usecase: LoadModelUseCase,
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given
    mock_repository.start_worker.side_effect = Exception("Worker start error")

    # When & Then
    with pytest.raises(Exception, match="Worker start error"):
        await load_model_usecase.execute()

    mock_repository.start_worker.assert_called_once()
    mock_logger.info.assert_called_once_with("Loading embedding model")


@pytest.mark.asyncio
async def test_execute_with_different_config(
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given
    config = Mock(AppConfig)
    config.embedding_model_name = "custom/model"
    config.device = "cpu"

    usecase = LoadModelUseCase(config, mock_logger, mock_repository)
    mock_repository.start_worker.return_value = None

    # When
    result = await usecase.execute()

    # Then
    assert result == "Model loaded successfully"
    mock_repository.start_worker.assert_called_once()
    mock_logger.info.assert_any_call("Loading embedding model")
    mock_logger.info.assert_any_call("Embedding model loaded successfully")


@pytest.mark.asyncio
async def test_execute_returns_correct_type(
    load_model_usecase: LoadModelUseCase,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given
    mock_repository.start_worker.return_value = None

    # When
    result = await load_model_usecase.execute()

    # Then
    assert isinstance(result, str)
    assert result == "Model loaded successfully"


@pytest.mark.asyncio
async def test_execute_logger_calls_order(
    load_model_usecase: LoadModelUseCase,
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given
    mock_repository.start_worker.return_value = None

    # When
    await load_model_usecase.execute()

    # Then
    expected_calls = [
        ("Loading embedding model",),
        ("Embedding model loaded successfully",),
    ]
    actual_calls = [call.args for call in mock_logger.info.call_args_list]
    assert actual_calls == expected_calls


@pytest.mark.asyncio
async def test_execute_multiple_calls(
    load_model_usecase: LoadModelUseCase,
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given
    mock_repository.start_worker.return_value = None

    # When
    result1 = await load_model_usecase.execute()
    result2 = await load_model_usecase.execute()

    # Then
    assert result1 == "Model loaded successfully"
    assert result2 == "Model loaded successfully"
    assert mock_repository.start_worker.call_count == 2
    assert mock_logger.info.call_count == 4


def test_usecase_initialization(
    mock_config: AppConfig,
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given / When
    usecase = LoadModelUseCase(mock_config, mock_logger, mock_repository)

    # Then
    assert usecase.config is mock_config
    assert usecase.logger is mock_logger
    assert usecase.embedding_repository is mock_repository
