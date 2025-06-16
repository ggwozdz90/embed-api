from unittest.mock import Mock

import pytest

from application.usecases.unload_model_usecase import UnloadModelUseCase
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
def unload_model_usecase(
    mock_config: AppConfig,
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> UnloadModelUseCase:
    return UnloadModelUseCase(mock_config, mock_logger, mock_repository)


@pytest.mark.asyncio
async def test_execute_success(
    unload_model_usecase: UnloadModelUseCase,
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given
    mock_repository.stop_worker.return_value = None

    # When
    result = await unload_model_usecase.execute()

    # Then
    assert result == "Model unloaded successfully"
    mock_repository.stop_worker.assert_called_once()
    mock_logger.info.assert_any_call("Unloading embedding model")
    mock_logger.info.assert_any_call("Embedding model unloaded successfully")


@pytest.mark.asyncio
async def test_execute_repository_exception(
    unload_model_usecase: UnloadModelUseCase,
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given
    mock_repository.stop_worker.side_effect = Exception("Worker stop error")

    # When & Then
    with pytest.raises(Exception, match="Worker stop error"):
        await unload_model_usecase.execute()

    mock_repository.stop_worker.assert_called_once()
    mock_logger.info.assert_called_once_with("Unloading embedding model")


@pytest.mark.asyncio
async def test_execute_with_different_config(
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given
    config = Mock(AppConfig)
    config.embedding_model_name = "custom/model"
    config.device = "cpu"

    usecase = UnloadModelUseCase(config, mock_logger, mock_repository)
    mock_repository.stop_worker.return_value = None

    # When
    result = await usecase.execute()

    # Then
    assert result == "Model unloaded successfully"
    mock_repository.stop_worker.assert_called_once()
    mock_logger.info.assert_any_call("Unloading embedding model")
    mock_logger.info.assert_any_call("Embedding model unloaded successfully")


@pytest.mark.asyncio
async def test_execute_returns_correct_type(
    unload_model_usecase: UnloadModelUseCase,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given
    mock_repository.stop_worker.return_value = None

    # When
    result = await unload_model_usecase.execute()

    # Then
    assert isinstance(result, str)
    assert result == "Model unloaded successfully"


@pytest.mark.asyncio
async def test_execute_logger_calls_order(
    unload_model_usecase: UnloadModelUseCase,
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given
    mock_repository.stop_worker.return_value = None

    # When
    await unload_model_usecase.execute()

    # Then
    expected_calls = [
        ("Unloading embedding model",),
        ("Embedding model unloaded successfully",),
    ]
    actual_calls = [call.args for call in mock_logger.info.call_args_list]
    assert actual_calls == expected_calls


@pytest.mark.asyncio
async def test_execute_multiple_calls(
    unload_model_usecase: UnloadModelUseCase,
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given
    mock_repository.stop_worker.return_value = None

    # When
    result1 = await unload_model_usecase.execute()
    result2 = await unload_model_usecase.execute()

    # Then
    assert result1 == "Model unloaded successfully"
    assert result2 == "Model unloaded successfully"
    assert mock_repository.stop_worker.call_count == 2
    assert mock_logger.info.call_count == 4


def test_usecase_initialization(
    mock_config: AppConfig,
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given / When
    usecase = UnloadModelUseCase(mock_config, mock_logger, mock_repository)

    # Then
    assert usecase.config is mock_config
    assert usecase.logger is mock_logger
    assert usecase.embedding_repository is mock_repository


@pytest.mark.asyncio
async def test_execute_worker_already_stopped(
    unload_model_usecase: UnloadModelUseCase,
    mock_logger: Logger,
    mock_repository: EmbeddingModelRepositoryImpl,
) -> None:
    # Given
    mock_repository.stop_worker.return_value = None

    # When
    result = await unload_model_usecase.execute()

    # Then
    assert result == "Model unloaded successfully"
    mock_repository.stop_worker.assert_called_once()
    mock_logger.info.assert_any_call("Unloading embedding model")
    mock_logger.info.assert_any_call("Embedding model unloaded successfully")
