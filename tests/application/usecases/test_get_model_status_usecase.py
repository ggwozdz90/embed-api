from unittest.mock import Mock

import pytest

from application.usecases.get_model_status_usecase import GetModelStatusUseCase
from core.config.app_config import AppConfig
from domain.repositories.embedding_model_repository import EmbeddingModelRepository


@pytest.fixture
def mock_config() -> AppConfig:
    config = Mock(AppConfig)
    config.embedding_model_name = "BAAI/bge-m3"
    config.device = "cuda"
    return config


@pytest.fixture
def mock_repository() -> EmbeddingModelRepository:
    return Mock(EmbeddingModelRepository)


@pytest.fixture
def get_model_status_usecase(
    mock_config: AppConfig,
    mock_repository: EmbeddingModelRepository,
) -> GetModelStatusUseCase:
    return GetModelStatusUseCase(mock_config, mock_repository)


@pytest.mark.asyncio
async def test_execute_model_loaded(
    get_model_status_usecase: GetModelStatusUseCase,
    mock_config: AppConfig,
    mock_repository: EmbeddingModelRepository,
) -> None:
    # Given
    mock_repository.is_model_loaded.return_value = True

    # When
    result = await get_model_status_usecase.execute()

    # Then
    expected = {
        "is_loaded": True,
        "model_name": "BAAI/bge-m3",
        "device": "cuda",
    }
    assert result == expected
    mock_repository.is_model_loaded.assert_called_once()


@pytest.mark.asyncio
async def test_execute_model_not_loaded(
    get_model_status_usecase: GetModelStatusUseCase,
    mock_config: AppConfig,
    mock_repository: EmbeddingModelRepository,
) -> None:
    # Given
    mock_repository.is_model_loaded.return_value = False

    # When
    result = await get_model_status_usecase.execute()

    # Then
    expected = {
        "is_loaded": False,
        "model_name": "BAAI/bge-m3",
        "device": "cuda",
    }
    assert result == expected
    mock_repository.is_model_loaded.assert_called_once()


@pytest.mark.asyncio
async def test_execute_with_different_config(
    mock_repository: EmbeddingModelRepository,
) -> None:
    # Given
    config = Mock(AppConfig)
    config.embedding_model_name = "custom/model"
    config.device = "cpu"

    usecase = GetModelStatusUseCase(config, mock_repository)
    mock_repository.is_model_loaded.return_value = True

    # When
    result = await usecase.execute()

    # Then
    expected = {
        "is_loaded": True,
        "model_name": "custom/model",
        "device": "cpu",
    }
    assert result == expected
    mock_repository.is_model_loaded.assert_called_once()


@pytest.mark.asyncio
async def test_execute_repository_exception(
    get_model_status_usecase: GetModelStatusUseCase,
    mock_repository: EmbeddingModelRepository,
) -> None:
    # Given
    mock_repository.is_model_loaded.side_effect = Exception("Repository error")

    # When & Then
    with pytest.raises(Exception, match="Repository error"):
        await get_model_status_usecase.execute()

    mock_repository.is_model_loaded.assert_called_once()


@pytest.mark.asyncio
async def test_execute_returns_correct_types(
    get_model_status_usecase: GetModelStatusUseCase,
    mock_repository: EmbeddingModelRepository,
) -> None:
    # Given
    mock_repository.is_model_loaded.return_value = True

    # When
    result = await get_model_status_usecase.execute()

    # Then
    assert isinstance(result, dict)
    assert isinstance(result["is_loaded"], bool)
    assert isinstance(result["model_name"], str)
    assert isinstance(result["device"], str)
    assert "is_loaded" in result
    assert "model_name" in result
    assert "device" in result
