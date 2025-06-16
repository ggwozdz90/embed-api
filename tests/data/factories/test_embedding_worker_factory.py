from unittest.mock import Mock

import pytest

from core.config.app_config import AppConfig
from core.logger.logger import Logger
from data.factories.embedding_worker_factory import EmbeddingWorkerFactory
from data.workers.bge_m3_embedding_worker import BgeM3EmbeddingWorker
from domain.exceptions.unsupported_model_configuration_error import (
    UnsupportedModelConfigurationError,
)


@pytest.fixture
def mock_logger() -> Logger:
    return Mock(Logger)


@pytest.fixture
def mock_config() -> AppConfig:
    return Mock(AppConfig)


def test_create_worker_bge_m3(mock_config: AppConfig, mock_logger: Logger) -> None:
    # Given
    mock_config.embedding_model_name = "BAAI/bge-m3"
    mock_config.device = "cpu"
    mock_config.log_level = "INFO"

    factory = EmbeddingWorkerFactory(config=mock_config, logger=mock_logger)

    # When
    worker = factory.create()

    # Then
    assert isinstance(worker, BgeM3EmbeddingWorker)
    assert worker._config.device == "cpu"
    assert worker._config.model_name == "BAAI/bge-m3"
    assert worker._config.log_level == "INFO"


def test_create_worker_bge_m3_cuda(mock_config: AppConfig, mock_logger: Logger) -> None:
    # Given
    mock_config.embedding_model_name = "BAAI/bge-m3"
    mock_config.device = "cuda"
    mock_config.log_level = "DEBUG"

    factory = EmbeddingWorkerFactory(config=mock_config, logger=mock_logger)

    # When
    worker = factory.create()

    # Then
    assert isinstance(worker, BgeM3EmbeddingWorker)
    assert worker._config.device == "cuda"
    assert worker._config.model_name == "BAAI/bge-m3"
    assert worker._config.log_level == "DEBUG"


def test_create_worker_unsupported_model(mock_config: AppConfig, mock_logger: Logger) -> None:
    # Given
    mock_config.embedding_model_name = "unsupported-model"
    factory = EmbeddingWorkerFactory(config=mock_config, logger=mock_logger)

    # When / Then
    with pytest.raises(UnsupportedModelConfigurationError, match="Unsupported model configuration: unsupported-model"):
        factory.create()


def test_create_worker_empty_model_name(mock_config: AppConfig, mock_logger: Logger) -> None:
    # Given
    mock_config.embedding_model_name = ""
    factory = EmbeddingWorkerFactory(config=mock_config, logger=mock_logger)

    # When / Then
    with pytest.raises(UnsupportedModelConfigurationError, match="Unsupported model configuration: "):
        factory.create()


def test_create_worker_none_model_name(mock_config: AppConfig, mock_logger: Logger) -> None:
    # Given
    mock_config.embedding_model_name = None
    factory = EmbeddingWorkerFactory(config=mock_config, logger=mock_logger)

    # When / Then
    with pytest.raises(UnsupportedModelConfigurationError, match="Unsupported model configuration: None"):
        factory.create()


def test_factory_initialization(mock_config: AppConfig, mock_logger: Logger) -> None:
    # Given / When
    factory = EmbeddingWorkerFactory(config=mock_config, logger=mock_logger)

    # Then
    assert factory.config is mock_config
    assert factory.logger is mock_logger


def test_create_worker_case_sensitive_model_name(mock_config: AppConfig, mock_logger: Logger) -> None:
    # Given
    mock_config.embedding_model_name = "baai/bge-m3"  # lowercase
    factory = EmbeddingWorkerFactory(config=mock_config, logger=mock_logger)

    # When / Then
    with pytest.raises(UnsupportedModelConfigurationError, match="Unsupported model configuration: baai/bge-m3"):
        factory.create()


def test_create_worker_similar_model_name(mock_config: AppConfig, mock_logger: Logger) -> None:
    # Given
    mock_config.embedding_model_name = "BAAI/bge-m3-v2"  # Similar but different
    factory = EmbeddingWorkerFactory(config=mock_config, logger=mock_logger)

    # When / Then
    with pytest.raises(UnsupportedModelConfigurationError, match="Unsupported model configuration: BAAI/bge-m3-v2"):
        factory.create()
