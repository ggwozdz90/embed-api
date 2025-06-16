from unittest.mock import Mock, patch

import pytest

from core.config.app_config import AppConfig
from core.logger.logger import Logger
from core.timer.timer import Timer, TimerFactory
from data.factories.embedding_worker_factory import EmbeddingWorkerFactory
from data.repositories.embedding_model_repository_impl import (
    EmbeddingModelRepositoryImpl,
)
from data.workers.bge_m3_embedding_worker import (
    BgeM3EmbeddingWorker,
    EmbeddingRequest,
    EmbeddingResult,
)


@pytest.fixture
def mock_config() -> AppConfig:
    config = Mock(AppConfig)
    config.log_level = "INFO"
    config.embedding_model_name = "BAAI/bge-m3"
    config.device = "cpu"
    config.model_idle_timeout = 60
    return config


@pytest.fixture
def mock_timer() -> Mock:
    return Mock(spec=Timer)


@pytest.fixture
def mock_timer_factory(mock_timer: Mock) -> Mock:
    factory = Mock(spec=TimerFactory)
    factory.create.return_value = mock_timer
    return factory


@pytest.fixture
def mock_logger() -> Mock:
    return Mock(spec=Logger)


@pytest.fixture
def mock_worker() -> Mock:
    return Mock(spec=BgeM3EmbeddingWorker)


@pytest.fixture
def mock_worker_factory(mock_worker: Mock) -> Mock:
    factory = Mock(spec=EmbeddingWorkerFactory)
    factory.create.return_value = mock_worker
    return factory


@pytest.fixture
def embedding_repository_impl(
    mock_config: Mock,
    mock_timer_factory: Mock,
    mock_logger: Mock,
    mock_worker_factory: Mock,
) -> EmbeddingModelRepositoryImpl:
    with patch.object(EmbeddingModelRepositoryImpl, "_instance", None):
        return EmbeddingModelRepositoryImpl(
            config=mock_config,
            timer_factory=mock_timer_factory,
            logger=mock_logger,
            worker_factory=mock_worker_factory,
        )


@pytest.fixture
def sample_embedding_result() -> EmbeddingResult:
    return EmbeddingResult(
        embeddings=[
            {
                "text": "Hello world",
                "dense": [0.1, 0.2, 0.3],
                "sparse": {"indices": [1, 2, 3], "values": [0.5, 0.6, 0.7]},
            },
        ],
    )


def test_create_embeddings_success_worker_not_alive(
    embedding_repository_impl: EmbeddingModelRepositoryImpl,
    mock_worker: Mock,
    mock_timer: Mock,
    sample_embedding_result: EmbeddingResult,
) -> None:
    # Given
    mock_worker.is_alive.return_value = False
    mock_worker.create_embeddings.return_value = sample_embedding_result
    texts = ["Hello world"]

    # When
    with patch("time.time", return_value=1234567890.0):
        result = embedding_repository_impl.create_embeddings(
            texts=texts,
            include_dense=True,
            include_sparse=False,
            include_colbert=False,
        )

    # Then
    assert result == sample_embedding_result
    mock_worker.start.assert_called_once()

    mock_worker.create_embeddings.assert_called_once()
    call_args = mock_worker.create_embeddings.call_args[0][0]
    assert isinstance(call_args, EmbeddingRequest)
    assert call_args.texts == texts
    assert call_args.include_dense is True
    assert call_args.include_sparse is False
    assert call_args.include_colbert is False

    mock_timer.start.assert_called_once_with(60, embedding_repository_impl._check_idle_timeout)
    assert embedding_repository_impl.last_access_time == 1234567890.0


def test_create_embeddings_success_worker_alive(
    embedding_repository_impl: EmbeddingModelRepositoryImpl,
    mock_worker: Mock,
    mock_timer: Mock,
    sample_embedding_result: EmbeddingResult,
) -> None:
    # Given
    mock_worker.is_alive.return_value = True
    mock_worker.create_embeddings.return_value = sample_embedding_result
    texts = ["Hello world", "Test text"]

    # When
    result = embedding_repository_impl.create_embeddings(
        texts=texts,
        include_dense=True,
        include_sparse=True,
        include_colbert=True,
    )

    # Then
    assert result == sample_embedding_result
    mock_worker.start.assert_not_called()
    mock_worker.create_embeddings.assert_called_once()
    call_args = mock_worker.create_embeddings.call_args[0][0]
    assert isinstance(call_args, EmbeddingRequest)
    assert call_args.texts == texts
    assert call_args.include_dense is True
    assert call_args.include_sparse is True
    assert call_args.include_colbert is True

    mock_timer.start.assert_called_once_with(60, embedding_repository_impl._check_idle_timeout)


def test_create_embeddings_default_parameters(
    embedding_repository_impl: EmbeddingModelRepositoryImpl,
    mock_worker: Mock,
    sample_embedding_result: EmbeddingResult,
) -> None:
    # Given
    mock_worker.is_alive.return_value = True
    mock_worker.create_embeddings.return_value = sample_embedding_result
    texts = ["Test text"]

    # When
    result = embedding_repository_impl.create_embeddings(texts=texts)

    # Then
    assert result == sample_embedding_result

    call_args = mock_worker.create_embeddings.call_args[0][0]
    assert call_args.include_dense is True  # Default
    assert call_args.include_sparse is False  # Default
    assert call_args.include_colbert is False  # Default


def test_start_worker_not_alive(
    embedding_repository_impl: EmbeddingModelRepositoryImpl,
    mock_worker: Mock,
    mock_logger: Mock,
) -> None:
    # Given
    mock_worker.is_alive.return_value = False

    # When
    embedding_repository_impl.start_worker()

    # Then
    mock_worker.start.assert_called_once()
    mock_logger.info.assert_called_with("Starting embedding worker")


def test_start_worker_already_alive(
    embedding_repository_impl: EmbeddingModelRepositoryImpl,
    mock_worker: Mock,
    mock_logger: Mock,
) -> None:
    # Given
    mock_worker.is_alive.return_value = True

    # When
    embedding_repository_impl.start_worker()

    # Then
    mock_worker.start.assert_not_called()
    mock_logger.info.assert_not_called()


def test_stop_worker_alive(
    embedding_repository_impl: EmbeddingModelRepositoryImpl,
    mock_worker: Mock,
    mock_timer: Mock,
    mock_logger: Mock,
) -> None:
    # Given
    mock_worker.is_alive.return_value = True

    # When
    embedding_repository_impl.stop_worker()

    # Then
    mock_worker.stop.assert_called_once()
    mock_timer.cancel.assert_called_once()
    mock_logger.info.assert_called_with("Embedding worker stopped manually")


def test_stop_worker_not_alive(
    embedding_repository_impl: EmbeddingModelRepositoryImpl,
    mock_worker: Mock,
    mock_timer: Mock,
    mock_logger: Mock,
) -> None:
    # Given
    mock_worker.is_alive.return_value = False

    # When
    embedding_repository_impl.stop_worker()

    # Then
    mock_worker.stop.assert_not_called()
    mock_timer.cancel.assert_not_called()
    mock_logger.info.assert_not_called()


def test_check_idle_timeout_stops_worker(
    embedding_repository_impl: EmbeddingModelRepositoryImpl,
    mock_worker: Mock,
    mock_timer: Mock,
    mock_logger: Mock,
) -> None:
    # Given
    mock_worker.is_alive.return_value = True
    mock_worker.is_processing.return_value = False

    # When
    embedding_repository_impl._check_idle_timeout()

    # Then
    mock_worker.stop.assert_called_once()
    mock_timer.cancel.assert_called_once()
    mock_logger.debug.assert_called_with("Checking embedding model idle timeout")
    mock_logger.info.assert_called_with("Embedding model stopped due to idle timeout")


def test_check_idle_timeout_worker_not_alive(
    embedding_repository_impl: EmbeddingModelRepositoryImpl,
    mock_worker: Mock,
    mock_timer: Mock,
    mock_logger: Mock,
) -> None:
    # Given
    mock_worker.is_alive.return_value = False

    # When
    embedding_repository_impl._check_idle_timeout()

    # Then
    mock_worker.stop.assert_not_called()
    mock_timer.cancel.assert_not_called()
    mock_logger.debug.assert_called_with("Checking embedding model idle timeout")
    mock_logger.info.assert_not_called()


def test_check_idle_timeout_worker_processing(
    embedding_repository_impl: EmbeddingModelRepositoryImpl,
    mock_worker: Mock,
    mock_timer: Mock,
    mock_logger: Mock,
) -> None:
    # Given
    mock_worker.is_alive.return_value = True
    mock_worker.is_processing.return_value = True

    # When
    embedding_repository_impl._check_idle_timeout()

    # Then
    mock_worker.stop.assert_not_called()
    mock_timer.cancel.assert_not_called()
    mock_logger.debug.assert_called_with("Checking embedding model idle timeout")
    mock_logger.info.assert_not_called()


def test_is_model_loaded_true(
    embedding_repository_impl: EmbeddingModelRepositoryImpl,
    mock_worker: Mock,
) -> None:
    # Given
    mock_worker.is_alive.return_value = True

    # When
    result = embedding_repository_impl.is_model_loaded()

    # Then
    assert result is True


def test_is_model_loaded_false(
    embedding_repository_impl: EmbeddingModelRepositoryImpl,
    mock_worker: Mock,
) -> None:
    # Given
    mock_worker.is_alive.return_value = False

    # When
    result = embedding_repository_impl.is_model_loaded()

    # Then
    assert result is False


def test_singleton_pattern(
    mock_config: Mock,
    mock_timer_factory: Mock,
    mock_logger: Mock,
    mock_worker_factory: Mock,
) -> None:
    # Given
    with patch.object(EmbeddingModelRepositoryImpl, "_instance", None):
        # When
        instance1 = EmbeddingModelRepositoryImpl(
            config=mock_config,
            timer_factory=mock_timer_factory,
            logger=mock_logger,
            worker_factory=mock_worker_factory,
        )
        instance2 = EmbeddingModelRepositoryImpl(
            config=mock_config,
            timer_factory=mock_timer_factory,
            logger=mock_logger,
            worker_factory=mock_worker_factory,
        )

        # Then
        assert instance1 is instance2


def test_thread_safety_singleton(
    mock_config: Mock,
    mock_timer_factory: Mock,
    mock_logger: Mock,
    mock_worker_factory: Mock,
) -> None:
    import threading

    instances = []

    def create_instance() -> EmbeddingModelRepositoryImpl:
        instance = EmbeddingModelRepositoryImpl(
            config=mock_config,
            timer_factory=mock_timer_factory,
            logger=mock_logger,
            worker_factory=mock_worker_factory,
        )
        instances.append(instance)

    with patch.object(EmbeddingModelRepositoryImpl, "_instance", None):
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(instances) == 5
        assert all(instance is instances[0] for instance in instances)


def test_initialization_called_once(
    mock_config: Mock,
    mock_timer_factory: Mock,
    mock_logger: Mock,
    mock_worker_factory: Mock,
) -> None:
    # Given
    with (
        patch.object(EmbeddingModelRepositoryImpl, "_instance", None),
        patch.object(EmbeddingModelRepositoryImpl, "_initialize") as mock_initialize,
    ):
        # When
        instance1 = EmbeddingModelRepositoryImpl(
            config=mock_config,
            timer_factory=mock_timer_factory,
            logger=mock_logger,
            worker_factory=mock_worker_factory,
        )
        instance2 = EmbeddingModelRepositoryImpl(
            config=mock_config,
            timer_factory=mock_timer_factory,
            logger=mock_logger,
            worker_factory=mock_worker_factory,
        )

        # Then
        assert mock_initialize.call_count == 1
        assert instance1 is instance2
