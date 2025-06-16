import multiprocessing
from unittest.mock import MagicMock, Mock, patch

import pytest
from FlagEmbedding import BGEM3FlagModel

from core.logger.logger import Logger
from data.workers.bge_m3_embedding_worker import (
    BgeM3EmbeddingConfig,
    BgeM3EmbeddingWorker,
    EmbeddingRequest,
    EmbeddingResult,
)
from domain.exceptions.worker_not_running_error import WorkerNotRunningError


@pytest.fixture
def mock_config() -> BgeM3EmbeddingConfig:
    return BgeM3EmbeddingConfig(
        device="cpu",
        model_name="BAAI/bge-m3",
        log_level="INFO",
    )


@pytest.fixture
def mock_logger() -> Logger:
    return Mock(Logger)


@pytest.fixture
def mock_worker(mock_config: BgeM3EmbeddingConfig, mock_logger: Logger) -> BgeM3EmbeddingWorker:
    worker = BgeM3EmbeddingWorker(config=mock_config, logger=mock_logger)
    worker._pipe_parent = Mock()
    return worker


@pytest.fixture
def sample_embedding_request() -> EmbeddingRequest:
    return EmbeddingRequest(
        texts=["Hello world", "This is a test"],
        include_dense=True,
        include_sparse=True,
        include_colbert=False,
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
            {
                "text": "This is a test",
                "dense": [0.4, 0.5, 0.6],
                "sparse": {"indices": [4, 5, 6], "values": [0.8, 0.9, 1.0]},
            },
        ],
    )


def test_create_embeddings_success(
    mock_worker: BgeM3EmbeddingWorker,
    sample_embedding_request: EmbeddingRequest,
    sample_embedding_result: EmbeddingResult,
) -> None:
    # Given
    mock_worker.is_alive = Mock(return_value=True)
    mock_worker._pipe_parent.recv = Mock(return_value=sample_embedding_result)

    # When
    result = mock_worker.create_embeddings(sample_embedding_request)

    # Then
    assert result == sample_embedding_result
    mock_worker._pipe_parent.send.assert_called_once_with(("create_embeddings", sample_embedding_request))


def test_create_embeddings_worker_not_running(
    mock_worker: BgeM3EmbeddingWorker,
    sample_embedding_request: EmbeddingRequest,
) -> None:
    # Given
    mock_worker.is_alive = Mock(return_value=False)

    # When / Then
    with pytest.raises(WorkerNotRunningError):
        mock_worker.create_embeddings(sample_embedding_request)


def test_create_embeddings_exception(
    mock_worker: BgeM3EmbeddingWorker,
    sample_embedding_request: EmbeddingRequest,
) -> None:
    # Given
    mock_worker.is_alive = Mock(return_value=True)
    test_exception = Exception("Embedding error")
    mock_worker._pipe_parent.recv = Mock(return_value=test_exception)

    # When / Then
    with pytest.raises(Exception, match="Embedding error"):
        mock_worker.create_embeddings(sample_embedding_request)


def test_initialize_shared_object(mock_config: BgeM3EmbeddingConfig, mock_logger: Logger) -> None:
    # Given
    with patch.object(BGEM3FlagModel, "__init__", return_value=None) as mock_model_init:
        worker = BgeM3EmbeddingWorker(config=mock_config, logger=mock_logger)

        # When
        model = worker.initialize_shared_object(mock_config)

        # Then
        mock_model_init.assert_called_once_with(
            mock_config.model_name,
            device=mock_config.device,
        )
        assert model is not None


def test_handle_command_create_embeddings_dense_and_sparse(
    mock_worker: BgeM3EmbeddingWorker,
    mock_config: BgeM3EmbeddingConfig,
) -> None:
    # Given
    mock_pipe = Mock()
    is_processing = multiprocessing.Value("b", False)
    processing_lock = multiprocessing.Lock()
    mock_model = MagicMock()
    mock_embeddings = {
        "dense_vecs": [
            MagicMock(tolist=lambda: [0.1, 0.2, 0.3]),
            MagicMock(tolist=lambda: [0.4, 0.5, 0.6]),
        ],
        "lexical_weights": [
            {1: 0.5, 2: 0.6, 3: 0.7},
            {4: 0.8, 5: 0.9, 6: 1.0},
        ],
    }
    mock_model.encode.return_value = mock_embeddings

    request = EmbeddingRequest(
        texts=["Hello world", "This is a test"],
        include_dense=True,
        include_sparse=True,
        include_colbert=False,
    )

    # When
    mock_worker.handle_command(
        "create_embeddings",
        request,
        mock_model,
        mock_config,
        mock_pipe,
        is_processing,
        processing_lock,
    )

    # Then
    mock_model.encode.assert_called_once_with(
        request.texts,
        batch_size=12,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    assert mock_pipe.send.call_count == 1
    sent_result = mock_pipe.send.call_args[0][0]
    assert isinstance(sent_result, EmbeddingResult)
    assert len(sent_result.embeddings) == 2

    first_embedding = sent_result.embeddings[0]
    assert first_embedding["text"] == "Hello world"
    assert first_embedding["dense"] == [0.1, 0.2, 0.3]
    assert first_embedding["sparse"] == {"indices": [1, 2, 3], "values": [0.5, 0.6, 0.7]}

    second_embedding = sent_result.embeddings[1]
    assert second_embedding["text"] == "This is a test"
    assert second_embedding["dense"] == [0.4, 0.5, 0.6]
    assert second_embedding["sparse"] == {"indices": [4, 5, 6], "values": [0.8, 0.9, 1.0]}


def test_handle_command_create_embeddings_with_colbert(
    mock_worker: BgeM3EmbeddingWorker,
    mock_config: BgeM3EmbeddingConfig,
) -> None:
    # Given
    mock_pipe = Mock()
    is_processing = multiprocessing.Value("b", False)
    processing_lock = multiprocessing.Lock()
    mock_model = MagicMock()

    mock_embeddings = {
        "dense_vecs": [
            MagicMock(tolist=lambda: [0.1, 0.2, 0.3]),
        ],
        "colbert_vecs": [
            MagicMock(tolist=lambda: [[0.1, 0.2], [0.3, 0.4]]),
        ],
    }
    mock_model.encode.return_value = mock_embeddings

    request = EmbeddingRequest(
        texts=["Hello world"],
        include_dense=True,
        include_sparse=False,
        include_colbert=True,
    )

    # When
    mock_worker.handle_command(
        "create_embeddings",
        request,
        mock_model,
        mock_config,
        mock_pipe,
        is_processing,
        processing_lock,
    )

    # Then
    mock_model.encode.assert_called_once_with(
        request.texts,
        batch_size=12,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=True,
    )

    assert mock_pipe.send.call_count == 1
    sent_result = mock_pipe.send.call_args[0][0]
    assert isinstance(sent_result, EmbeddingResult)
    assert len(sent_result.embeddings) == 1

    embedding = sent_result.embeddings[0]
    assert embedding["text"] == "Hello world"
    assert embedding["dense"] == [0.1, 0.2, 0.3]
    assert embedding["colbert"] == [[0.1, 0.2], [0.3, 0.4]]
    assert "sparse" not in embedding


def test_handle_command_create_embeddings_only_dense(
    mock_worker: BgeM3EmbeddingWorker,
    mock_config: BgeM3EmbeddingConfig,
) -> None:
    # Given
    mock_pipe = Mock()
    is_processing = multiprocessing.Value("b", False)
    processing_lock = multiprocessing.Lock()
    mock_model = MagicMock()
    mock_embeddings = {
        "dense_vecs": [
            MagicMock(tolist=lambda: [0.1, 0.2, 0.3]),
        ],
    }
    mock_model.encode.return_value = mock_embeddings

    request = EmbeddingRequest(
        texts=["Hello world"],
        include_dense=True,
        include_sparse=False,
        include_colbert=False,
    )

    # When
    mock_worker.handle_command(
        "create_embeddings",
        request,
        mock_model,
        mock_config,
        mock_pipe,
        is_processing,
        processing_lock,
    )

    # Then
    assert mock_pipe.send.call_count == 1
    sent_result = mock_pipe.send.call_args[0][0]
    assert isinstance(sent_result, EmbeddingResult)
    assert len(sent_result.embeddings) == 1

    embedding = sent_result.embeddings[0]
    assert embedding["text"] == "Hello world"
    assert embedding["dense"] == [0.1, 0.2, 0.3]
    assert "sparse" not in embedding
    assert "colbert" not in embedding


def test_handle_command_create_embeddings_exception(
    mock_worker: BgeM3EmbeddingWorker,
    mock_config: BgeM3EmbeddingConfig,
) -> None:
    # Given
    mock_pipe = Mock()
    is_processing = multiprocessing.Value("b", False)
    processing_lock = multiprocessing.Lock()
    mock_model = MagicMock()
    mock_model.encode.side_effect = RuntimeError("Encoding error")

    request = EmbeddingRequest(
        texts=["Hello world"],
        include_dense=True,
        include_sparse=False,
        include_colbert=False,
    )

    # When
    mock_worker.handle_command(
        "create_embeddings",
        request,
        mock_model,
        mock_config,
        mock_pipe,
        is_processing,
        processing_lock,
    )

    # Then
    assert mock_pipe.send.call_count == 1
    sent_exception = mock_pipe.send.call_args[0][0]
    assert isinstance(sent_exception, RuntimeError)
    assert str(sent_exception) == "Encoding error"


def test_handle_command_processing_lock(
    mock_worker: BgeM3EmbeddingWorker,
    mock_config: BgeM3EmbeddingConfig,
) -> None:
    # Given
    mock_pipe = Mock()
    is_processing = multiprocessing.Value("b", False)
    processing_lock = multiprocessing.Lock()
    mock_model = MagicMock()

    mock_embeddings = {
        "dense_vecs": [MagicMock(tolist=lambda: [0.1, 0.2, 0.3])],
    }
    mock_model.encode.return_value = mock_embeddings

    request = EmbeddingRequest(
        texts=["Hello world"],
        include_dense=True,
        include_sparse=False,
        include_colbert=False,
    )

    # When
    mock_worker.handle_command(
        "create_embeddings",
        request,
        mock_model,
        mock_config,
        mock_pipe,
        is_processing,
        processing_lock,
    )

    # Then
    assert not is_processing.value


def test_get_worker_name(mock_worker: BgeM3EmbeddingWorker) -> None:
    # When
    name = mock_worker.get_worker_name()

    # Then
    assert name == "BgeM3EmbeddingWorker"


def test_embedding_request_dataclass() -> None:
    # Given / When
    request = EmbeddingRequest(
        texts=["test text"],
        include_dense=True,
        include_sparse=False,
        include_colbert=True,
    )

    # Then
    assert request.texts == ["test text"]
    assert request.include_dense is True
    assert request.include_sparse is False
    assert request.include_colbert is True


def test_embedding_result_dataclass() -> None:
    # Given / When
    result = EmbeddingResult(
        embeddings=[
            {"text": "test", "dense": [0.1, 0.2]},
        ],
    )

    # Then
    assert len(result.embeddings) == 1
    assert result.embeddings[0]["text"] == "test"
    assert result.embeddings[0]["dense"] == [0.1, 0.2]


def test_bge_m3_embedding_config_dataclass() -> None:
    # Given / When
    config = BgeM3EmbeddingConfig(
        device="cuda",
        model_name="BAAI/bge-m3",
        log_level="DEBUG",
    )

    # Then
    assert config.device == "cuda"
    assert config.model_name == "BAAI/bge-m3"
    assert config.log_level == "DEBUG"
