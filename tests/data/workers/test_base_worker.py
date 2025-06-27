import multiprocessing
from multiprocessing.sharedctypes import Synchronized
from typing import Generator
from unittest.mock import Mock, patch

import pytest

from core.logger.logger import Logger
from data.workers.base_worker import BaseWorker


class MockBaseWorker(BaseWorker[str, str, dict, None]):  # type: ignore
    def initialize_shared_object(
        self,
        config: str,
    ) -> None:
        return None

    def handle_command(
        self,
        command: str,
        args: str,
        shared_object: None,
        config: str,
        pipe: multiprocessing.connection.Connection,
        is_processing: Synchronized,  # type: ignore
        processing_lock: multiprocessing.synchronize.Lock,
    ) -> None:
        pass

    def get_worker_name(self) -> str:
        return "MockBaseWorker"


@pytest.fixture
def base_worker(base_config: str, mock_logger: Logger) -> Generator[MockBaseWorker, None, None]:
    worker = MockBaseWorker(base_config, mock_logger)
    yield worker
    worker.stop()


@pytest.fixture
def mock_logger() -> Logger:
    return Mock(Logger)


@pytest.fixture
def base_config() -> str:
    return "cpu"


def test_start_creates_new_process(base_worker: MockBaseWorker) -> None:
    # Given
    with patch("multiprocessing.Process") as MockProcess:
        mock_process = Mock()
        MockProcess.return_value = mock_process

        # When
        base_worker.start()

        # Then
        MockProcess.assert_called_once()
        mock_process.start.assert_called_once()
        assert base_worker.is_alive()
        assert not base_worker.is_processing()


def test_stop_terminates_process(base_worker: MockBaseWorker) -> None:
    with patch("multiprocessing.Process") as MockProcess:
        mock_process = Mock()
        MockProcess.return_value = mock_process

        # Given
        base_worker.start()
        assert base_worker.is_alive()

        # When
        base_worker.stop()

        # Then
        mock_process.join.assert_called_once_with(timeout=5)
        mock_process.terminate.assert_called_once()
        assert not base_worker.is_alive()


def test_is_alive_returns_correct_status(base_worker: MockBaseWorker) -> None:
    with patch("multiprocessing.Process") as MockProcess:
        mock_process = Mock()
        MockProcess.return_value = mock_process

        # Given
        base_worker.start()

        # When
        alive_status = base_worker.is_alive()

        # Then
        assert alive_status

        # When
        base_worker.stop()
        alive_status = base_worker.is_alive()

        # Then
        assert not alive_status


def test_is_processing_returns_correct_status(base_worker: MockBaseWorker) -> None:
    # Given
    base_worker._is_processing.value = True

    # When
    processing_status = base_worker.is_processing()

    # Then
    assert processing_status

    # Given
    base_worker._is_processing.value = False

    # When
    processing_status = base_worker.is_processing()

    # Then
    assert not processing_status


def test_start_when_process_already_alive(base_worker: MockBaseWorker) -> None:
    # Given
    with patch("multiprocessing.Process") as MockProcess:
        mock_process = Mock()
        mock_process.is_alive.return_value = True
        MockProcess.return_value = mock_process
        base_worker._process = mock_process

        # When
        base_worker.start()

        # Then
        MockProcess.assert_not_called()


def test_stop_when_no_process(base_worker: MockBaseWorker) -> None:
    # Given
    base_worker._process = None

    # When
    base_worker.stop()

    # Then
    assert base_worker._process is None


def test_stop_when_process_not_alive(base_worker: MockBaseWorker) -> None:
    # Given
    with patch("multiprocessing.Process") as MockProcess:
        mock_process = Mock()
        mock_process.is_alive.return_value = False
        MockProcess.return_value = mock_process
        base_worker._process = mock_process

        # When
        base_worker.stop()

        # Then
        mock_process.join.assert_not_called()
        mock_process.terminate.assert_not_called()


def test_is_alive_when_no_process(base_worker: MockBaseWorker) -> None:
    # Given
    base_worker._process = None

    # When
    result = base_worker.is_alive()

    # Then
    assert not result


def test_run_process_handles_commands(base_worker: MockBaseWorker) -> None:
    # Given
    mock_pipe = Mock()
    mock_stop_event = Mock()
    mock_stop_event.is_set.side_effect = [False, True]
    mock_pipe.poll.return_value = True
    mock_pipe.recv.return_value = ("test_command", "test_args")

    config = Mock()
    config.log_level = "INFO"
    is_processing = multiprocessing.Value("b", False)
    processing_lock = multiprocessing.Lock()

    with (
        patch.object(base_worker, "initialize_shared_object", return_value=None) as mock_init,
        patch.object(base_worker, "handle_command") as mock_handle,
        patch.object(base_worker, "get_worker_name", return_value="TestWorker"),
        patch.object(base_worker._logger, "set_level"),
        patch.object(base_worker._logger, "info"),
        patch.object(base_worker._logger, "debug"),
    ):
        # When
        base_worker._run_process(config, mock_pipe, mock_stop_event, is_processing, processing_lock)

    # Then
    mock_init.assert_called_once_with(config)
    mock_handle.assert_called_once_with(
        "test_command",
        "test_args",
        None,
        config,
        mock_pipe,
        is_processing,
        processing_lock,
    )
    mock_pipe.close.assert_called_once()


def test_run_process_handles_no_commands(base_worker: MockBaseWorker) -> None:
    # Given
    mock_pipe = Mock()
    mock_stop_event = Mock()
    mock_stop_event.is_set.side_effect = [False, True]
    mock_pipe.poll.return_value = False

    config = Mock()
    config.log_level = "INFO"
    is_processing = multiprocessing.Value("b", False)
    processing_lock = multiprocessing.Lock()

    with (
        patch.object(base_worker, "initialize_shared_object", return_value=None) as mock_init,
        patch.object(base_worker, "handle_command") as mock_handle,
        patch.object(base_worker, "get_worker_name", return_value="TestWorker"),
        patch.object(base_worker._logger, "set_level"),
        patch.object(base_worker._logger, "info"),
        patch.object(base_worker._logger, "debug"),
    ):
        # When
        base_worker._run_process(config, mock_pipe, mock_stop_event, is_processing, processing_lock)

    # Then
    mock_init.assert_called_once_with(config)
    mock_handle.assert_not_called()
    mock_pipe.close.assert_called_once()
