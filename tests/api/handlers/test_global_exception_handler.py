from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.handlers.global_exception_handler import GlobalExceptionHandler
from core.logger.logger import Logger


@pytest.fixture
def mock_logger() -> Logger:
    return Mock(spec=Logger)


@pytest.fixture
def mock_app() -> FastAPI:
    return FastAPI()


@pytest.fixture
def exception_handler(mock_app: FastAPI, mock_logger: Logger) -> GlobalExceptionHandler:
    return GlobalExceptionHandler(mock_app, mock_logger)


@pytest.mark.asyncio
async def test_handle_value_error(exception_handler: GlobalExceptionHandler, mock_logger: Logger) -> None:
    # Given
    request = Mock(spec=Request)
    value_error = ValueError("Test value error")
    handlers = exception_handler.app.exception_handlers
    value_error_handler = None
    for exc_type, handler in handlers.items():
        if exc_type == ValueError:
            value_error_handler = handler
            break

    assert value_error_handler is not None

    # When
    response = await value_error_handler(request, value_error)

    # Then
    assert isinstance(response, JSONResponse)
    assert response.status_code == 422
    content = bytes(response.body).decode("utf-8")
    import json

    data = json.loads(content)
    assert data["status_code"] == 422
    assert data["message"] == "Value error"
    assert data["details"]["error_type"] == "ValueError"
    assert data["details"]["error_message"] == "Test value error"
    assert "trace" in data

    mock_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_handle_generic_exception(exception_handler: GlobalExceptionHandler, mock_logger: Logger) -> None:
    # Given
    request = Mock(spec=Request)
    generic_error = RuntimeError("Test runtime error")

    # Find the registered handler for Exception
    handlers = exception_handler.app.exception_handlers
    exception_handler_func = None
    for exc_type, handler in handlers.items():
        if exc_type == Exception:
            exception_handler_func = handler
            break

    assert exception_handler_func is not None

    # When
    response = await exception_handler_func(request, generic_error)

    # Then
    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    content = bytes(response.body).decode("utf-8")
    import json

    data = json.loads(content)
    assert data["status_code"] == 500
    assert data["message"] == "Internal server error"
    assert data["details"]["error_type"] == "RuntimeError"
    assert data["details"]["error_message"] == "Test runtime error"
    assert "trace" in data

    mock_logger.error.assert_called_once()


def test_register_handlers(mock_app: FastAPI, mock_logger: Logger) -> None:
    # Given
    initial_handlers_count = len(mock_app.exception_handlers)

    # When
    GlobalExceptionHandler(mock_app, mock_logger)

    # Then
    # Should register 2 handlers: ValueError and Exception
    assert len(mock_app.exception_handlers) == initial_handlers_count + 2
    assert ValueError in mock_app.exception_handlers
    assert Exception in mock_app.exception_handlers
