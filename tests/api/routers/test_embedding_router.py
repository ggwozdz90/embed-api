from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dtos.create_embeddings_dto import CreateEmbeddingsDto
from api.dtos.create_embeddings_result_dto import (
    CreateEmbeddingsResultDto,
    TextEmbedding,
)
from api.routers.embedding_router import EmbeddingRouter
from application.usecases.create_embeddings_usecase import CreateEmbeddingsUseCase
from data.workers.bge_m3_embedding_worker import EmbeddingResult


@pytest.fixture
def mock_create_embeddings_usecase() -> CreateEmbeddingsUseCase:
    return Mock(CreateEmbeddingsUseCase)


@pytest.fixture
def embedding_router() -> EmbeddingRouter:
    return EmbeddingRouter()


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


@pytest.fixture
def client(
    mock_create_embeddings_usecase: CreateEmbeddingsUseCase,
) -> TestClient:
    router = EmbeddingRouter()
    app = FastAPI()
    app.include_router(router.router)
    app.dependency_overrides[CreateEmbeddingsUseCase] = lambda: mock_create_embeddings_usecase
    return TestClient(app)


def test_embedding_router_initialization() -> None:
    # When
    router = EmbeddingRouter()

    # Then
    assert router.router is not None
    assert len(router.router.routes) == 1
    assert router.router.routes[0].path == "/embeddings"
    assert "POST" in router.router.routes[0].methods


@pytest.mark.asyncio
async def test_create_embeddings_success(
    mock_create_embeddings_usecase: CreateEmbeddingsUseCase,
    embedding_router: EmbeddingRouter,
    mock_embedding_result: EmbeddingResult,
) -> None:
    # Given
    mock_create_embeddings_usecase.execute = AsyncMock(return_value=mock_embedding_result)

    create_embeddings_dto = CreateEmbeddingsDto(
        texts=["Hello world", "Test text"],
        include_dense=True,
        include_sparse=False,
        include_colbert=False,
    )

    # When
    result = await embedding_router.create_embeddings(mock_create_embeddings_usecase, create_embeddings_dto)

    # Then
    assert isinstance(result, CreateEmbeddingsResultDto)
    assert len(result.embeddings) == 2
    assert result.embeddings[0].text == "Hello world"
    assert result.embeddings[1].text == "Test text"
    assert result.embeddings[0].dense == [0.1, 0.2, 0.3]
    assert result.embeddings[1].dense == [0.4, 0.5, 0.6]

    mock_create_embeddings_usecase.execute.assert_called_once_with(["Hello world", "Test text"], True, False, False)


@pytest.mark.asyncio
async def test_create_embeddings_with_all_embedding_types(
    mock_create_embeddings_usecase: CreateEmbeddingsUseCase,
    embedding_router: EmbeddingRouter,
) -> None:
    # Given
    embeddings = [
        TextEmbedding(
            text="Hello world",
            dense=[0.1, 0.2, 0.3],
            sparse=None,
            colbert=[[0.1, 0.2], [0.3, 0.4]],
        ),
    ]
    mock_result = EmbeddingResult(embeddings=embeddings)
    mock_create_embeddings_usecase.execute = AsyncMock(return_value=mock_result)

    create_embeddings_dto = CreateEmbeddingsDto(
        texts=["Hello world"],
        include_dense=True,
        include_sparse=True,
        include_colbert=True,
    )

    # When
    result = await embedding_router.create_embeddings(mock_create_embeddings_usecase, create_embeddings_dto)

    # Then
    assert isinstance(result, CreateEmbeddingsResultDto)
    assert len(result.embeddings) == 1
    assert result.embeddings[0].text == "Hello world"
    assert result.embeddings[0].dense == [0.1, 0.2, 0.3]
    assert result.embeddings[0].colbert == [[0.1, 0.2], [0.3, 0.4]]

    mock_create_embeddings_usecase.execute.assert_called_once_with(["Hello world"], True, True, True)


@pytest.mark.asyncio
async def test_create_embeddings_empty_texts(
    mock_create_embeddings_usecase: CreateEmbeddingsUseCase,
    embedding_router: EmbeddingRouter,
) -> None:
    # Given
    mock_result = EmbeddingResult(embeddings=[])
    mock_create_embeddings_usecase.execute = AsyncMock(return_value=mock_result)

    create_embeddings_dto = CreateEmbeddingsDto(
        texts=[],
        include_dense=True,
        include_sparse=False,
        include_colbert=False,
    )

    # When
    result = await embedding_router.create_embeddings(mock_create_embeddings_usecase, create_embeddings_dto)

    # Then
    assert isinstance(result, CreateEmbeddingsResultDto)
    assert len(result.embeddings) == 0

    mock_create_embeddings_usecase.execute.assert_called_once_with([], True, False, False)


@pytest.mark.asyncio
async def test_create_embeddings_single_text(
    mock_create_embeddings_usecase: CreateEmbeddingsUseCase,
    embedding_router: EmbeddingRouter,
) -> None:
    # Given
    embeddings = [
        TextEmbedding(
            text="Single text",
            dense=[0.7, 0.8, 0.9],
            sparse=None,
            colbert=None,
        ),
    ]
    mock_result = EmbeddingResult(embeddings=embeddings)
    mock_create_embeddings_usecase.execute = AsyncMock(return_value=mock_result)

    create_embeddings_dto = CreateEmbeddingsDto(
        texts=["Single text"],
        include_dense=True,
        include_sparse=False,
        include_colbert=False,
    )

    # When
    result = await embedding_router.create_embeddings(mock_create_embeddings_usecase, create_embeddings_dto)

    # Then
    assert isinstance(result, CreateEmbeddingsResultDto)
    assert len(result.embeddings) == 1
    assert result.embeddings[0].text == "Single text"
    assert result.embeddings[0].dense == [0.7, 0.8, 0.9]

    mock_create_embeddings_usecase.execute.assert_called_once_with(["Single text"], True, False, False)


@pytest.mark.asyncio
async def test_create_embeddings_usecase_exception(
    mock_create_embeddings_usecase: CreateEmbeddingsUseCase,
    embedding_router: EmbeddingRouter,
) -> None:
    # Given
    mock_create_embeddings_usecase.execute = AsyncMock(side_effect=Exception("Embedding creation failed"))

    create_embeddings_dto = CreateEmbeddingsDto(
        texts=["Hello world"],
        include_dense=True,
        include_sparse=False,
        include_colbert=False,
    )

    # When & Then
    with pytest.raises(Exception, match="Embedding creation failed"):
        await embedding_router.create_embeddings(mock_create_embeddings_usecase, create_embeddings_dto)

    mock_create_embeddings_usecase.execute.assert_called_once_with(["Hello world"], True, False, False)


@pytest.mark.asyncio
async def test_create_embeddings_with_defaults(
    mock_create_embeddings_usecase: CreateEmbeddingsUseCase,
    embedding_router: EmbeddingRouter,
    mock_embedding_result: EmbeddingResult,
) -> None:
    # Given
    mock_create_embeddings_usecase.execute = AsyncMock(return_value=mock_embedding_result)

    create_embeddings_dto = CreateEmbeddingsDto(texts=["Hello world"])

    # When
    result = await embedding_router.create_embeddings(mock_create_embeddings_usecase, create_embeddings_dto)

    # Then
    assert isinstance(result, CreateEmbeddingsResultDto)
    assert len(result.embeddings) == 2

    mock_create_embeddings_usecase.execute.assert_called_once_with(["Hello world"], True, False, False)


def test_create_embeddings_http_success_with_defaults(
    client: TestClient,
    mock_create_embeddings_usecase: CreateEmbeddingsUseCase,
    mock_embedding_result: EmbeddingResult,
) -> None:
    # Given
    mock_create_embeddings_usecase.execute = AsyncMock(return_value=mock_embedding_result)

    request_data = {"texts": ["Hello world", "Test text"]}

    # When
    response = client.post("/embeddings", json=request_data)

    # Then
    assert response.status_code == 200
    response_data = response.json()
    assert "embeddings" in response_data
    assert len(response_data["embeddings"]) == 2

    mock_create_embeddings_usecase.execute.assert_awaited_once_with(
        ["Hello world", "Test text"],
        True,
        False,
        False,
    )


def test_create_embeddings_http_success_with_all_parameters(
    client: TestClient,
    mock_create_embeddings_usecase: CreateEmbeddingsUseCase,
    mock_embedding_result: EmbeddingResult,
) -> None:
    # Given
    mock_create_embeddings_usecase.execute = AsyncMock(return_value=mock_embedding_result)

    request_data = {
        "texts": ["Hello world", "Test text"],
        "include_dense": True,
        "include_sparse": True,
        "include_colbert": False,
    }

    # When
    response = client.post("/embeddings", json=request_data)

    # Then
    assert response.status_code == 200
    response_data = response.json()
    assert "embeddings" in response_data
    assert len(response_data["embeddings"]) == 2

    mock_create_embeddings_usecase.execute.assert_awaited_once_with(
        ["Hello world", "Test text"],
        True,
        True,
        False,
    )


def test_create_embeddings_http_missing_texts(client: TestClient) -> None:
    # Given
    request_data = {
        "include_dense": True,
        "include_sparse": False,
        "include_colbert": False,
    }

    # When
    response = client.post("/embeddings", json=request_data)

    # Then
    assert response.status_code == 422


def test_create_embeddings_http_empty_texts(
    client: TestClient,
    mock_create_embeddings_usecase: CreateEmbeddingsUseCase,
) -> None:
    # Given
    empty_result = EmbeddingResult(embeddings=[])
    mock_create_embeddings_usecase.execute = AsyncMock(return_value=empty_result)

    request_data: Dict[str, Any] = {"texts": []}

    # When
    response = client.post("/embeddings", json=request_data)

    # Then
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["embeddings"] == []


def test_create_embeddings_http_invalid_json(client: TestClient) -> None:
    # When
    response = client.post("/embeddings", content="invalid json")

    # Then
    assert response.status_code == 422


def test_create_embeddings_http_usecase_exception(
    client: TestClient,
    mock_create_embeddings_usecase: CreateEmbeddingsUseCase,
) -> None:
    # Given
    mock_create_embeddings_usecase.execute = AsyncMock(side_effect=Exception("Embedding creation failed"))

    request_data = {"texts": ["Hello world"]}

    # When & Then
    try:
        response = client.post("/embeddings", json=request_data)
        assert response.status_code == 500
    except Exception as e:
        assert str(e) == "Embedding creation failed"


def test_create_embeddings_http_missing_body(client: TestClient) -> None:
    # When
    response = client.post("/embeddings")

    # Then
    assert response.status_code == 422
