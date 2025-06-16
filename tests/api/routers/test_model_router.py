from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.model_router import ModelRouter
from application.usecases.get_model_status_usecase import GetModelStatusUseCase
from application.usecases.load_model_usecase import LoadModelUseCase
from application.usecases.unload_model_usecase import UnloadModelUseCase


@pytest.fixture
def mock_load_model_usecase() -> LoadModelUseCase:
    return Mock(LoadModelUseCase)


@pytest.fixture
def mock_unload_model_usecase() -> UnloadModelUseCase:
    return Mock(UnloadModelUseCase)


@pytest.fixture
def mock_get_model_status_usecase() -> GetModelStatusUseCase:
    return Mock(GetModelStatusUseCase)


@pytest.fixture
def client(
    mock_load_model_usecase: LoadModelUseCase,
    mock_unload_model_usecase: UnloadModelUseCase,
    mock_get_model_status_usecase: GetModelStatusUseCase,
) -> TestClient:
    router = ModelRouter()
    app = FastAPI()
    app.include_router(router.router)
    app.dependency_overrides[LoadModelUseCase] = lambda: mock_load_model_usecase
    app.dependency_overrides[UnloadModelUseCase] = lambda: mock_unload_model_usecase
    app.dependency_overrides[GetModelStatusUseCase] = lambda: mock_get_model_status_usecase
    return TestClient(app)


def test_load_model_success(
    client: TestClient,
    mock_load_model_usecase: LoadModelUseCase,
) -> None:
    # Given
    mock_load_model_usecase.execute = AsyncMock(return_value="Model loaded successfully")

    # When
    response = client.post("/model/load")

    # Then
    assert response.status_code == 200
    assert response.json() == {"message": "Model loaded successfully"}
    mock_load_model_usecase.execute.assert_awaited_once()


def test_load_model_usecase_exception(
    client: TestClient,
    mock_load_model_usecase: LoadModelUseCase,
) -> None:
    # Given
    mock_load_model_usecase.execute = AsyncMock(side_effect=Exception("Load failed"))

    # When & Then
    # Exception will bubble up without global exception handler
    try:
        response = client.post("/model/load")
        # If we reach here, it means an exception handler caught it
        assert response.status_code == 500
    except Exception as e:
        # This is expected behavior without global exception handler
        assert str(e) == "Load failed"

    mock_load_model_usecase.execute.assert_awaited_once()


def test_unload_model_success(
    client: TestClient,
    mock_unload_model_usecase: UnloadModelUseCase,
) -> None:
    # Given
    mock_unload_model_usecase.execute = AsyncMock(return_value="Model unloaded successfully")

    # When
    response = client.post("/model/unload")

    # Then
    assert response.status_code == 200
    assert response.json() == {"message": "Model unloaded successfully"}
    mock_unload_model_usecase.execute.assert_awaited_once()


def test_unload_model_usecase_exception(
    client: TestClient,
    mock_unload_model_usecase: UnloadModelUseCase,
) -> None:
    # Given
    mock_unload_model_usecase.execute = AsyncMock(side_effect=Exception("Unload failed"))

    # When & Then
    # Exception will bubble up without global exception handler
    try:
        response = client.post("/model/unload")
        # If we reach here, it means an exception handler caught it
        assert response.status_code == 500
    except Exception as e:
        # This is expected behavior without global exception handler
        assert str(e) == "Unload failed"

    mock_unload_model_usecase.execute.assert_awaited_once()


def test_get_model_status_success_loaded(
    client: TestClient,
    mock_get_model_status_usecase: GetModelStatusUseCase,
) -> None:
    # Given
    status_data = {"is_loaded": True, "model_name": "BAAI/bge-m3", "device": "cuda"}
    mock_get_model_status_usecase.execute = AsyncMock(return_value=status_data)

    # When
    response = client.get("/model/status")

    # Then
    assert response.status_code == 200
    assert response.json() == {"is_loaded": True, "model_name": "BAAI/bge-m3", "device": "cuda"}
    mock_get_model_status_usecase.execute.assert_awaited_once()


def test_get_model_status_success_not_loaded(
    client: TestClient,
    mock_get_model_status_usecase: GetModelStatusUseCase,
) -> None:
    # Given
    status_data = {"is_loaded": False, "model_name": "BAAI/bge-m3", "device": "cpu"}
    mock_get_model_status_usecase.execute = AsyncMock(return_value=status_data)

    # When
    response = client.get("/model/status")

    # Then
    assert response.status_code == 200
    assert response.json() == {"is_loaded": False, "model_name": "BAAI/bge-m3", "device": "cpu"}
    mock_get_model_status_usecase.execute.assert_awaited_once()


def test_get_model_status_usecase_exception(
    client: TestClient,
    mock_get_model_status_usecase: GetModelStatusUseCase,
) -> None:
    # Given
    mock_get_model_status_usecase.execute = AsyncMock(side_effect=Exception("Status check failed"))

    # When & Then
    # Exception will bubble up without global exception handler
    try:
        response = client.get("/model/status")
        # If we reach here, it means an exception handler caught it
        assert response.status_code == 500
    except Exception as e:
        # This is expected behavior without global exception handler
        assert str(e) == "Status check failed"

    mock_get_model_status_usecase.execute.assert_awaited_once()


def test_get_model_status_with_different_model(
    client: TestClient,
    mock_get_model_status_usecase: GetModelStatusUseCase,
) -> None:
    # Given
    status_data = {"is_loaded": True, "model_name": "custom/embedding-model", "device": "mps"}
    mock_get_model_status_usecase.execute = AsyncMock(return_value=status_data)

    # When
    response = client.get("/model/status")

    # Then
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["is_loaded"] is True
    assert response_data["model_name"] == "custom/embedding-model"
    assert response_data["device"] == "mps"


def test_load_model_multiple_calls(
    client: TestClient,
    mock_load_model_usecase: LoadModelUseCase,
) -> None:
    # Given
    mock_load_model_usecase.execute = AsyncMock(return_value="Model loaded successfully")

    # When
    response1 = client.post("/model/load")
    response2 = client.post("/model/load")

    # Then
    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json() == response2.json()
    assert mock_load_model_usecase.execute.await_count == 2


def test_unload_model_multiple_calls(
    client: TestClient,
    mock_unload_model_usecase: UnloadModelUseCase,
) -> None:
    # Given
    mock_unload_model_usecase.execute = AsyncMock(return_value="Model unloaded successfully")

    # When
    response1 = client.post("/model/unload")
    response2 = client.post("/model/unload")

    # Then
    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json() == response2.json()
    assert mock_unload_model_usecase.execute.await_count == 2


def test_model_router_endpoints_exist() -> None:
    # Given
    router = ModelRouter()

    # When
    routes = router.router.routes  # Then
    assert len(routes) == 3

    # Extract route paths and methods
    route_info = [(route.path, list(route.methods)) for route in routes]
    expected_routes = [
        ("/model/load", ["POST"]),
        ("/model/unload", ["POST"]),
        ("/model/status", ["GET"]),
    ]

    for expected_path, expected_methods in expected_routes:
        matching_routes = [(path, methods) for path, methods in route_info if path == expected_path]
        assert len(matching_routes) == 1
        path, methods = matching_routes[0]
        for method in expected_methods:
            assert method in methods


def test_load_model_wrong_method(client: TestClient) -> None:
    # When
    response = client.get("/model/load")  # GET instead of POST

    # Then
    assert response.status_code == 405  # Method Not Allowed


def test_unload_model_wrong_method(client: TestClient) -> None:
    # When
    response = client.get("/model/unload")  # GET instead of POST

    # Then
    assert response.status_code == 405  # Method Not Allowed


def test_get_model_status_wrong_method(client: TestClient) -> None:
    # When
    response = client.post("/model/status")  # POST instead of GET

    # Then
    assert response.status_code == 405  # Method Not Allowed


def test_load_model_with_body(
    client: TestClient,
    mock_load_model_usecase: LoadModelUseCase,
) -> None:
    # Given
    mock_load_model_usecase.execute = AsyncMock(return_value="Model loaded successfully")

    # When
    response = client.post("/model/load", json={"unnecessary": "data"})

    # Then
    # Should still work even with unnecessary body data
    assert response.status_code == 200
    assert response.json() == {"message": "Model loaded successfully"}
    mock_load_model_usecase.execute.assert_awaited_once()


def test_unload_model_with_body(
    client: TestClient,
    mock_unload_model_usecase: UnloadModelUseCase,
) -> None:
    # Given
    mock_unload_model_usecase.execute = AsyncMock(return_value="Model unloaded successfully")

    # When
    response = client.post("/model/unload", json={"unnecessary": "data"})

    # Then
    # Should still work even with unnecessary body data
    assert response.status_code == 200
    assert response.json() == {"message": "Model unloaded successfully"}
    mock_unload_model_usecase.execute.assert_awaited_once()
