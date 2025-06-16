from typing import List

from api.dtos.create_embeddings_result_dto import (
    CreateEmbeddingsResultDto,
    SparseEmbedding,
    TextEmbedding,
)


def test_sparse_embedding_creation() -> None:
    # Given
    indices = [1, 5, 10]
    values = [0.5, 0.3, 0.8]

    # When
    sparse_embedding = SparseEmbedding(indices=indices, values=values)

    # Then
    assert sparse_embedding.indices == indices
    assert sparse_embedding.values == values


def test_sparse_embedding_empty() -> None:
    # Given
    indices: List[int] = []
    values: List[float] = []

    # When
    sparse_embedding = SparseEmbedding(indices=indices, values=values)

    # Then
    assert sparse_embedding.indices == []
    assert sparse_embedding.values == []


def test_text_embedding_creation_with_all_embeddings() -> None:
    # Given
    text = "Hello world"
    dense = [0.1, 0.2, 0.3]
    sparse = SparseEmbedding(indices=[1, 2], values=[0.5, 0.7])
    colbert = [[0.1, 0.2], [0.3, 0.4]]

    # When
    text_embedding = TextEmbedding(text=text, dense=dense, sparse=sparse, colbert=colbert)

    # Then
    assert text_embedding.text == text
    assert text_embedding.dense == dense
    assert text_embedding.sparse == sparse
    assert text_embedding.colbert == colbert


def test_text_embedding_creation_with_defaults() -> None:
    # Given
    text = "Hello world"

    # When
    text_embedding = TextEmbedding(text=text)

    # Then
    assert text_embedding.text == text
    assert text_embedding.dense is None
    assert text_embedding.sparse is None
    assert text_embedding.colbert is None


def test_text_embedding_creation_with_dense_only() -> None:
    # Given
    text = "Hello world"
    dense = [0.1, 0.2, 0.3, 0.4, 0.5]

    # When
    text_embedding = TextEmbedding(text=text, dense=dense)

    # Then
    assert text_embedding.text == text
    assert text_embedding.dense == dense
    assert text_embedding.sparse is None
    assert text_embedding.colbert is None


def test_text_embedding_creation_with_sparse_only() -> None:
    # Given
    text = "Hello world"
    sparse = SparseEmbedding(indices=[0, 5, 10], values=[0.2, 0.8, 0.5])

    # When
    text_embedding = TextEmbedding(text=text, sparse=sparse)

    # Then
    assert text_embedding.text == text
    assert text_embedding.dense is None
    assert text_embedding.sparse == sparse
    assert text_embedding.colbert is None


def test_text_embedding_creation_with_colbert_only() -> None:
    # Given
    text = "Hello world"
    colbert = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

    # When
    text_embedding = TextEmbedding(text=text, colbert=colbert)

    # Then
    assert text_embedding.text == text
    assert text_embedding.dense is None
    assert text_embedding.sparse is None
    assert text_embedding.colbert == colbert


def test_create_embeddings_result_dto_creation() -> None:
    # Given
    embeddings = [
        TextEmbedding(
            text="Hello world",
            dense=[0.1, 0.2, 0.3],
            sparse=SparseEmbedding(indices=[1, 2], values=[0.5, 0.7]),
            colbert=[[0.1, 0.2], [0.3, 0.4]],
        ),
        TextEmbedding(text="Test text", dense=[0.4, 0.5, 0.6]),
    ]

    # When
    result_dto = CreateEmbeddingsResultDto(embeddings=embeddings)

    # Then
    assert len(result_dto.embeddings) == 2
    assert result_dto.embeddings[0].text == "Hello world"
    assert result_dto.embeddings[1].text == "Test text"
    assert result_dto.embeddings[0].dense == [0.1, 0.2, 0.3]
    assert result_dto.embeddings[1].dense == [0.4, 0.5, 0.6]


def test_create_embeddings_result_dto_empty() -> None:
    # Given
    embeddings: List[TextEmbedding] = []

    # When
    result_dto = CreateEmbeddingsResultDto(embeddings=embeddings)

    # Then
    assert len(result_dto.embeddings) == 0
    assert result_dto.embeddings == []


def test_create_embeddings_result_dto_serialization() -> None:
    # Given
    embeddings = [
        TextEmbedding(
            text="Hello world",
            dense=[0.1, 0.2],
            sparse=SparseEmbedding(indices=[1], values=[0.5]),
        ),
    ]
    result_dto = CreateEmbeddingsResultDto(embeddings=embeddings)

    # When
    serialized = result_dto.model_dump()

    # Then
    expected = {
        "embeddings": [
            {
                "text": "Hello world",
                "dense": [0.1, 0.2],
                "sparse": {"indices": [1], "values": [0.5]},
                "colbert": None,
            },
        ],
    }
    assert serialized == expected


def test_create_embeddings_result_dto_deserialization() -> None:
    # Given
    data = {
        "embeddings": [
            {
                "text": "Hello world",
                "dense": [0.1, 0.2],
                "sparse": {"indices": [1], "values": [0.5]},
                "colbert": None,
            },
        ],
    }

    # When
    result_dto = CreateEmbeddingsResultDto.model_validate(data)

    # Then
    assert len(result_dto.embeddings) == 1
    assert result_dto.embeddings[0].text == "Hello world"
    assert result_dto.embeddings[0].dense == [0.1, 0.2]
    assert result_dto.embeddings[0].sparse.indices == [1]
    assert result_dto.embeddings[0].sparse.values == [0.5]
    assert result_dto.embeddings[0].colbert is None


def test_sparse_embedding_serialization() -> None:
    # Given
    sparse = SparseEmbedding(indices=[1, 2, 3], values=[0.1, 0.2, 0.3])

    # When
    serialized = sparse.model_dump()

    # Then
    expected = {"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]}
    assert serialized == expected


def test_text_embedding_serialization_with_all_fields() -> None:
    # Given
    text_embedding = TextEmbedding(
        text="Test",
        dense=[0.1, 0.2],
        sparse=SparseEmbedding(indices=[1], values=[0.5]),
        colbert=[[0.1, 0.2]],
    )

    # When
    serialized = text_embedding.model_dump()

    # Then
    expected = {
        "text": "Test",
        "dense": [0.1, 0.2],
        "sparse": {"indices": [1], "values": [0.5]},
        "colbert": [[0.1, 0.2]],
    }
    assert serialized == expected
