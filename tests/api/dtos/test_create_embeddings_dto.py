from api.dtos.create_embeddings_dto import CreateEmbeddingsDto


def test_create_embeddings_dto_creation_with_defaults() -> None:
    # Given
    texts = ["Hello world", "Test text"]

    # When
    dto = CreateEmbeddingsDto(texts=texts)

    # Then
    assert dto.texts == texts
    assert dto.include_dense is True
    assert dto.include_sparse is False
    assert dto.include_colbert is False


def test_create_embeddings_dto_creation_with_all_parameters() -> None:
    # Given
    texts = ["Hello world", "Test text"]
    include_dense = False
    include_sparse = True
    include_colbert = True

    # When
    dto = CreateEmbeddingsDto(
        texts=texts,
        include_dense=include_dense,
        include_sparse=include_sparse,
        include_colbert=include_colbert,
    )

    # Then
    assert dto.texts == texts
    assert dto.include_dense is False
    assert dto.include_sparse is True
    assert dto.include_colbert is True


def test_create_embeddings_dto_empty_texts() -> None:
    # Given
    texts: list[str] = []

    # When
    dto = CreateEmbeddingsDto(texts=texts)

    # Then
    assert dto.texts == []
    assert dto.include_dense is True
    assert dto.include_sparse is False
    assert dto.include_colbert is False


def test_create_embeddings_dto_single_text() -> None:
    # Given
    texts = ["Single text"]

    # When
    dto = CreateEmbeddingsDto(texts=texts)

    # Then
    assert dto.texts == ["Single text"]
    assert dto.include_dense is True
    assert dto.include_sparse is False
    assert dto.include_colbert is False


def test_create_embeddings_dto_multiple_texts() -> None:
    # Given
    texts = ["Text 1", "Text 2", "Text 3", "Text 4"]

    # When
    dto = CreateEmbeddingsDto(texts=texts)

    # Then
    assert dto.texts == texts
    assert len(dto.texts) == 4


def test_create_embeddings_dto_with_none_optional_fields() -> None:
    # Given
    texts = ["Hello world"]

    # When
    dto = CreateEmbeddingsDto(
        texts=texts,
        include_dense=None,
        include_sparse=None,
        include_colbert=None,
    )

    # Then
    assert dto.texts == texts
    assert dto.include_dense is None
    assert dto.include_sparse is None
    assert dto.include_colbert is None


def test_create_embeddings_dto_serialization() -> None:
    # Given
    texts = ["Hello world", "Test text"]
    dto = CreateEmbeddingsDto(
        texts=texts,
        include_dense=True,
        include_sparse=False,
        include_colbert=True,
    )

    # When
    serialized = dto.model_dump()

    # Then
    expected = {
        "texts": ["Hello world", "Test text"],
        "include_dense": True,
        "include_sparse": False,
        "include_colbert": True,
    }
    assert serialized == expected


def test_create_embeddings_dto_deserialization() -> None:
    # Given
    data = {
        "texts": ["Hello world", "Test text"],
        "include_dense": True,
        "include_sparse": False,
        "include_colbert": True,
    }

    # When
    dto = CreateEmbeddingsDto.model_validate(data)

    # Then
    assert dto.texts == ["Hello world", "Test text"]
    assert dto.include_dense is True
    assert dto.include_sparse is False
    assert dto.include_colbert is True
