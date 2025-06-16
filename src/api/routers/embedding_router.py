from typing import Annotated

from fastapi import APIRouter, Body, Depends

from api.dtos.create_embeddings_dto import CreateEmbeddingsDto
from api.dtos.create_embeddings_result_dto import CreateEmbeddingsResultDto
from application.usecases.create_embeddings_usecase import CreateEmbeddingsUseCase


class EmbeddingRouter:
    def __init__(self) -> None:
        self.router = APIRouter()
        self.router.post("/embeddings")(self.create_embeddings)

    async def create_embeddings(
        self,
        create_embeddings_usecase: Annotated[CreateEmbeddingsUseCase, Depends()],
        create_embeddings_dto: CreateEmbeddingsDto = Body(...),
    ) -> CreateEmbeddingsResultDto:
        result = await create_embeddings_usecase.execute(
            create_embeddings_dto.texts,
            create_embeddings_dto.include_dense,
            create_embeddings_dto.include_sparse,
            create_embeddings_dto.include_colbert,
        )

        return CreateEmbeddingsResultDto(embeddings=result.embeddings)
