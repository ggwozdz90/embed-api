import multiprocessing
import multiprocessing.connection
import multiprocessing.synchronize
from dataclasses import dataclass
from multiprocessing.sharedctypes import Synchronized
from typing import Any, Dict, List

from FlagEmbedding import BGEM3FlagModel

from data.workers.base_worker import BaseWorker
from domain.exceptions.worker_not_running_error import WorkerNotRunningError


@dataclass
class BgeM3EmbeddingConfig:
    device: str
    model_name: str
    log_level: str


@dataclass
class EmbeddingRequest:
    texts: List[str]
    include_dense: bool
    include_sparse: bool
    include_colbert: bool


@dataclass
class EmbeddingResult:
    embeddings: List[Dict[str, Any]]


class BgeM3EmbeddingWorker(
    BaseWorker[  # type: ignore
        EmbeddingRequest,
        EmbeddingResult,
        BgeM3EmbeddingConfig,
        BGEM3FlagModel,
    ],
):
    def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResult:
        if not self.is_alive():
            raise WorkerNotRunningError()

        self._pipe_parent.send(("create_embeddings", request))
        result = self._pipe_parent.recv()

        if isinstance(result, Exception):
            raise result

        return result  # type: ignore

    def initialize_shared_object(
        self,
        config: BgeM3EmbeddingConfig,
    ) -> BGEM3FlagModel:
        model = BGEM3FlagModel(
            config.model_name,
            device=config.device,
        )

        return model

    def handle_command(
        self,
        command: str,
        args: EmbeddingRequest,
        shared_object: BGEM3FlagModel,
        config: BgeM3EmbeddingConfig,
        pipe: multiprocessing.connection.Connection,
        is_processing: Synchronized,  # type: ignore
        processing_lock: multiprocessing.synchronize.Lock,
    ) -> None:
        if command == "create_embeddings":
            try:
                with processing_lock:
                    is_processing.value = True

                request = args
                model = shared_object

                embeddings = model.encode(
                    request.texts,
                    batch_size=12,
                    max_length=8192,
                    return_dense=request.include_dense,
                    return_sparse=request.include_sparse,
                    return_colbert_vecs=request.include_colbert,
                )

                text_embeddings = []

                for i, text in enumerate(request.texts):
                    text_embedding: Dict[str, Any] = {"text": text}

                    if request.include_dense and "dense_vecs" in embeddings:
                        text_embedding["dense"] = embeddings["dense_vecs"][i].tolist()

                    if request.include_sparse and "lexical_weights" in embeddings:
                        indices = []
                        values = []
                        for key, value in embeddings["lexical_weights"][i].items():
                            indices.append(int(key))
                            values.append(float(value))

                        text_embedding["sparse"] = {"indices": indices, "values": values}

                    if request.include_colbert and "colbert_vecs" in embeddings:
                        text_embedding["colbert"] = embeddings["colbert_vecs"][i].tolist()

                    text_embeddings.append(text_embedding)

                result = EmbeddingResult(embeddings=text_embeddings)

                pipe.send(result)

            except Exception as e:
                pipe.send(e)

            finally:
                with processing_lock:
                    is_processing.value = False

    def get_worker_name(self) -> str:
        return type(self).__name__
