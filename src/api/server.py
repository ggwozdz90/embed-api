import uvicorn
from fastapi import FastAPI

from api.handlers.global_exception_handler import GlobalExceptionHandler
from api.middlewares.process_time_middleware import ProcessTimeMiddleware
from api.routers.embedding_router import EmbeddingRouter
from api.routers.health_check_router import HealthCheckRouter
from api.routers.model_router import ModelRouter
from core.config.app_config import AppConfig
from core.logger.logger import Logger


class APIServer:
    def __init__(
        self,
        config: AppConfig,
        logger: Logger,
    ) -> None:
        self.config = config
        self.logger = logger
        self.app = FastAPI()
        self.exception_handler = GlobalExceptionHandler(self.app, logger)
        self.app.add_middleware(ProcessTimeMiddleware, logger=logger)
        self.app.include_router(HealthCheckRouter().router, tags=["HealthCheck"])
        self.app.include_router(EmbeddingRouter().router, tags=["Embeddings"])
        self.app.include_router(ModelRouter().router, tags=["Model"])

    def start(self) -> None:
        self.logger.info("Starting FastAPI server...")
        uvicorn.run(
            self.app,
            host=self.config.fastapi_host,
            port=self.config.fastapi_port,
            server_header=False,
            log_config=None,
        )
