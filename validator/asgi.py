import asyncio
import os

import uvicorn
from dotenv import load_dotenv

from validator.utils.util import try_db_connections


load_dotenv(os.getenv("ENV_FILE", ".env"))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from scalar_fastapi import get_scalar_api_reference

from validator.core.config import load_config
from validator.endpoints.auditing import factory_router as auditing_router
from validator.endpoints.health import factory_router as health_router
from validator.endpoints.miner_details import factory_router as miner_details_router
from validator.endpoints.tasks import factory_router as tasks_router
from validator.endpoints.tournament_analytics import factory_router as tournament_analytics_router
from validator.endpoints.tournament_orchestrator import factory_router as tournament_orchestrator_router
from validator.utils.logging import get_logger
from validator.utils.miner_analytics import miner_performance_cache_worker


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug("Entering lifespan context manager")
    config = load_config()
    await try_db_connections(config)

    app.state.config = config

    cache_task = asyncio.create_task(miner_performance_cache_worker(config))

    logger.info("Starting up...")

    yield

    logger.info("Shutting down...")
    cache_task.cancel()
    await config.psql_db.close()
    await config.redis_db.close()


def factory() -> FastAPI:
    logger.debug("Entering factory function")
    app = FastAPI(lifespan=lifespan)

    app.add_api_route(
        "/scalar",
        lambda: get_scalar_api_reference(openapi_url=app.openapi_url, title=app.title),
        methods=["GET"],
    )

    app.include_router(health_router())
    app.include_router(tasks_router())
    app.include_router(auditing_router())
    app.include_router(miner_details_router())
    app.include_router(tournament_analytics_router())
    app.include_router(tournament_orchestrator_router())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    FastAPIInstrumentor().instrument_app(app)

    logger.debug(f"App created with {len(app.routes)} routes")
    return app


if __name__ == "__main__":
    logger.info("Starting main validator")

    uvicorn.run(factory(), host="0.0.0.0", port=8010)
