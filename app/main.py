import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import Optional

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando aplicação e ciclo de vida (lifespan)...")
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB")

    logger.debug(f"Lifespan: Lidas variáveis do BD: DB_USER='{db_user}', DB_PASSWORD='{'*' * len(db_password) if db_password else 'None'}', DB_HOST='{db_host}', DB_PORT='{db_port}', DB_NAME='{db_name}'")

    if db_user and db_password and db_host and db_port and db_name:
        conninfo_str_lifespan = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        logger.info(f"Lifespan: String de conexão para o pool: {conninfo_str_lifespan.replace(db_password, '********') if db_password else conninfo_str_lifespan}")
        
        try:
            async with AsyncConnectionPool(
                conninfo=conninfo_str_lifespan,
                min_size=1,
                max_size=5
            ) as pool:
                app.state.db_pool = pool
                logger.info("Lifespan: AsyncConnectionPool criado e armazenado em app.state.db_pool.")

                logger.info("Lifespan: Tentando executar AsyncPostgresSaver.setup()...")
                async with pool.connection() as setup_conn:
                    await setup_conn.set_autocommit(True)
                    try:
                        checkpointer_for_setup = AsyncPostgresSaver(conn=setup_conn)
                        await checkpointer_for_setup.setup()
                        logger.info("Lifespan: AsyncPostgresSaver.setup() executado com sucesso.")
                    except Exception as setup_exc:
                        logger.error(f"Lifespan: Erro durante AsyncPostgresSaver.setup(): {setup_exc}", exc_info=True)
                        raise RuntimeError(f"Falha crítica no setup do checkpointer: {setup_exc}") from setup_exc

                yield

        except Exception as pool_exc:
            logger.error(f"Lifespan: Erro CRÍTICO ao criar ou abrir AsyncConnectionPool: {pool_exc}", exc_info=True)
            raise RuntimeError(f"Falha ao inicializar o pool de conexões: {pool_exc}") from pool_exc
    else:
        logger.error("Lifespan: Variáveis de ambiente para o banco de dados estão ausentes ou incompletas. Pool de conexões NÃO será criado. O Checkpointer setup NÃO será executado.")
        yield

    logger.info("Encerrando aplicação e ciclo de vida (lifespan)...")
    if hasattr(app.state, 'db_pool') and app.state.db_pool:
        logger.info("Lifespan: Fechando AsyncConnectionPool...")
        await app.state.db_pool.close()
        logger.info("Lifespan: AsyncConnectionPool fechado.")


from app.interfaces.api.v1.endpoints import whatsapp_webhook, zapi_webhook

app = FastAPI(
    title="HealthAI Assistant API",
    description="API para interagir com o assistente virtual da HealthAI via Webhooks.",
    version="0.1.0",
    lifespan=lifespan
)

@app.get("/")
async def read_root():
    return {"message": "HealthAI Assistant API is running!"}

# app.include_router(
#     whatsapp_webhook.router,
#     prefix="/api/v1/webhooks",
#     tags=["WhatsApp"]
# )

app.include_router(zapi_webhook.router, prefix="/api/v1/webhooks", tags=["WhatsApp Z-API"])

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
