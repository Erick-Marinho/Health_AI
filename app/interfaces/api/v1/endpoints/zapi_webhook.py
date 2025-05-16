import logging
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError

from app.interfaces.models.zapi_payload import ZapiReceivedMessagePayload
from app.application.workflows.main_conversation_flow import arun_main_conversation_flow
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.infrastructure.clients.zapi_client import ZapiClient

logger = logging.getLogger(__name__)
router = APIRouter()

async def process_incoming_zapi_message(
    payload: ZapiReceivedMessagePayload, 
    db_pool: AsyncConnectionPool 
):
    try:
        logger.info(f"ZAPI_WEBHOOK: Payload Z-API validado com sucesso. Message ID: {payload.message_id}")

        if payload.from_me:
            logger.info(f"ZAPI_WEBHOOK: Mensagem de {payload.phone} (ID: {payload.message_id}) é 'fromMe'. Ignorando.")
            return

        if payload.is_group:
            logger.info(f"ZAPI_WEBHOOK: Mensagem de {payload.phone} (ID: {payload.message_id}) é de grupo. Ignorando por enquanto.")
            return

        user_text = None
        if payload.text and payload.text.message:
            user_text = payload.text.message
        
        if not user_text:
            logger.info(f"ZAPI_WEBHOOK: Mensagem de {payload.phone} (ID: {payload.message_id}) não contém texto ou o campo esperado está vazio. Ignorando.")
            return

        session_id = payload.phone
        user_phone_number = payload.phone
        
        logger.info(f"ZAPI_WEBHOOK: Processando mensagem (ID Z-API: {payload.message_id}) de {session_id} (Telefone: {user_phone_number}) : '{user_text}'")
        logger.info(f"ZAPI_WEBHOOK: Direcionando para arun_scheduling_flow para session_id: {session_id}")

        agent_response_text = None
        async with db_pool.connection() as conn:
            if conn is None:
                logger.error(f"ZAPI_WEBHOOK: ERRO CRÍTICO: Falha ao obter conexão do db_pool para session_id: {session_id}.")
                return 
            
            logger.debug(f"ZAPI_WEBHOOK: Conexão obtida do pool para session_id: {session_id}")
            checkpointer = AsyncPostgresSaver(conn=conn)
            
            agent_response_text = await arun_main_conversation_flow(
                user_text=user_text,
                session_id=session_id,
                checkpointer=checkpointer,
                user_phone_number=user_phone_number
            )

        if agent_response_text:
            logger.info(f"ZAPI_WEBHOOK: Resposta da IA para {session_id}: {agent_response_text}")
            
            try:
                logger.debug("ZAPI_WEBHOOK: Tentando instanciar ZapiClient...")
                zapi_client = ZapiClient() 
                logger.debug("ZAPI_WEBHOOK: ZapiClient instanciado. Enviando mensagem...")
                
                response_status = await zapi_client.send_text_message(
                    to_phone=session_id, 
                    message_text=agent_response_text,
                    original_received_message_id=payload.message_id 
                )
                logger.info(f"ZAPI_WEBHOOK: Status do envio da resposta via Z-API: {response_status}")
                if isinstance(response_status, dict) and response_status.get("error"):
                    logger.error(f"ZAPI_WEBHOOK: Falha ao enviar mensagem via Z-API. Detalhes: {response_status.get('details')}")

            except ValueError as e:
                logger.error(f"ZAPI_WEBHOOK: Falha ao inicializar ZapiClient (credenciais ZAPI ausentes?): {e}")
            except Exception as e: 
                logger.error(f"ZAPI_WEBHOOK: Erro inesperado ao tentar enviar resposta via ZapiClient: {e}", exc_info=True)
        else:
            logger.warning(f"ZAPI_WEBHOOK: Nenhuma resposta do agente para a mensagem Z-API de {session_id}")

    except ValidationError as ve:
        logger.error(f"ZAPI_WEBHOOK: Erro de validação Pydantic no payload Z-API: {ve.errors()}", exc_info=True)
    except Exception as e:
        logger.error(f"ZAPI_WEBHOOK: Erro inesperado no processamento da mensagem Z-API: {e}", exc_info=True)


@router.post("/zapi", summary="Webhook para receber mensagens da Z-API")
async def zapi_on_message_received_webhook(
    request: Request, 
    payload_dict: dict,
    background_tasks: BackgroundTasks
):
    logger.info("ZAPI_WEBHOOK: Recebido payload POST no endpoint /zapi.")
    logger.debug(f"ZAPI_WEBHOOK: RAW Payload Recebido: {payload_dict}")
    
    try:
        payload = ZapiReceivedMessagePayload.model_validate(payload_dict)
        
        db_pool_from_state = request.app.state.db_pool
        if not db_pool_from_state:
            logger.error("ZAPI_WEBHOOK: CRITICAL: db_pool não encontrado em request.app.state.")
            raise HTTPException(status_code=500, detail="Configuração interna do servidor incorreta (DB Pool).")

        background_tasks.add_task(process_incoming_zapi_message, payload, db_pool_from_state)
        
        logger.info("ZAPI_WEBHOOK: Tarefa de processamento Z-API adicionada ao background. Retornando 200 OK.")
        return {"status": "zapi_webhook_payload_received_for_processing"}

    except ValidationError as ve:
        logger.error(f"ZAPI_WEBHOOK: Erro de validação Pydantic no payload Z-API: {ve.errors()}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Payload Z-API inválido: {ve.errors()}")
    except Exception as e:
        logger.error(f"ZAPI_WEBHOOK: Erro inesperado ao receber webhook Z-API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno do servidor ao processar webhook Z-API.")
