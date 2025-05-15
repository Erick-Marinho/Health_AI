import logging
import os
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError
from app.interfaces.models.whatsapp_payload import WhatsAppWebhookPayload
from app.application.workflows.main_conversation_flow import arun_main_conversation_flow
from app.infrastructure.clients.whatsapp_client import WhatsAppClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = logging.getLogger(__name__)
router = APIRouter()

VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

async def process_incoming_whatsapp_message(
    payload: WhatsAppWebhookPayload, 
    db_pool: AsyncConnectionPool 
):
    try:
        logger.info("Payload validado com sucesso em background.")
        change_value = payload.entry[0].changes[0].value
        
        if change_value.messages:
            message_obj = change_value.messages[0]
            if message_obj.type == "text" and message_obj.text:
                user_text = message_obj.text.body
                session_id = message_obj.from_number
                message_id = message_obj.id
                logger.info(f"Processando como mensagem de entrada do usuário.")
                logger.info(f"Mensagem (ID: {message_id}) de {session_id}: '{user_text}'")

                logger.info(f"Direcionando mensagem para arun_scheduling_flow para session_id: {session_id}")

                agent_response_text = None
                
                async with db_pool.connection() as conn:
                    if conn is None:
                        logger.error(f"ERRO CRÍTICO: Falha ao obter conexão do db_pool para session_id: {session_id}.")
                        return

                    logger.debug(f"Conexão obtida do pool para session_id: {session_id}")
                    
                    checkpointer = AsyncPostgresSaver(conn=conn)
                    
                    agent_response_text = await arun_main_conversation_flow(
                        user_text=user_text,
                        session_id=session_id,
                        checkpointer=checkpointer 
                    )

                if agent_response_text:
                    whatsapp_client = WhatsAppClient() 
                    await whatsapp_client.send_text_message(to=session_id, text=agent_response_text)
                else:
                    logger.warning(f"Nenhuma resposta do agente para a mensagem de {session_id}")

            elif message_obj.type == "image":
                logger.info(f"Recebida mensagem do tipo imagem de {message_obj.from_}. Não processada.")
            else:
                logger.info(f"Recebido tipo de mensagem não tratado: {message_obj.type} de {message_obj.from_}")
        
        elif change_value.statuses:
            status_obj = change_value.statuses[0]
            message_id = status_obj.id
            status_type_str = status_obj.status
            logger.info(f"Recebida notificação de status: {status_type_str} para mensagem ID {message_id}")

        else:
            logger.info("Payload não continha mensagens de usuário ou status esperados.")

    except ValidationError as ve:
        logger.error(f"Erro de validação Pydantic no processamento em background: {ve.errors()}", exc_info=True)
    except Exception as e:
        logger.error(f"Erro inesperado no processamento em background: {e}", exc_info=True)

@router.post("/whatsapp", summary="Webhook principal para receber mensagens do WhatsApp")
async def whatsapp_main_webhook(
    request: Request, 
    payload_dict: dict,
    background_tasks: BackgroundTasks
):
    logger.info("Recebido payload POST do WhatsApp no endpoint principal.")
    logger.info(f"RAW WhatsApp Payload Recebido: {payload_dict}")
    try:
        payload = WhatsAppWebhookPayload.model_validate(payload_dict)
        
        
        db_pool_from_state = request.app.state.db_pool
        if not db_pool_from_state:
            logger.error("CRITICAL: db_pool não encontrado em request.app.state. Não é possível processar a mensagem.")
            raise HTTPException(status_code=500, detail="Configuração interna do servidor incorreta (DB Pool).")

        background_tasks.add_task(process_incoming_whatsapp_message, payload, db_pool_from_state)
        
        logger.info("Tarefa de processamento adicionada ao background. Retornando 200 OK.")
        return {"status": "success", "message": "Webhook recebido e processamento iniciado."}

    except ValidationError as ve:
        logger.error(f"Erro de validação Pydantic ao receber webhook: {ve.errors()}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Payload inválido: {ve.errors()}")
    except Exception as e:
        logger.error(f"Erro inesperado ao receber webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno do servidor.")


@router.get("/whatsapp", summary="Verificação do Webhook do WhatsApp (GET)")
async def whatsapp_verify_webhook(request: Request):
    logger.info(f"Recebida requisição GET para verificação do webhook. Query params: {request.query_params}")
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            logger.info("Token de verificação válido. Respondendo com o challenge.")
            return int(challenge)
        else:
            logger.warning("Falha na verificação do webhook: Modo ou Token inválidos.")
            raise HTTPException(status_code=403, detail="Falha na verificação: Modo ou Token inválidos.")
    else:
        logger.warning("Falha na verificação do webhook: Parâmetros ausentes.")
        raise HTTPException(status_code=400, detail="Falha na verificação: Parâmetros ausentes.")
