import httpx
import logging
import os
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ZapiClient:
    def __init__(self):
        # Credenciais Z-API (mantidas caso alguma outra parte do sistema as use, ou para recebimento)
        self.zapi_base_url = "https://api.z-api.io/instances"
        self.zapi_instance_id = os.getenv("ZAPI_INSTANCE_ID")
        self.zapi_instance_token = os.getenv("ZAPI_INSTANCE_TOKEN")
        self.zapi_client_security_token = os.getenv("ZAPI_CLIENT_SECURITY_TOKEN")

        # URL para enviar respostas via Z-API (não será mais usada por send_text_message se N8N_WEBHOOK_URL estiver configurada)
        self.zapi_send_text_url = None
        if self.zapi_instance_id and self.zapi_instance_token:
            self.zapi_send_text_url = f"{self.zapi_base_url}/{self.zapi_instance_id}/token/{self.zapi_instance_token}/send-text"

        # Nova URL para o webhook N8N
        self.n8n_webhook_url = os.getenv("N8N_WEBHOOK_URL")
        if not self.n8n_webhook_url:
            logger.warning("ZapiClient: A variável de ambiente N8N_WEBHOOK_URL não está configurada. As respostas não serão enviadas para o n8n.")
            # Você pode optar por levantar um ValueError aqui se o envio para n8n for mandatório
            # raise ValueError("N8N_WEBHOOK_URL não configurada nas variáveis de ambiente.")
        else:
            logger.info(f"ZapiClient: Respostas da aplicação serão enviadas para o webhook N8N: {self.n8n_webhook_url}")

        # Headers para Z-API (mantidos para referência ou outros usos)
        self.zapi_headers = {}
        if self.zapi_client_security_token:
            self.zapi_headers = {
                'Content-Type': 'application/json',
                'Client-Token': self.zapi_client_security_token
            }
        
        # Headers padrão para o webhook N8N (ajuste se necessário)
        self.n8n_headers = {
            'Content-Type': 'application/json'
        }

        logger.info("ZapiClient inicializado.")
        if not all([self.zapi_instance_id, self.zapi_instance_token, self.zapi_client_security_token]):
            logger.warning("ZapiClient: Alguma das credenciais Z-API (instância, token da instância ou client token) não está configurada. Funcionalidades da Z-API podem ser limitadas.")


    async def send_text_message(self, to_phone: str, message_text: str, original_received_message_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Envia uma mensagem de texto. Se N8N_WEBHOOK_URL estiver configurada,
        envia para o webhook N8N. Caso contrário, tenta enviar via Z-API (se configurada).
        """
        if not self.n8n_webhook_url:
            logger.error("N8N_WEBHOOK_URL não configurada. Não é possível enviar a mensagem de resposta.")
            return {"error": "ConfigurationError", "details": "N8N_WEBHOOK_URL não está configurada."}

        # Payload para o webhook N8N
        n8n_payload: Dict[str, Any] = {
            "phone": to_phone, # Número de telefone do destinatário original
            "message": message_text, # A resposta da IA
            "original_received_message_id": original_received_message_id # ID da mensagem original do usuário
        }
        
        payload_json = json.dumps(n8n_payload)
        
        logger.info(f"N8N_CLIENT: Enviando mensagem para webhook N8N. Destinatário original: {to_phone}. Texto: '{message_text[:50]}...'. URL: {self.n8n_webhook_url}")
        logger.debug(f"N8N_CLIENT: Payload: {payload_json}")
        logger.debug(f"N8N_CLIENT: Headers: {self.n8n_headers}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    self.n8n_webhook_url, 
                    content=payload_json,
                    headers=self.n8n_headers # Usar headers para N8N
                )
                response.raise_for_status() # Levanta exceção para status HTTP 4xx/5xx
                
                # Webhooks geralmente retornam 200, 201 ou 204 para sucesso.
                # O corpo da resposta pode ou não ser JSON ou relevante.
                response_content = response.text
                logger.info(f"N8N_CLIENT: Mensagem enviada com sucesso para o webhook N8N. Status: {response.status_code}. Resposta: {response_content[:200]}")
                return {"status_code": response.status_code, "response_body": response_content}
            except httpx.HTTPStatusError as e:
                logger.error(f"N8N_CLIENT: Erro HTTP ao enviar mensagem para webhook N8N. Status: {e.response.status_code}. Detalhes: {e.response.text}", exc_info=True)
                error_details = {"error": "HTTPStatusError", "status_code": e.response.status_code, "request_payload": n8n_payload}
                try:
                    error_details["response_body"] = e.response.json()
                except json.JSONDecodeError:
                    error_details["response_body"] = e.response.text
                return error_details
            except httpx.RequestError as e:
                logger.error(f"N8N_CLIENT: Erro de requisição ao enviar mensagem para webhook N8N (URL: {e.request.url}): {str(e)}", exc_info=True)
                return {"error": "RequestError", "details": str(e), "request_payload": n8n_payload}
            except Exception as e:
                logger.error(f"N8N_CLIENT: Erro inesperado ao enviar mensagem para webhook N8N: {str(e)}", exc_info=True)
                return {"error": "UnexpectedError", "details": str(e), "request_payload": n8n_payload}