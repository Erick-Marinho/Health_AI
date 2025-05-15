# Em: app/infrastructure/clients/whatsapp_client.py
import os
import httpx # Usar httpx para requisições assíncronas
import logging
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
# É bom ter load_dotenv() aqui se este módulo puder ser usado de forma independente,
# mas se app.main.py já o chama no início, pode não ser estritamente necessário
# que cada módulo o chame novamente. No entanto, não prejudica.
load_dotenv()

logger = logging.getLogger(__name__)

WHATSAPP_API_TOKEN = os.getenv("WHATSAPP_API_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID") # Nome da variável como usado antes
WHATSAPP_API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v19.0") # Permitir configuração da versão via .env, com fallback

class WhatsAppClient:
    def __init__(self):
        if not WHATSAPP_API_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
            logger.error("CRÍTICO: Token da API do WhatsApp ou ID do número de telefone não configurados nas variáveis de ambiente.")
            # Considerar levantar um erro aqui para falhar rapidamente se a configuração estiver ausente
            # raise ValueError("Configuração do WhatsApp (API Token ou Phone Number ID) ausente.")
            self.is_configured = False
        else:
            self.is_configured = True
            self.base_url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
            self.headers = {
                "Authorization": f"Bearer {WHATSAPP_API_TOKEN}",
                "Content-Type": "application/json",
            }
            logger.info(f"WhatsAppClient inicializado. Base URL: {self.base_url.replace(WHATSAPP_PHONE_NUMBER_ID, 'PHONE_ID_HIDDEN')}")


    async def send_text_message(self, to: str, text: str):
        if not self.is_configured:
            logger.error(f"Não é possível enviar mensagem para {to}: WhatsAppClient não está configurado (API Token ou Phone ID ausente).")
            # Poderia retornar um objeto de resposta de erro ou levantar uma exceção customizada
            return None 

        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": text}, # preview_url: False é uma boa prática para evitar surpresas
        }

        logger.info(f"Enviando mensagem para {to}: '{text}' via WhatsAppClient")
        async with httpx.AsyncClient(timeout=15.0) as client: # Adicionado timeout
            try:
                response = await client.post(self.base_url, json=payload, headers=self.headers)
                response.raise_for_status()  # Levanta uma exceção para respostas HTTP 4xx/5xx
                
                response_data = response.json()
                logger.info(f"Mensagem enviada com sucesso para {to}. Response: {response_data}")
                return response_data
            except httpx.HTTPStatusError as e:
                error_response_text = "No response text"
                try:
                    error_details = e.response.json()
                    error_response_text = f"Details: {error_details}"
                except Exception:
                    error_response_text = e.response.text if e.response.text else "No response text"
                
                logger.error(
                    f"Erro HTTP ao enviar mensagem para {to}: {e.response.status_code} - {error_response_text}", 
                    exc_info=True
                )
                # Você pode querer retornar o erro da API do WhatsApp ou uma mensagem genérica
                return {"status": "error", "code": e.response.status_code, "details": error_details if 'error_details' in locals() else error_response_text}
            except httpx.RequestError as e:
                logger.error(f"Erro de requisição (ex: timeout, DNS) ao enviar mensagem para {to}: {e}", exc_info=True)
                return {"status": "error", "message": f"Erro de requisição: {type(e).__name__}"}
            except Exception as e: # Captura genérica para outros erros inesperados
                logger.error(f"Erro inesperado ao enviar mensagem para {to}: {e}", exc_info=True)
                return {"status": "error", "message": "Erro inesperado no cliente WhatsApp."}
