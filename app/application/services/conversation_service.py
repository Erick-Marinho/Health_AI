import logging
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.application.prompts.conversation_prompts import (
    CATEGORIZATION_PROMPT_TEMPLATE, GREETING_FAREWELL_PROMPT_TEMPLATE
)
logger = logging.getLogger(__name__)

def get_last_user_message_content(messages: List[BaseMessage]) -> Optional[str]:
    """Extrai o conteúdo da última mensagem do usuário (HumanMessage)."""
    if not messages:
        return None
    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        return last_message.content
    return None

def categorize_intent_service(user_query: str, llm_client: ChatOpenAI) -> str:
    """
    Categoriza a intenção do usuário usando o LLM.
    """
    if not user_query:
        return "Indefinido" # Ou uma categoria padrão de erro/fallback
    
    chain = CATEGORIZATION_PROMPT_TEMPLATE | llm_client
    try:
        response = chain.invoke({"user_query": user_query})
        categoria_llm = response.content
        # Limpeza adicional para garantir que apenas a categoria seja retornada
        categoria_limpa = categoria_llm.replace('Categoria: ', '').replace('"', '').strip()
        return categoria_limpa
    except Exception as e:
        logger.error(f"Erro durante a categorização com LLM: {e}")
        return "Erro na Categorização" # Categoria de fallback em caso de erro

def generate_greeting_farewell_service(user_query: Optional[str], llm_client: ChatOpenAI) -> str:
    """
    Gera uma resposta de saudação ou despedida humanizada usando o LLM.
    """
    if not user_query:
        user_query = "Olá" # Default para o prompt

    chain = GREETING_FAREWELL_PROMPT_TEMPLATE | llm_client
    logger.info(f"Geração de saudação/despedida com LLM: {user_query}")
    try:
        response = chain.invoke({"user_message": user_query})
        logger.info(f"Resposta gerada pelo LLM: {response.content.strip()}")
        return response.content.strip()
    except Exception as e:
        logger.error(f"Erro ao gerar saudação/despedida com LLM: {e}")
        # Fallback para uma resposta genérica em caso de erro do LLM
        if user_query and any(term in user_query.lower() for term in ["tchau", "até mais", "obrigado", "adeus", "encerrar", "fim"]):
            return "Até logo! Se precisar de algo mais, estou à disposição."
        return "Olá! Como posso ajudar?"