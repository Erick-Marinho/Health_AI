from langchain_openai import ChatOpenAI
from app.core.config import settings

_llm_client_cache: ChatOpenAI | None = None

def get_llm_client() -> ChatOpenAI:
    """
    Retorna uma instância configurada do ChatOpenAI.
    Utiliza um cache simples para retornar a mesma instância se já criada.
    """
    global _llm_client_cache
    if _llm_client_cache is None:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY não está configurada nas variáveis de ambiente ou no arquivo .env.")
        
        _llm_client_cache = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL_NAME,
            temperature=settings.OPENAI_TEMPERATURE
            
        )
    return _llm_client_cache

