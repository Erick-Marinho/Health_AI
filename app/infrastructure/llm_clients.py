from langchain_openai import ChatOpenAI
from app.core.config import settings # Importa a instância das configurações

# Cache para a instância do LLM para evitar recriação desnecessária
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
            # Você pode adicionar outros parâmetros aqui se necessário,
            # como max_tokens, etc., e configurá-los via settings.
        )
        # print("DEBUG: Nova instância ChatOpenAI criada.") # Para debug
    # else:
        # print("DEBUG: Usando instância ChatOpenAI do cache.") # Para debug
    return _llm_client_cache

# Opcional: Uma função para limpar o cache, útil para testes se você precisar
# forçar a recriação do cliente com configurações diferentes (embora para settings
# carregadas no início, isso seja menos comum).
# def clear_llm_client_cache():
#     global _llm_client_cache
#     _llm_client_cache = None