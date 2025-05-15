from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Configurações do OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"  # Valor padrão se não definido no .env
    OPENAI_TEMPERATURE: float = 0.2        # Valor padrão

    # Adicione outras configurações da sua aplicação aqui, por exemplo:
    APPHEALTH_API_TOKEN: str
    # DATABASE_URL: str
    # ZAPI_INSTANCE_TOKEN: str
    # ZAPI_CLIENT_TOKEN: str
    # PROJECT_NAME: str = "HealthAI Assistant"
    # API_V1_STR: str = "/api/v1"

    # Configuração para Pydantic-settings carregar de variáveis de ambiente
    # e de arquivos .env
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Cria uma instância global das configurações que pode ser importada em outros módulos
settings = Settings()

#Para verificar se as configurações estão sendo carregadas (opcional, para debug):
if __name__ == "__main__":
    print("Configurações carregadas:")
    print(f"  OpenAI API Key: {'*' * (len(settings.OPENAI_API_KEY) - 4) + settings.OPENAI_API_KEY[-4:] if settings.OPENAI_API_KEY else 'Não definida'}")
    print(f"  OpenAI Model Name: {settings.OPENAI_MODEL_NAME}")
    print(f"  OpenAI Temperature: {settings.OPENAI_TEMPERATURE}")