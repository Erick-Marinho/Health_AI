from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"  
    OPENAI_TEMPERATURE: float = 0.2        

    APPHEALTH_API_TOKEN: str
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

if __name__ == "__main__":
    print("Configurações carregadas:")
    print(f"  OpenAI API Key: {'*' * (len(settings.OPENAI_API_KEY) - 4) + settings.OPENAI_API_KEY[-4:] if settings.OPENAI_API_KEY else 'Não definida'}")
    print(f"  OpenAI Model Name: {settings.OPENAI_MODEL_NAME}")
    print(f"  OpenAI Temperature: {settings.OPENAI_TEMPERATURE}")