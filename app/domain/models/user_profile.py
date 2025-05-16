import re
from pydantic import BaseModel, field_validator, Field

class FullNameModel(BaseModel):
    """
    Representa e valida o nome completo de um usuário/paciente.
    """
    full_name: str = Field(description="O nome completo do paciente.")

    @field_validator('full_name', mode='before')
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("O nome completo deve ser uma string.")
        
        name_cleaned = v.strip()

        if not name_cleaned:
            raise ValueError("O nome completo não pode ser vazio.")
        
        if len(name_cleaned.split()) < 2:
            raise ValueError("O nome completo deve conter pelo menos um nome e um sobrenome.")
        
        if not re.match(r'^[a-zA-ZÀ-ÿ\s]+$', name_cleaned):
            raise ValueError("O nome completo deve conter apenas letras e espaços.")
        
        return name_cleaned