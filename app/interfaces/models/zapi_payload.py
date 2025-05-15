from typing import Optional, Dict
from pydantic import BaseModel, Field

class ZapiTextMessage(BaseModel):
    """
    Pydantic model para representar o payload recebido da API do Zapi.
    """
    message: Optional[str] = None
    
class ZapiReceivedMessagePayload(BaseModel):
    """
    Pydantic model para representar o payload recebido da API do Zapi.
    """
    
    phone: str
    text: Optional[ZapiTextMessage] = None
    message_id: str = Field(alias="messageId")
    from_me: bool = Field(alias="fromMe")

    is_group: Optional[bool] = Field(alias="isGroup")

    class Config:
        populate_by_name = True
        extra = "ignore"