from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class UserQueryInput(BaseModel):
    text: str = Field(..., description="O texto da mensagem do usuário")
    session_id: str = Field(..., description="Um identificador para a sessão da conversa")

class WhatsAppMessageText(BaseModel):
    body: str

class WhatsAppStatus(BaseModel):
    id: str
    status: str
    timestamp: str
    recipient_id: str
    conversation: Optional[Dict[str, Any]] = None 
    pricing: Optional[Dict[str, Any]] = None 

class WhatsAppMessage(BaseModel):
    id: str # ID da mensagem do WhatsApp
    from_number: str = Field(alias="from") 
    timestamp: str
    type: str 
    text: Optional[WhatsAppMessageText] = None 

class WhatsAppValue(BaseModel):
    messaging_product: str
    metadata: dict
    contacts: Optional[List[dict]] = None
    messages: Optional[List[WhatsAppMessage]] = None
    statuses: Optional[List[WhatsAppStatus]] = None

class WhatsAppChange(BaseModel):
    value: WhatsAppValue
    field: str

class WhatsAppEntry(BaseModel):
    id: str
    changes: List[WhatsAppChange]

class WhatsAppWebhookPayload(BaseModel):
    object: str
    entry: List[WhatsAppEntry]
