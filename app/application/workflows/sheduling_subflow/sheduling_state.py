from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class SchedulingWorkflowState(TypedDict):
    """
    Estado específico para o subfluxo de coleta de informações de agendamento.
    """
    messages: Annotated[List[BaseMessage], add_messages] # Histórico de mensagens do subfluxo
    nome_completo_usuario: Optional[str]
    proximo_passo_agendamento: Optional[str] # Ex: "SOLICITAR_NOME", "CONFIRMAR_NOME"
    resposta_para_usuario: Optional[str] # Mensagem gerada pelo subfluxo    
    agendamento_concluido: bool # Indica se o agendamento foi concluído com sucesso
    contador_teste: Optional[int]