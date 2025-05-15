from typing import Annotated, Optional, TypedDict, List, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
#from app.domain.models.user_profile import FullNameModel

class MainWorkflowState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    categoria: Optional[str] = None
    current_operation: Optional[str]
    response_to_user: Optional[str] = None

    # --- Campos para o fluxo de Agendamento ---
    scheduling_step: Optional[str] = None
    user_full_name: Optional[str] = None
    user_chosen_specialty: Optional[str] = None
    user_chosen_specialty_id: Optional[int]

    # --- NOVOS CAMPOS PARA PREFERÊNCIA DE PROFISSIONAL ---
    professional_preference_type: Optional[str] = None 
    user_provided_professional_name: Optional[str] = None # Nome que o usuário digitou

    user_chosen_professional_id: Optional[int] # ID do profissional da API, após validação/escolha
    user_chosen_professional_name: Optional[str] # Nome oficial do profissional da API, após validação/escolha

    available_professionals_list: Optional[List[Dict]] 

    user_chosen_turn: Optional[str] # NOVO: "MANHA" ou "TARDE"
    available_dates_presented: Optional[List[str]] # Lista de datas (formato "AAAA-MM-DD") apresentadas
    user_chosen_date: Optional[str] # Data escolhida (formato "AAAA-MM-DD")
    available_times_presented: Optional[List[Dict[str, str]]] # Lista de horários (formato "HH:MM") apresentados
    user_chosen_time: Optional[str] # Horário escolhido (formato "HH:MM")
    user_chosen_time_fim: Optional[str] # Horário escolhido (formato "HH:MM")

    scheduling_completed: bool
    scheduling_values_confirmed: Optional[Dict]