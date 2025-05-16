from typing import Annotated, Optional, TypedDict, List, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
#from app.domain.models.user_profile import FullNameModel

class MainWorkflowState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    categoria: Optional[str] = None
    current_operation: Optional[str]
    response_to_user: Optional[str] = None

    scheduling_step: Optional[str] = None
    user_full_name: Optional[str] = None
    user_phone: Optional[str] = None 
    user_chosen_specialty: Optional[str] = None
    user_chosen_specialty_id: Optional[int]

    professional_preference_type: Optional[str] = None 
    user_provided_professional_name: Optional[str] = None 

    user_chosen_professional_id: Optional[int] 
    user_chosen_professional_name: Optional[str]

    available_professionals_list: Optional[List[Dict]] 

    user_chosen_turn: Optional[str] 
    available_dates_presented: Optional[List[str]] 
    user_chosen_date: Optional[str] 
    available_times_presented: Optional[List[Dict[str, str]]] 
    user_chosen_time: Optional[str] 
    user_chosen_time_fim: Optional[str] 

    scheduling_completed: bool
    scheduling_values_confirmed: Optional[Dict]