import logging
import json
import re
import requests

from datetime import datetime, date, timedelta, time
from functools import partial
from typing import Optional, List, Dict
from app.core.config import settings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.prompts import ChatPromptTemplate

from .main_workflow_state import MainWorkflowState
from app.application.services.conversation_service import (
    categorize_intent_service,
    generate_greeting_farewell_service,
    get_last_user_message_content
)
from app.application.prompts.conversation_prompts import (
    REQUEST_FULL_NAME_PROMPT_TEMPLATE, 
    REQUEST_SPECIALTY_PROMPT_TEMPLATE,
    REQUEST_PROFESSIONAL_PREFERENCE_PROMPT_TEMPLATE, 
    CLASSIFY_PROFESSIONAL_PREFERENCE_PROMPT_TEMPLATE, 
    REQUEST_DATE_TIME_PROMPT_TEMPLATE,
    REQUEST_SPECIFIC_PROFESSIONAL_NAME_PROMPT_TEMPLATE, 
    REQUEST_TURN_PREFERENCE_PROMPT_TEMPLATE,
    VALIDATE_SPECIALTY_PROMPT_TEMPLATE,
    MATCH_OFFICIAL_SPECIALTY_PROMPT_TEMPLATE,
    MATCH_SPECIFIC_PROFESSIONAL_NAME_PROMPT_TEMPLATE,
    VALIDATE_CHOSEN_DATE_PROMPT_TEMPLATE,
    PRESENT_AVAILABLE_TIMES_PROMPT_TEMPLATE,
    VALIDATE_CHOSEN_TIME_PROMPT_TEMPLATE,
    FINAL_SCHEDULING_CONFIRMATION_PROMPT_TEMPLATE,
    VALIDATE_FINAL_CONFIRMATION_PROMPT_TEMPLATE,
    EXTRACT_FULL_NAME_PROMPT_TEMPLATE,
    CHECK_CANCELLATION_PROMPT_TEMPLATE,
    SCHEDULING_SUCCESS_MESSAGE_PROMPT_TEMPLATE,
    VALIDATE_FALLBACK_CHOICE_PROMPT_TEMPLATE
)
from app.domain.models.user_profile import FullNameModel
from app.infrastructure.llm_clients import get_llm_client

logger = logging.getLogger(__name__)

# === NÓS DO GRAFO PRINCIPAL ===
def categorize_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Nó para categorizar a intenção do usuário.
    Chama o serviço de categorização, passando o llm_client.
    """
    logger.debug("--- Nó: categorize_node ---")
    user_message_content = get_last_user_message_content(state["messages"])
    
    if not user_message_content:
        logger.warning("Nenhuma mensagem de usuário para categorizar.")
        return {
            "response_to_user": "Não consegui entender sua solicitação. Pode tentar novamente?",
            "current_operation": None,
            "categoria": "Indefinido"
        }

    # Passa o llm_client para o serviço
    categoria = categorize_intent_service(user_message_content, llm_client)
    logger.info(f"Intenção do usuário categorizada como: '{categoria}'")
    return {"categoria": categoria}

def greeting_farewell_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Nó para lidar com saudações ou despedidas.
    Chama o serviço para gerar a resposta apropriada, passando o llm_client.
    """
    logger.debug("--- Nó: greeting_farewell_node ---")
    user_message_content = get_last_user_message_content(state["messages"])
    
    response_text = generate_greeting_farewell_service(user_message_content, llm_client)
    logger.info(f"Resposta de saudação/despedida gerada: '{response_text}'")
    
    return {"response_to_user": response_text, "current_operation": None}

def solicitar_nome_agendamento_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Nó para solicitar o nome completo ao iniciar o agendamento.
    """
    logger.debug("--- Nó Agendamento: solicitar_nome_agendamento_node ---")
    
    
    prompt = REQUEST_FULL_NAME_PROMPT_TEMPLATE.format_messages()
    logger.debug(f"Gerando solicitação de nome completo com o LLM. Prompt: {prompt}")
    ai_response = llm_client.invoke(prompt)
    resposta_llm = ai_response.content.strip()
    logger.info(f"Resposta do LLM para solicitação de nome completo: {resposta_llm}")

    return {
        "response_to_user": resposta_llm,
        "scheduling_step": "VALIDATING_FULL_NAME", 
        "current_operation": "SCHEDULING"
    }

def coletar_validar_nome_agendamento_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Nó para coletar a resposta do usuário, extrair o nome e validá-lo.
    """
    logger.debug("--- Nó Agendamento: coletar_validar_nome_agendamento_node ---")
    user_message_content = get_last_user_message_content(state["messages"])

    if not user_message_content:
        logger.warning("Nenhuma resposta do usuário para coletar/validar o nome.")
        reprompt_messages = REQUEST_FULL_NAME_PROMPT_TEMPLATE.format_messages()
        ai_reprompt_response = llm_client.invoke(reprompt_messages)
        return {
            "response_to_user": "Não recebi seu nome. " + ai_reprompt_response.content.strip(),
            "scheduling_step": "VALIDATING_FULL_NAME", 
            "current_operation": "SCHEDULING"
        }

    logger.info(f"Mensagem recebida do usuário para extração de nome: '{user_message_content}'")

    
    extracted_name = "NOME_NAO_IDENTIFICADO"
    try:
        extraction_prompt_messages = EXTRACT_FULL_NAME_PROMPT_TEMPLATE.format_messages(user_message=user_message_content)
        llm_extraction_response = llm_client.invoke(extraction_prompt_messages)
        extracted_name = llm_extraction_response.content.strip()
        logger.info(f"Nome extraído pelo LLM: '{extracted_name}' (da entrada: '{user_message_content}')")
    except Exception as e:
        logger.error(f"Erro ao invocar LLM para extração de nome: {e}", exc_info=True)
        return {
            "response_to_user": "Tive um problema ao entender o nome informado. Poderia, por favor, me dizer seu nome completo novamente?",
            "scheduling_step": "VALIDATING_FULL_NAME",
            "current_operation": "SCHEDULING"
        }

    if not extracted_name or extracted_name == "NOME_NAO_IDENTIFICADO":
        logger.warning(f"LLM não conseguiu extrair um nome válido da entrada: '{user_message_content}'. Retorno do LLM: '{extracted_name}'")
        reprompt_messages = REQUEST_FULL_NAME_PROMPT_TEMPLATE.format_messages()
        ai_reprompt_response = llm_client.invoke(reprompt_messages)
        return {
            "response_to_user": "Não consegui identificar um nome válido na sua resposta. " + ai_reprompt_response.content.strip(),
            "scheduling_step": "VALIDATING_FULL_NAME",
            "current_operation": "SCHEDULING"
        }

    try:
        validated_name_model = FullNameModel(full_name=extracted_name)
        final_validated_name = validated_name_model.full_name
        logger.info(f"Nome extraído e validado com sucesso: {final_validated_name}")

        prompt_messages_especialidade = REQUEST_SPECIALTY_PROMPT_TEMPLATE.format_messages(user_name=final_validated_name)
        ai_response_especialidade = llm_client.invoke(prompt_messages_especialidade)
        pergunta_especialidade = ai_response_especialidade.content.strip()
        logger.info(f"Pergunta sobre especialidade gerada após validar nome: '{pergunta_especialidade}'")
        
        return {
            "user_full_name": final_validated_name,
            "response_to_user": pergunta_especialidade,
            "scheduling_step": "VALIDATING_SPECIALTY",
            "current_operation": "SCHEDULING",
        }
    except ValueError as e: 
        error_message_for_user = str(e)
        logger.warning(f"Falha na validação Pydantic do nome extraído '{extracted_name}': {error_message_for_user}")
        
        if "O nome completo deve conter pelo menos um nome e um sobrenome" in error_message_for_user:
            friendly_error = f"O nome '{extracted_name}' não parece ser um nome completo. Por favor, informe seu nome e sobrenome."
        elif "O nome completo deve conter apenas letras e espaços" in error_message_for_user:
            friendly_error = f"O nome '{extracted_name}' contém caracteres que não são permitidos. Por favor, use apenas letras e espaços."
        else:
            friendly_error = f"Houve um problema com o nome '{extracted_name}'. Poderia tentar informar seu nome completo novamente?"

        return {
            "response_to_user": friendly_error,
            "scheduling_step": "VALIDATING_FULL_NAME", 
            "current_operation": "SCHEDULING"
        }
    except Exception as e:
        logger.error(f"Erro inesperado durante validação Pydantic do nome '{extracted_name}': {e}", exc_info=True)
        return {
            "response_to_user": "Ocorreu um erro ao processar seu nome. Poderia, por favor, informar seu nome completo novamente?",
            "scheduling_step": "VALIDATING_FULL_NAME",
            "current_operation": "SCHEDULING"
        }

def coletar_validar_especialidade_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    logger.debug("--- Nó Agendamento: coletar_validar_especialidade_node ---")
    messages = state.get("messages", [])
    last_user_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""
    user_full_name = state.get("user_full_name", "Prezado(a) cliente")

    if not last_user_message:
        logger.warning("Nenhuma mensagem do usuário para validar especialidade.")
        return {
            "response_to_user": "Por favor, informe a especialidade médica desejada.",
            "scheduling_step": "VALIDATING_SPECIALTY",
            "user_chosen_specialty": None,
            "user_chosen_specialty_id": None
        }

    api_token = settings.APPHEALTH_API_TOKEN
    api_headers = {
        "Authorization": f"{api_token}",
        "Content-Type": "application/json"
    }
    url_especialidades_todas = "https://back.homologacao.apphealth.com.br:9090/api-vizi/especialidades"
    especialidades_api_list = []
    nomes_especialidades_oficiais = []

    try:
        response = requests.get(url_especialidades_todas, headers=api_headers, timeout=10)
        response.raise_for_status()
        especialidades_api_list = response.json()

        if not especialidades_api_list or not isinstance(especialidades_api_list, list):
            logger.warning(f"API de especialidades ({url_especialidades_todas}) retornou dados vazios ou em formato inesperado.")
            return {
                "response_to_user": "Desculpe, estou com dificuldades para carregar a lista de especialidades no momento. Por favor, tente mais tarde.",
                "scheduling_step": "VALIDATING_SPECIALTY",
                "user_chosen_specialty": None,
                "user_chosen_specialty_id": None
            }

        nomes_especialidades_oficiais = [
            item["especialidade"] for item in especialidades_api_list if "especialidade" in item and isinstance(item.get("especialidade"), str)
        ]
        if not nomes_especialidades_oficiais:
            logger.warning("Não foi possível extrair nomes de especialidades válidos da resposta da API.")
            return {
                "response_to_user": "Parece que nossa lista de especialidades está temporariamente indisponível. Por favor, tente mais tarde.",
                "scheduling_step": "VALIDATING_SPECIALTY",
                "user_chosen_specialty": None,
                "user_chosen_specialty_id": None
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"Erro ao buscar todas as especialidades da API ({url_especialidades_todas}): {e}")
        return {
            "response_to_user": "Desculpe, estou com dificuldades para acessar nossas especialidades no momento. Por favor, tente novamente em alguns instantes.",
            "scheduling_step": "VALIDATING_SPECIALTY",
            "user_chosen_specialty": None,
            "user_chosen_specialty_id": None
        }

    prompt_validate_specialty_messages = VALIDATE_SPECIALTY_PROMPT_TEMPLATE.format_messages(
        user_input_specialty=last_user_message
    )
    cleaned_specialty_name = ""
    try:
        validated_specialty_response = llm_client.invoke(prompt_validate_specialty_messages)
        cleaned_specialty_name = validated_specialty_response.content.strip()
        logger.info(f"Resultado da validação/classificação da entrada de especialidade: '{cleaned_specialty_name}' para entrada '{last_user_message}'")

    except Exception as e:
        logger.error(f"Erro ao invocar LLM para validar/classificar entrada de especialidade: {e}")
        return {
            "response_to_user": "Desculpe, tive um problema ao tentar entender sua solicitação sobre a especialidade. Poderia tentar novamente?",
            "scheduling_step": "VALIDATING_SPECIALTY",
            "user_chosen_specialty": None,
            "user_chosen_specialty_id": None
        }

    if cleaned_specialty_name == "LISTAR_ESPECIALIDADES":
        logger.info(f"Usuário '{user_full_name}' solicitou a listagem de especialidades.")
        available_specialties_str = "\n".join([f"- {esp}" for esp in nomes_especialidades_oficiais[:15]]) # Mostra até 15
        response_text = (
            f"Claro, {user_full_name}! Atualmente, trabalhamos com as seguintes especialidades:\n{available_specialties_str}\n"
            "Por favor, escolha uma da lista ou, se preferir, pode digitar 'cancelar'."
        )
        return {
            "response_to_user": response_text,
            "scheduling_step": "VALIDATING_SPECIALTY",
            "user_chosen_specialty": None,
            "user_chosen_specialty_id": None
        }

    if cleaned_specialty_name == "ENTRADA_INVALIDA_NAO_EH_ESPECIALIDADE":
        logger.info(f"Entrada '{last_user_message}' não foi considerada uma especialidade válida nem uma solicitação de listagem.")
        return {
            "response_to_user": f"Desculpe, não entendi '{last_user_message}' como uma especialidade médica. Poderia tentar informar novamente? Por exemplo: Cardiologia, Ortopedia, etc. Se desejar ver a lista de especialidades, pode perguntar 'quais especialidades vocês têm?'.",
            "scheduling_step": "VALIDATING_SPECIALTY",
            "user_chosen_specialty": None,
            "user_chosen_specialty_id": None
        }

    prompt_match_specialty_messages = MATCH_OFFICIAL_SPECIALTY_PROMPT_TEMPLATE.format_messages(
        normalized_user_input_specialty=cleaned_specialty_name,
        official_specialties_list_str=", ".join(nomes_especialidades_oficiais)
    )

    try:
        match_response = llm_client.invoke(prompt_match_specialty_messages)
        nome_especialidade_llm_match = match_response.content.strip()
        logger.info(f"LLM de correspondência sugeriu: '{nome_especialidade_llm_match}' para a entrada normalizada '{cleaned_specialty_name}'")

    except Exception as e:
        logger.error(f"Erro ao invocar LLM para correspondência de especialidade: {e}")
        return {
            "response_to_user": "Desculpe, tive um problema ao tentar identificar a especialidade exata em nossa lista. Poderia tentar novamente?",
            "scheduling_step": "VALIDATING_SPECIALTY",
            "user_chosen_specialty": cleaned_specialty_name,
            "user_chosen_specialty_id": None
        }

    if nome_especialidade_llm_match != "NENHUMA_CORRESPONDENCIA" and nome_especialidade_llm_match:
        for item in especialidades_api_list:
            if item.get("especialidade") == nome_especialidade_llm_match and isinstance(item.get("id"), int):
                specialty_id_found = item["id"]
                official_specialty_name = item["especialidade"]
                logger.info(f"Sucesso! Entrada original '{last_user_message}' (normalizada para '{cleaned_specialty_name}') correspondeu a '{official_specialty_name}' (ID: {specialty_id_found}).")

                next_question_prompt = REQUEST_PROFESSIONAL_PREFERENCE_PROMPT_TEMPLATE.format_messages(
                    user_name=user_full_name,
                    user_specialty=official_specialty_name
                )
                next_question_response = llm_client.invoke(next_question_prompt)
                response_text_for_user = next_question_response.content.strip()

                return {
                    "response_to_user": response_text_for_user,
                    "scheduling_step": "CLASSIFYING_PROFESSIONAL_PREFERENCE",
                    "user_chosen_specialty": official_specialty_name,
                    "user_chosen_specialty_id": specialty_id_found,
                    "error_message": None
                }

        logger.warning(f"Especialidade '{nome_especialidade_llm_match}' sugerida pelo LLM de correspondência não foi encontrada na lista original da API.")

    logger.info(f"Nenhuma correspondência oficial encontrada para '{cleaned_specialty_name}' (entrada original: '{last_user_message}').")

    available_specialties_str_fallback = "\n".join([f"- {esp}" for esp in nomes_especialidades_oficiais[:10]])
    response_text_fallback = (
        f"Desculpe, {user_full_name}, não consegui encontrar a especialidade '{cleaned_specialty_name}' em nossa lista, ou ela não está disponível no momento.\n"
        f"Atualmente, trabalhamos com as seguintes especialidades:\n{available_specialties_str_fallback}\n"
        "Por favor, escolha uma da lista ou, se preferir, pode digitar 'cancelar'."
    )
    return {
        "response_to_user": response_text_fallback,
        "scheduling_step": "VALIDATING_SPECIALTY",
        "user_chosen_specialty": None, # Limpa a tentativa anterior se não houve match oficial
        "user_chosen_specialty_id": None,
        "error_message": f"Especialidade '{cleaned_specialty_name}' (original: '{last_user_message}') não encontrada ou mapeada."
    }

def solicitar_preferencia_profissional_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Nó para perguntar ao usuário sobre sua preferência de profissional.
    """
    logger.debug("--- Nó Agendamento: solicitar_preferencia_profissional_node ---")
    user_name = state.get("user_full_name", "Paciente")
    user_specialty = state.get("user_chosen_specialty", "a especialidade escolhida")

    prompt_messages = REQUEST_PROFESSIONAL_PREFERENCE_PROMPT_TEMPLATE.format_messages(
        user_name=user_name, 
        user_specialty=user_specialty
    )
    ai_response = llm_client.invoke(prompt_messages)
    pergunta_preferencia = ai_response.content.strip()
    logger.info(f"Pergunta sobre preferência de profissional gerada: '{pergunta_preferencia}'")

    return {
        "response_to_user": pergunta_preferencia,
        "scheduling_step": "CLASSIFYING_PROFESSIONAL_PREFERENCE",
        "current_operation": "SCHEDULING"
    }

def coletar_classificar_preferencia_profissional_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Nó para coletar a resposta do usuário sobre preferência de profissional e classificá-la usando LLM.
    """
    logger.debug("--- Nó Agendamento: coletar_classificar_preferencia_profissional_node ---")
    user_response_content = get_last_user_message_content(state["messages"])
    user_specialty = state.get("user_chosen_specialty", "a especialidade escolhida")

    if not user_response_content:
        logger.warning("Nenhuma resposta do usuário para coletar/classificar a preferência de profissional.")
        return {
            "response_to_user": "Não recebi sua preferência sobre o profissional. Poderia me dizer?",
            "scheduling_step": "CLASSIFYING_PROFESSIONAL_PREFERENCE",
            "current_operation": "SCHEDULING"
        }

    logger.info(f"Resposta do usuário sobre preferência: '{user_response_content}' para especialidade '{user_specialty}'")

    prompt_messages = CLASSIFY_PROFESSIONAL_PREFERENCE_PROMPT_TEMPLATE.format_messages(
        user_specialty=user_specialty,
        user_response=user_response_content
    )
    
    try:
        llm_classification_response_str = llm_client.invoke(prompt_messages).content.strip()
        logger.debug(f"Resposta de classificação do LLM (raw): {llm_classification_response_str}")
        
        match = re.search(r'\{.*\}', llm_classification_response_str, re.DOTALL)
        if not match:
            logger.error(f"Nenhum JSON encontrado na resposta de classificação do LLM: '{llm_classification_response_str}'")
            return {
                "response_to_user": "Desculpe, não consegui processar sua resposta sobre a preferência de profissional. Poderia tentar novamente de forma mais direta, por exemplo, 'gostaria de uma indicação' ou 'prefiro escolher o Dr. Nome Sobrenome'?",
                "scheduling_step": "CLASSIFYING_PROFESSIONAL_PREFERENCE",
                "current_operation": "SCHEDULING"
            }
            
        parsed_llm_response = json.loads(match.group(0))
        preference_type = parsed_llm_response.get("preference_type")
        extracted_name = parsed_llm_response.get("extracted_professional_name")

        logger.info(f"Preferência classificada: Tipo='{preference_type}', Nome Extraído='{extracted_name}'")

        if not preference_type or preference_type == "AMBIGUOUS_OR_NEGATIVE":
            logger.warning(f"Preferência classificada como ambígua, negativa ou tipo ausente: {preference_type}")
            reprompt_message = "Desculpe, não consegui entender sua preferência. Você gostaria que eu buscasse opções de profissionais para você, ou você prefere nomear um específico?"
            return {
                "response_to_user": reprompt_message,
                "scheduling_step": "CLASSIFYING_PROFESSIONAL_PREFERENCE",
                "current_operation": "SCHEDULING",
                "professional_preference_type": None,
                "user_provided_professional_name": None 
            }

        update_dict = {
            "professional_preference_type": preference_type,
            "user_provided_professional_name": extracted_name if preference_type == "SPECIFIC_NAME_PROVIDED" else None,
            "scheduling_step": "PROCESSING_PROFESSIONAL_LOGIC", 
            "current_operation": "SCHEDULING",
            "response_to_user": None 
        }
        
        return update_dict

    except json.JSONDecodeError as e:
        logger.error(f"Erro ao decodificar JSON da resposta de classificação do LLM: {e}. Resposta: {llm_classification_response_str}")
        return {
            "response_to_user": "Tive um problema técnico ao entender sua preferência. Poderia tentar de novo?",
            "scheduling_step": "CLASSIFYING_PROFESSIONAL_PREFERENCE",
            "current_operation": "SCHEDULING"
        }
    except Exception as e:
        logger.error(f"Erro inesperado durante classificação de preferência profissional: {e}", exc_info=True)
        return {
            "response_to_user": "Ocorreu um erro ao processar sua preferência. Vamos tentar novamente.",
            "scheduling_step": "CLASSIFYING_PROFESSIONAL_PREFERENCE",
            "current_operation": "SCHEDULING"
        }

async def processing_professional_logic_node(
    state: MainWorkflowState,
    *, 
    llm_client: ChatOpenAI
) -> dict:
    logger.debug(f"--- Nó Agendamento: processing_professional_logic_node. Estado: {state} ---")
    
    preference_type = state.get("professional_preference_type")
    user_typed_name = state.get("user_provided_professional_name") 
    user_full_name = state.get("user_full_name", "Paciente") 
    user_specialty_name = state.get("user_chosen_specialty", "a especialidade escolhida")
    user_chosen_specialty_id = state.get("user_chosen_specialty_id")

    updates_for_state = {
        "user_chosen_professional_id": None, 
        "user_chosen_professional_name": None, 
        "response_to_user": None, 
        "scheduling_step": None, 
        "current_operation": "SCHEDULING",
        "available_professionals_list": None, 
        "error_message": None 
    }

    if preference_type == "SPECIFIC_NAME_PROVIDED":
        if not user_typed_name:
            logger.error("Tipo de preferência é SPECIFIC_NAME_PROVIDED, mas user_provided_professional_name está vazio.")
            updates_for_state["response_to_user"] = "Houve um problema ao identificar o nome do profissional que você mencionou. Poderia me dizer novamente?"
            updates_for_state["scheduling_step"] = "CLASSIFYING_PROFESSIONAL_PREFERENCE" 
            return updates_for_state
        
        if not user_chosen_specialty_id: 
            logger.error(f"Não há ID de especialidade para validar o profissional '{user_typed_name}'.")
            updates_for_state["response_to_user"] = "Desculpe, preciso saber a especialidade antes de procurar por um profissional específico. Poderia me informar a especialidade primeiro?"
            updates_for_state["scheduling_step"] = "VALIDATING_SPECIALTY" 
            return updates_for_state

        logger.info(f"Usuário forneceu nome específico: '{user_typed_name}' para especialidade ID: {user_chosen_specialty_id}. Validando...")

        api_token = settings.APPHEALTH_API_TOKEN
        api_headers = {"Authorization": f"{api_token}", "Content-Type": "application/json"}
        url_prof_especialidade = f"https://back.homologacao.apphealth.com.br:9090/api-vizi/profissionais"
        params_prof = {"status": "true", "especialidadeId": user_chosen_specialty_id} 
        
        lista_profissionais_da_especialidade_api = []
        try:
            response_prof = requests.get(url_prof_especialidade, headers=api_headers, params=params_prof, timeout=10)
            response_prof.raise_for_status()
            lista_profissionais_da_especialidade_api = response_prof.json() 
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao buscar lista de profissionais da API para validar nome '{user_typed_name}': {e}")
            updates_for_state["response_to_user"] = "Desculpe, tive um problema ao consultar nossos profissionais para validar o nome. Poderia tentar novamente em instantes?"
            updates_for_state["scheduling_step"] = "CLASSIFYING_PROFESSIONAL_PREFERENCE" 
            return updates_for_state

        if not lista_profissionais_da_especialidade_api:
            logger.warning(f"Nenhum profissional encontrado na API para especialidade ID {user_chosen_specialty_id} ao tentar validar '{user_typed_name}'.") 
            updates_for_state["response_to_user"] = f"Para a especialidade {user_specialty_name}, não encontrei nenhum profissional cadastrado. Gostaria de tentar outra especialidade ou ver se digitou o nome corretamente?"
            updates_for_state["scheduling_step"] = "CLASSIFYING_PROFESSIONAL_PREFERENCE" 
            return updates_for_state

        nomes_api_para_match = [prof.get("nome") for prof in lista_profissionais_da_especialidade_api if prof.get("nome")]
        if not nomes_api_para_match:
            logger.error(f"Profissionais da especialidade {user_chosen_specialty_id} obtidos, mas sem nomes válidos na API.") 
            updates_for_state["response_to_user"] = f"Encontrei registros para {user_specialty_name}, mas estou com dificuldade para obter os nomes dos profissionais. Por favor, contate o suporte."
            updates_for_state["scheduling_step"] = "CLASSIFYING_PROFESSIONAL_PREFERENCE" 
            return updates_for_state

        cleaned_user_typed_name = user_typed_name
        cleaned_user_typed_name = re.sub(r"^(dr\.?|dra\.?)\s+", "", cleaned_user_typed_name, flags=re.IGNORECASE).strip()
        logger.info(f"Nome do usuário após limpeza de títulos: '{cleaned_user_typed_name}' (original: '{user_typed_name}')")

        matched_name_from_llm = "NENHUMA_CORRESPONDENCIA" 
        try:
            match_name_prompt_messages = MATCH_SPECIFIC_PROFESSIONAL_NAME_PROMPT_TEMPLATE.format_messages(
                user_typed_name=cleaned_user_typed_name, 
                professional_names_from_api_list_str=", ".join(nomes_api_para_match)
            )
            llm_match_response = llm_client.invoke(match_name_prompt_messages)
            matched_name_from_llm = llm_match_response.content.strip().strip('.').strip(',') 
            logger.info(f"LLM de correspondência de nome sugeriu (e foi limpo para): '{matched_name_from_llm}' para a entrada limpa '{cleaned_user_typed_name}'")

        except Exception as e:
            logger.error(f"Erro ao invocar LLM para correspondência de nome de profissional: {e}")
            updates_for_state["response_to_user"] = "Desculpe, tive um problema ao tentar validar o nome do profissional. Poderia tentar novamente?"
            updates_for_state["scheduling_step"] = "CLASSIFYING_PROFESSIONAL_PREFERENCE"
            return updates_for_state
        
        matched_professional_obj = None
        if matched_name_from_llm != "NENHUMA_CORRESPONDENCIA" and matched_name_from_llm:
            for prof_obj in lista_profissionais_da_especialidade_api:
                if prof_obj.get("nome") and prof_obj.get("nome").strip() == matched_name_from_llm:
                    matched_professional_obj = prof_obj
                    break 
        
        if matched_professional_obj and matched_professional_obj.get("id") and matched_professional_obj.get("nome"):
            official_prof_id = matched_professional_obj["id"]
            official_prof_name = matched_professional_obj["nome"]
            logger.info(f"Nome '{user_typed_name}' (limpo para '{cleaned_user_typed_name}') validado via LLM para: {official_prof_name} (ID: {official_prof_id})")
            
            updates_for_state["user_chosen_professional_id"] = official_prof_id
            updates_for_state["user_chosen_professional_name"] = official_prof_name
            updates_for_state["scheduling_step"] = "VALIDATING_TURN_PREFERENCE"

            turn_request_prompt = REQUEST_TURN_PREFERENCE_PROMPT_TEMPLATE.format_messages(
                user_name=user_full_name,
                professional_name_or_specialty_based=official_prof_name
            )
            updates_for_state["response_to_user"] = llm_client.invoke(turn_request_prompt).content.strip()
            
        else: 
            logger.warning(f"Nome '{user_typed_name}' (LLM match: '{matched_name_from_llm}') não validado ou não encontrado na API para especialidade {user_specialty_name}.")
            updates_for_state["response_to_user"] = (
                f"Desculpe, não encontrei um profissional chamado '{user_typed_name}' (ou um nome claramente correspondente como '{matched_name_from_llm}') para a especialidade {user_specialty_name} em nossa lista. "
                "Gostaria de tentar um nome diferente ou prefere que eu liste as opções disponíveis?"
            )
            updates_for_state["scheduling_step"] = "CLASSIFYING_PROFESSIONAL_PREFERENCE"
            updates_for_state["user_provided_professional_name"] = None 

    elif preference_type == "RECOMMENDATION": 
        logger.info(f"Preferência do usuário é '{preference_type}'. Próximo passo: listar profissionais disponíveis.")
        updates_for_state["scheduling_step"] = "LISTING_AVAILABLE_PROFESSIONALS"

    elif preference_type == "SPECIFIC_NAME_TO_PROVIDE_LATER": 
        logger.info("Usuário quer fornecer nome específico, mas ainda não o fez. Solicitando nome.")
        prompt_ask_name_messages = REQUEST_SPECIFIC_PROFESSIONAL_NAME_PROMPT_TEMPLATE.format_messages(
            user_name=user_full_name, 
            user_specialty=user_specialty_name
        )
        updates_for_state["response_to_user"] = llm_client.invoke(prompt_ask_name_messages).content.strip()
        updates_for_state["scheduling_step"] = "CLASSIFYING_PROFESSIONAL_PREFERENCE" 
        
    else: 
        logger.warning(f"Tipo de preferência não tratado ou inesperado em processing_professional_logic: '{preference_type}'. Solicitando esclarecimento.")
        updates_for_state["response_to_user"] = (
            f"Desculpe, não consegui determinar sua preferência de profissional claramente. "
            f"Você gostaria de nomear alguém para {user_specialty_name}, ou prefere que eu sugira opções?"
        )
        updates_for_state["scheduling_step"] = "CLASSIFYING_PROFESSIONAL_PREFERENCE" 

    logger.info(f"Atualizações do processing_professional_logic_node: {updates_for_state}")
    return updates_for_state

def solicitar_turno_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Nó para perguntar ao usuário sua preferência de turno (manhã/tarde).
    """
    logger.debug("--- Nó Agendamento: solicitar_turno_node ---")
    user_name = state.get("user_full_name", "Paciente")
    professional_name = state.get("user_chosen_professional_name")
    specialty_name = state.get("user_chosen_specialty", "a especialidade")

    professional_for_prompt = professional_name if professional_name else f"um profissional de {specialty_name}"

    prompt_messages = REQUEST_TURN_PREFERENCE_PROMPT_TEMPLATE.format_messages(
        user_name=user_name,
        professional_name_or_specialty_based=professional_for_prompt
    )
    ai_response = llm_client.invoke(prompt_messages)
    pergunta_turno = ai_response.content.strip()
    logger.info(f"Pergunta sobre turno gerada: '{pergunta_turno}'")

    return {
        "response_to_user": pergunta_turno,
        "scheduling_step": "VALIDATING_TURN_PREFERENCE", 
        "current_operation": "SCHEDULING"
    }

def list_available_professionals_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Busca profissionais disponíveis para a especialidade escolhida e os apresenta.
    """
    logger.debug("--- Nó Agendamento: list_available_professionals_node ---")
    
    API_TOKEN = "TXH3xnwI7P4Sh5dS71aRDsQzDrx1GKeW8jXd5eHDpqTOi"
    api_headers = {
        "Authorization": f"{API_TOKEN}",
        "Content-Type": "application/json"
    }

    specialty_id = state.get("user_chosen_specialty_id")
    specialty_name = state.get("user_chosen_specialty", "a especialidade escolhida")
    user_full_name = state.get("user_full_name", "Paciente")

    if not specialty_id:
        logger.error(f"ID da especialidade '{specialty_name}' não encontrado no estado para listar profissionais.")
        error_prompt = ChatPromptTemplate.from_template(
            "Você é um assistente de agendamento. Informe ao usuário {user_name} que ocorreu um problema "
            "ao tentar identificar a especialidade para buscar os profissionais e que será necessário tentar novamente a seleção da especialidade."
        )
        error_response_content = llm_client.invoke(error_prompt.format_messages(user_name=user_full_name)).content.strip()
        return {
            "response_to_user": error_response_content,
            "scheduling_step": "VALIDATING_SPECIALTY",
            "current_operation": "SCHEDULING",
            "user_chosen_specialty_id": None,
            "available_professionals_list": None
        }

    url = "https://back.homologacao.apphealth.com.br:9090/api-vizi/profissionais"
    params = {"especialidadeId": specialty_id, "status": "true"}
    logger.info(f"Consultando API de profissionais: {url} com params: {params}")

    try:
        response = requests.get(url, headers=api_headers, params=params, timeout=10)
        response.raise_for_status()
        professionals_api_data = response.json()

        if not professionals_api_data or not isinstance(professionals_api_data, list) or len(professionals_api_data) == 0:
            logger.info(f"Nenhum profissional encontrado para a especialidade {specialty_name} (ID: {specialty_id}).")
            no_professionals_prompt = ChatPromptTemplate.from_template(
                "Você é um assistente de agendamento. Informe ao usuário {user_name} que, no momento, não foram encontrados profissionais disponíveis "
                "para a especialidade {specialty_name}. Pergunte se ele gostaria de tentar outra especialidade ou nomear um profissional específico."
            )
            no_professionals_response = llm_client.invoke(
                no_professionals_prompt.format_messages(user_name=user_full_name, specialty_name=specialty_name)
            ).content.strip()
            return {
                "response_to_user": no_professionals_response,
                "scheduling_step": "REQUESTING_SPECIALTY", 
                "current_operation": "SCHEDULING",
                "available_professionals_list": []
            }

        simplified_professionals_list = []
        for prof in professionals_api_data:
            if prof.get("id") and prof.get("nome"):
                simplified_professionals_list.append({"id": prof.get("id"), "nome": prof.get("nome")})
        
        if not simplified_professionals_list:
             logger.warning(f"Profissionais encontrados para {specialty_name}, mas sem ID/Nome válidos.")
             return {
                "response_to_user": f"Desculpe, {user_full_name}, encontrei registros para {specialty_name} mas estou com dificuldade para obter os nomes. Gostaria de tentar outra especialidade?",
                "scheduling_step": "REQUESTING_SPECIALTY",
                "current_operation": "SCHEDULING",
                "available_professionals_list": []
            }

        max_to_show = 5
        professionals_to_show = simplified_professionals_list[:max_to_show]
        
        names_list_for_prompt = ""
        for i, prof_data in enumerate(professionals_to_show):
            names_list_for_prompt += f"{i+1}. {prof_data['nome']}\n"
        
        present_professionals_prompt_str = """
        Você é um assistente de agendamento.
        Para o usuário {user_name} e a especialidade {specialty_name}, foram encontrados os seguintes profissionais:
        {list_of_professional_names_str}
        Construa uma mensagem para o usuário apresentando essa lista.
        Peça para ele escolher um profissional digitando o número da opção ou o nome completo.
        Sua resposta:
        """
        present_professionals_prompt = ChatPromptTemplate.from_template(present_professionals_prompt_str)

        total_found = len(simplified_professionals_list)
        shown_count = len(professionals_to_show)
        
        # additional_info = ""
        # if total_found > shown_count:
        #     additional_info = f" (e mais {total_found - shown_count} outros)"


        presentation_message = llm_client.invoke(
            present_professionals_prompt.format_messages(
                user_name=user_full_name,
                specialty_name=specialty_name,
                list_of_professional_names_str=names_list_for_prompt.strip()
            )
        ).content.strip()

        return {
            "response_to_user": presentation_message,
            "scheduling_step": "VALIDATING_CHOSEN_PROFESSIONAL_FROM_LIST",
            "current_operation": "SCHEDULING",
            "available_professionals_list": simplified_professionals_list # Salva a lista completa com IDs
        }

    except requests.exceptions.HTTPError as e:
        logger.error(f"Erro HTTP ao listar profissionais: {e.response.status_code} - {e.response.text if e.response else 'Sem corpo'}")
        # ... (tratamento de erro similar ao de datas, direcionando para tentar especialidade novamente) ...
        return {
            "response_to_user": f"Desculpe, {user_full_name}, houve um problema técnico (erro {e.response.status_code if e.response else ''}) ao buscar os profissionais. Poderia tentar outra especialidade?",
            "scheduling_step": "REQUESTING_SPECIALTY",
            "current_operation": "SCHEDULING"
            }
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro de requisição ao listar profissionais: {e}")
        # ... (tratamento de erro similar ao de datas) ...
        return {
            "response_to_user": f"Desculpe, {user_full_name}, não consegui conectar ao sistema para buscar profissionais. Poderia tentar novamente ou escolher outra especialidade?",
            "scheduling_step": "REQUESTING_SPECIALTY",
            "current_operation": "SCHEDULING"
            }
    except ValueError as json_err: # Erro ao decodificar JSON
        logger.error(f"Erro ao decodificar JSON da API de profissionais: {json_err}")
        return {
            "response_to_user": f"Desculpe, {user_full_name}, recebi uma resposta inesperada do sistema ao buscar os profissionais. Poderia tentar outra especialidade?",
            "scheduling_step": "REQUESTING_SPECIALTY",
            "current_operation": "SCHEDULING"
        }

def collect_validate_chosen_professional_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Coleta a escolha do usuário da lista de profissionais apresentada e a valida.
    """
    logger.debug("--- Nó Agendamento: collect_validate_chosen_professional_node ---")
    user_response_content = get_last_user_message_content(state["messages"]).strip()
    
    professionals_shown_list = state.get("available_professionals_list", []) 
    user_full_name = state.get("user_full_name", "Paciente")

    if not user_response_content:
        logger.warning("Nenhuma resposta do usuário para validar escolha do profissional.")
        reprompt_llm = ChatPromptTemplate.from_template(
            "Você é um assistente de agendamento. O usuário não respondeu qual profissional da lista ele gostaria de escolher. "
            "Relembre-o das opções (sem listar novamente, apenas diga para escolher da lista anterior) e peça para fornecer o nome ou número."
        ).invoke({}).content.strip()
        return {
            "response_to_user": reprompt_llm,
            "scheduling_step": "VALIDATING_CHOSEN_PROFESSIONAL_FROM_LIST",
            "current_operation": "SCHEDULING"
        }

    if not professionals_shown_list:
        logger.error("Lista de profissionais 'available_professionals_list' não encontrada no estado.")
        return {
            "response_to_user": f"Desculpe, {user_full_name}, ocorreu um problema e não consigo ver a lista de profissionais que apresentei. Vamos tentar listar novamente.",
            "scheduling_step": "LISTING_AVAILABLE_PROFESSIONALS",
            "current_operation": "SCHEDULING"
        }

    chosen_prof_id = None
    chosen_prof_name = None

    try:
        choice_num = int(user_response_content)
        if 1 <= choice_num <= len(professionals_shown_list):
            selected = professionals_shown_list[choice_num - 1]
            chosen_prof_id = selected.get("id")
            chosen_prof_name = selected.get("nome")
    except ValueError:
        cleaned_user_response = user_response_content.strip()
        professional_names_from_api_list_str = ", ".join(
            [p.get("nome", "") for p in professionals_shown_list if p.get("nome")]
        )

        if professional_names_from_api_list_str: 
            try:
                match_name_prompt_messages = MATCH_SPECIFIC_PROFESSIONAL_NAME_PROMPT_TEMPLATE.format_messages(
                    user_typed_name=cleaned_user_response,
                    professional_names_from_api_list_str=professional_names_from_api_list_str
                )
                llm_match_response = llm_client.invoke(match_name_prompt_messages)
                matched_name_from_llm = llm_match_response.content.strip().strip('.').strip(',')
                logger.info(f"LLM de correspondência de nome em collect_validate_chosen_professional_node sugeriu: '{matched_name_from_llm}' para a entrada '{cleaned_user_response}'")

                if matched_name_from_llm != "NENHUMA_CORRESPONDENCIA" and matched_name_from_llm:
                    for prof_data in professionals_shown_list:
                        if prof_data.get("nome") and prof_data["nome"] == matched_name_from_llm:
                            chosen_prof_id = prof_data.get("id")
                            chosen_prof_name = prof_data.get("nome")
                            logger.info(f"Profissional '{chosen_prof_name}' (ID: {chosen_prof_id}) selecionado via LLM match.")
                            break 
            except Exception as e:
                logger.error(f"Erro ao invocar LLM para correspondência de nome em collect_validate_chosen_professional_node: {e}", exc_info=True)

        if not (chosen_prof_id and chosen_prof_name):
            potential_matches = []
            for prof_data in professionals_shown_list:
                if prof_data.get("nome") and cleaned_user_response.lower() in prof_data["nome"].lower():
                    potential_matches.append(prof_data)
            
            if len(potential_matches) == 1:
                chosen_prof_id = potential_matches[0].get("id")
                chosen_prof_name = potential_matches[0].get("nome")
                logger.info(f"Profissional '{chosen_prof_name}' (ID: {chosen_prof_id}) selecionado via substring match único.")
            elif len(potential_matches) > 1:
                logger.warning(f"Escolha ambígua de profissional por substring: '{cleaned_user_response}'. Candidatos: {[p.get('nome') for p in potential_matches]}")
                ambiguous_prompt_str = """
                Você é um assistente de agendamento. O usuário {user_name} forneceu uma entrada ('{user_input}')
                que corresponde a mais de um profissional na lista.
                Liste os profissionais que correspondem (até 3 para brevidade) e peça para ele ser mais específico ou usar o número da opção.
                Profissionais correspondentes: {matched_names_list_str}
                Sua resposta:
                """
                matched_names_for_prompt = ", ".join([p.get('nome') for p in potential_matches[:3]]) + ("..." if len(potential_matches) > 3 else "")
                ambiguous_prompt = ChatPromptTemplate.from_template(ambiguous_prompt_str)
                ambiguous_response = llm_client.invoke(
                    ambiguous_prompt.format_messages(
                        user_name=user_full_name, 
                        user_input=cleaned_user_response,
                        matched_names_list_str=matched_names_for_prompt
                    )
                ).content.strip()
                return {
                    "response_to_user": ambiguous_response,
                    "scheduling_step": "VALIDATING_CHOSEN_PROFESSIONAL_FROM_LIST"
                }

    if chosen_prof_id and chosen_prof_name:
        logger.info(f"Usuário escolheu o profissional: ID={chosen_prof_id}, Nome='{chosen_prof_name}'")
        
        next_question_prompt = REQUEST_TURN_PREFERENCE_PROMPT_TEMPLATE.format_messages(
            user_name=user_full_name,
            professional_name_or_specialty_based=chosen_prof_name
        )
        next_question_response = llm_client.invoke(next_question_prompt)
        response_text_for_user = next_question_response.content.strip()
        
        return_state = {
            "user_chosen_professional_id": chosen_prof_id,
            "user_chosen_professional_name": chosen_prof_name,
            "response_to_user": response_text_for_user, 
            "scheduling_step": "VALIDATING_TURN_PREFERENCE", 
            "current_operation": "SCHEDULING",
            "available_professionals_list": None 
        }
        logger.debug(f"Retornando de collect_validate_chosen_professional_node (SUCESSO): {return_state}")
        return return_state
    else:
        logger.warning(f"Escolha do profissional '{user_response_content}' não validada (nem por número, nem LLM, nem substring).")
        invalid_choice_prompt_str = """
        Você é um assistente de agendamento. O usuário {user_name} fez uma escolha de profissional ('{user_input}') que não corresponde
        a nenhuma das opções apresentadas anteriormente ou não pôde ser claramente identificada.
        Relembre-o que ele precisa escolher um profissional da lista que foi apresentada.
        Peça para ele tentar novamente informando o nome completo do profissional ou o número da opção da lista.
        A lista original era: {original_list_str}
        Sua resposta:
        """
        original_list_for_reprompt = ", ".join([p['nome'] for p in professionals_shown_list[:3]]) + ("..." if len(professionals_shown_list) > 3 else "")

        invalid_choice_prompt = ChatPromptTemplate.from_template(invalid_choice_prompt_str)
        invalid_response = llm_client.invoke(
            invalid_choice_prompt.format_messages(
                user_name=user_full_name, 
                user_input=user_response_content,
                original_list_str=original_list_for_reprompt
                )
        ).content.strip()
        return_state_failure = {
            "response_to_user": invalid_response, # Certifique-se que invalid_response é só a mensagem de erro correta
            "scheduling_step": "VALIDATING_CHOSEN_PROFESSIONAL_FROM_LIST",
            "current_operation": "SCHEDULING"
        }
        logger.debug(f"Retornando de collect_validate_chosen_professional_node (FALHA VALIDAÇÃO): {return_state_failure}")
        return return_state_failure

def collect_validate_chosen_date_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    logger.debug("--- Nó Agendamento: collect_validate_chosen_date_node ---")
    user_response_content = get_last_user_message_content(state["messages"])
    
    available_dates_api_format = state.get("available_dates_presented", [])
    user_full_name = state.get("user_full_name", "Paciente")

    if not user_response_content:
        logger.warning("Nenhuma resposta do usuário para validar data escolhida.")
        return {
            "response_to_user": "Por favor, escolha uma das datas apresentadas.",
            "scheduling_step": "VALIDATING_CHOSEN_DATE",
            "current_operation": "SCHEDULING"
        }

    if not available_dates_api_format:
        logger.error("'available_dates_presented' não encontrado no estado. Não é possível validar a escolha.")
        return {
            "response_to_user": f"Desculpe, {user_full_name}, ocorreu um problema e não consigo ver as opções de data que apresentei. Vamos tentar buscar as datas novamente.",
            "scheduling_step": "FETCHING_AVAILABLE_DATES",
            "current_operation": "SCHEDULING"
        }

    date_options_display_list = []
    for i, date_str_api in enumerate(available_dates_api_format):
        try:
            dt_obj = datetime.strptime(date_str_api, "%Y-%m-%d")
            date_options_display_list.append(f"{i+1}. {dt_obj.strftime('%d/%m/%Y')}")
        except ValueError:
            logger.warning(f"Data inválida em available_dates_presented: {date_str_api}")

    date_options_display_list_str = "\n".join(date_options_display_list)
    date_options_internal_list_str = ", ".join(available_dates_api_format)

    logger.info(f"Validando escolha de data: '{user_response_content}' contra opções (API format): {available_dates_api_format}")

    chosen_date_api_format = "NENHUMA_CORRESPONDENCIA_OU_AMBIGUA" 
    try:
        prompt_messages = VALIDATE_CHOSEN_DATE_PROMPT_TEMPLATE.format_messages(
            date_options_display_list_str=date_options_display_list_str,
            user_response=user_response_content,
            date_options_internal_list_str=date_options_internal_list_str
        )
        llm_response = llm_client.invoke(prompt_messages).content.strip()
        logger.info(f"LLM para validação de data retornou: '{llm_response}'")
        
        if llm_response in available_dates_api_format:
            chosen_date_api_format = llm_response
        else:
            logger.warning(f"Resposta do LLM '{llm_response}' não é uma das datas válidas apresentadas: {available_dates_api_format}")
            chosen_date_api_format = "NENHUMA_CORRESPONDENCIA_OU_AMBIGUA"


    except Exception as e:
        logger.error(f"Erro ao invocar LLM para validar data escolhida: {e}", exc_info=True)

    if chosen_date_api_format != "NENHUMA_CORRESPONDENCIA_OU_AMBIGUA":
        logger.info(f"Usuário escolheu a data: {chosen_date_api_format}")
        
        return {
            "user_chosen_date": chosen_date_api_format,
            "response_to_user": None, 
            "scheduling_step": "FETCHING_AVAILABLE_TIMES", 
            "current_operation": "SCHEDULING",
            "available_dates_presented": None
        }
    else:
        logger.warning(f"Escolha da data '{user_response_content}' não pôde ser validada ou foi ambígua.")
        reprompt_text = (
            f"Desculpe, {user_full_name}, não consegui identificar qual data você escolheu a partir de '{user_response_content}'.\n"
            f"As opções eram:\n{date_options_display_list_str}\n"
            "Poderia, por favor, informar o número da opção ou a data completa (dd/mm/aaaa)?"
        )
        return {
            "response_to_user": reprompt_text,
            "scheduling_step": "VALIDATING_CHOSEN_DATE",
            "current_operation": "SCHEDULING"
        }

def fetch_and_present_available_times_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    logger.info(f"--- Nó: fetch_and_present_available_times_node (Session ID: {state.get('session_id', 'N/A')}) ---")
    user_full_name = state.get("user_full_name", "Usuário")
    professional_id = state.get("user_chosen_professional_id")
    chosen_date_str = state.get("user_chosen_date") 
    chosen_turn = state.get("user_chosen_turn")
    professional_name = state.get("user_chosen_professional_name", "o profissional escolhido")

    if not all([professional_id, chosen_date_str, chosen_turn]):
        logger.error(f"Dados insuficientes para buscar horários: professional_id={professional_id}, chosen_date={chosen_date_str}, chosen_turn={chosen_turn}")
        return {
            "response_to_user": "Desculpe, não consegui obter todas as informações necessárias (profissional, data ou turno) para buscar os horários. Poderia tentar novamente?",
            "scheduling_step": "SCHEDULING_ERROR" 
        }

    api_url = f"https://back.homologacao.apphealth.com.br:9090/api-vizi/agenda/profissionais/{professional_id}/horarios"
    headers = {"Authorization": settings.APPHEALTH_API_TOKEN} 
    params = {"data": chosen_date_str}

    response_to_user = ""
    next_scheduling_step = "AWAITING_TIME_CHOICE"
    available_times_presented_for_state: List[Dict[str, str]] = []


    try:
        logger.info(f"Chamando API de horários: GET {api_url} com params: {params}")
        api_response = requests.get(api_url, headers=headers, params=params, timeout=10)
        api_response.raise_for_status()
        available_slots_from_api = api_response.json() 
        logger.info(f"API de horários retornou {len(available_slots_from_api)} slots.")

        if not available_slots_from_api:
            response_to_user = f"Desculpe, {user_full_name}, parece que não há horários disponíveis para {professional_name} no dia {datetime.strptime(chosen_date_str, '%Y-%m-%d').strftime('%d/%m/%Y')} ({chosen_turn.lower()}). Gostaria de tentar outra data ou turno?"
            next_scheduling_step = "FETCHING_AVAILABLE_DATES"
        else:
            filtered_slots_details = []
            for slot in available_slots_from_api:
                if slot.get("horaInicio") and slot.get("horaFim"):
                    try:
                        hora_inicio_obj = datetime.strptime(slot["horaInicio"], "%H:%M:%S").time()
                        
                        if (chosen_turn == "MANHA" and hora_inicio_obj < time(12, 0)) or \
                           (chosen_turn == "TARDE" and hora_inicio_obj >= time(12, 0)):
                            
                            filtered_slots_details.append({
                                "display": slot["horaInicio"][:5],      
                                "horaInicio_api": slot["horaInicio"],   
                                "horaFim_api": slot["horaFim"]
                            })
                    except ValueError as ve:
                        logger.warning(f"Slot de horário com formato de hora inválido da API: {slot}. Erro: {ve}")
                        continue
            
            unique_filtered_slots_temp = {}
            for slot_detail in filtered_slots_details:
                if slot_detail["display"] not in unique_filtered_slots_temp:
                    unique_filtered_slots_temp[slot_detail["display"]] = slot_detail
            
            sorted_unique_slots = sorted(
                list(unique_filtered_slots_temp.values()), 
                key=lambda x: datetime.strptime(x["display"], "%H:%M").time()
            )

            if not sorted_unique_slots:
                 response_to_user = f"Desculpe, {user_full_name}, após filtrar pelo turno da {chosen_turn.lower()}, não encontramos horários disponíveis para {professional_name} no dia {datetime.strptime(chosen_date_str, '%Y-%m-%d').strftime('%d/%m/%Y')}. Gostaria de tentar outro turno ou data?"
                 next_scheduling_step = "VALIDATING_TURN_PREFERENCE" 
            else:
                available_times_presented_for_state = sorted_unique_slots[:3]
                
                if not available_times_presented_for_state:
                    response_to_user = f"Desculpe, {user_full_name}, não consegui encontrar horários para o turno da {chosen_turn.lower()} no dia {datetime.strptime(chosen_date_str, '%Y-%m-%d').strftime('%d/%m/%Y')} para {professional_name}. Gostaria de tentar outra data ou turno?"
                    next_scheduling_step = "VALIDATING_TURN_PREFERENCE"
                else:
                    times_list_for_prompt_display = [
                        f"{i+1}. {slot_data['display']}" for i, slot_data in enumerate(available_times_presented_for_state)
                    ]
                    times_list_str = "\n".join(times_list_for_prompt_display)
                    
                    prompt_messages = PRESENT_AVAILABLE_TIMES_PROMPT_TEMPLATE.format_messages(
                        user_name=user_full_name,
                        chosen_date=datetime.strptime(chosen_date_str, "%Y-%m-%d").strftime("%d/%m/%Y"),
                        professional_name=professional_name,
                        chosen_turn=chosen_turn.lower(),
                        available_times_list_str=times_list_str
                    )
                    if prompt_messages and isinstance(prompt_messages[-1], AIMessage):
                         response_to_user = prompt_messages[-1].content
                    else:
                        logger.error("Falha ao formatar a mensagem com PRESENT_AVAILABLE_TIMES_PROMPT_TEMPLATE.")
                        response_to_user = f"Temos estes horários para {professional_name} no dia {datetime.strptime(chosen_date_str, '%Y-%m-%d').strftime('%d/%m/%Y')} ({chosen_turn.lower()}):\n{times_list_str}\nQual você prefere?"

    except requests.exceptions.RequestException as e:
        logger.error(f"Erro ao chamar API de horários: {e}")
        response_to_user = "Desculpe, tivemos um problema ao buscar os horários disponíveis. Por favor, tente novamente mais tarde."
        next_scheduling_step = "SCHEDULING_ERROR"
    except Exception as e:
        logger.error(f"Erro inesperado em fetch_and_present_available_times_node: {e}", exc_info=True)
        response_to_user = "Desculpe, ocorreu um erro interno ao processar sua solicitação de horários."
        next_scheduling_step = "SCHEDULING_ERROR"

    logger.info(f"fetch_and_present_available_times_node: response_to_user='{response_to_user}', next_scheduling_step='{next_scheduling_step}', presented_times='{available_times_presented_for_state}'")
    return {
        "response_to_user": response_to_user,
        "scheduling_step": next_scheduling_step,
        "available_times_presented": available_times_presented_for_state, 
        "messages": AIMessage(content=response_to_user) 
    }

def coletar_validar_horario_escolhido_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    logger.info(f"--- Nó: coletar_validar_horario_escolhido_node (Session ID: {state.get('session_id', 'N/A')}) ---")
    user_response_content = get_last_user_message_content(state["messages"])
    
    available_times_details_list = state.get("available_times_presented", []) 
    user_full_name = state.get("user_full_name", "Paciente")
    chosen_specialty = state.get("user_chosen_specialty", "a especialidade")
    chosen_professional_name = state.get("user_chosen_professional_name", "o profissional")
    chosen_date_api_format = state.get("user_chosen_date") 

    if not user_response_content:
        logger.warning("Nenhuma resposta do usuário para validar horário escolhido.")
        return {
            "response_to_user": "Por favor, escolha um dos horários que apresentei.",
            "scheduling_step": "AWAITING_TIME_CHOICE",
            "messages": AIMessage(content="Por favor, escolha um dos horários que apresentei.")
        }

    if not available_times_details_list:
        logger.error("'available_times_presented' não encontrado ou vazio no estado. Não é possível validar a escolha do horário.")
        return {
            "response_to_user": "Desculpe, ocorreu um problema e não consigo ver os horários que apresentei. Vamos tentar buscar novamente.",
            "scheduling_step": "FETCHING_AVAILABLE_TIMES", 
            "messages": AIMessage(content="Desculpe, ocorreu um problema e não consigo ver os horários que apresentei. Vamos tentar buscar novamente.")
        }

    time_options_display_list_str = "\n".join([
        f"{i+1}. {slot_details['display']}" for i, slot_details in enumerate(available_times_details_list)
    ])
    time_options_internal_list_str = ", ".join([
        slot_details['display'] for slot_details in available_times_details_list
    ])

    logger.info(f"Validando escolha de horário: '{user_response_content}' contra opções de display: {[s['display'] for s in available_times_details_list]}")

    chosen_time_display_from_llm = "NENHUMA_CORRESPONDENCIA_OU_AMBIGUA"
    try:
        prompt_messages = VALIDATE_CHOSEN_TIME_PROMPT_TEMPLATE.format_messages(
            time_options_display_list_str=time_options_display_list_str,
            user_response=user_response_content,
            time_options_internal_list_str=time_options_internal_list_str
        )
        llm_response = llm_client.invoke(prompt_messages)
        chosen_time_display_from_llm = llm_response.content.strip() 
        logger.info(f"LLM para validação de horário (display HH:MM) retornou: '{chosen_time_display_from_llm}'")
    except Exception as e:
        logger.error(f"Erro ao invocar LLM para validar horário escolhido: {e}", exc_info=True)
    
    selected_slot_details: Optional[Dict[str, str]] = None
    if chosen_time_display_from_llm != "NENHUMA_CORRESPONDENCIA_OU_AMBIGUA":
        for slot_details in available_times_details_list:
            if slot_details['display'] == chosen_time_display_from_llm:
                selected_slot_details = slot_details
                break
    
    if selected_slot_details:
        user_chosen_time_hhmm = selected_slot_details['display'] 
        user_chosen_time_fim_api = selected_slot_details['horaFim_api']
        
        logger.info(f"Usuário escolheu o horário (display): {user_chosen_time_hhmm}. Detalhes completos do slot: {selected_slot_details}")
        
        chosen_date_display_for_confirmation = ""
        if chosen_date_api_format:
            try:
                chosen_date_display_for_confirmation = datetime.strptime(chosen_date_api_format, "%Y-%m-%d").strftime("%d/%m/%Y")
            except ValueError:
                chosen_date_display_for_confirmation = chosen_date_api_format

        try:
            confirmation_prompt_messages = FINAL_SCHEDULING_CONFIRMATION_PROMPT_TEMPLATE.format_messages(
                user_name=user_full_name,
                chosen_specialty=chosen_specialty,
                chosen_professional_name=chosen_professional_name,
                chosen_date_display=chosen_date_display_for_confirmation,
                chosen_time=user_chosen_time_hhmm
            )
            confirmation_response = llm_client.invoke(confirmation_prompt_messages)
            response_text_for_user = confirmation_response.content.strip()
        except Exception as e:
            logger.error(f"Erro ao gerar mensagem de confirmação final: {e}")
            response_text_for_user = f"Agendamento para {chosen_specialty} com {chosen_professional_name} no dia {chosen_date_display_for_confirmation} às {user_chosen_time_hhmm}. Confirmar?"

        return {
            "user_chosen_time": user_chosen_time_hhmm, 
            "user_chosen_time_fim": user_chosen_time_fim_api,
            "response_to_user": response_text_for_user,
            "scheduling_step": "AWAITING_FINAL_CONFIRMATION", 
            "current_operation": "SCHEDULING", 
            "messages": AIMessage(content=response_text_for_user),
            "available_times_presented": None
        }
    else:
        logger.warning(f"Escolha do horário '{user_response_content}' (LLM: '{chosen_time_display_from_llm}') não pôde ser validada ou não encontrada nos detalhes dos slots.")
        reprompt_text = (
            f"Desculpe, {user_full_name}, não consegui identificar qual horário você escolheu a partir de '{user_response_content}'.\n"
            f"As opções eram:\n{time_options_display_list_str}\n"
            "Poderia, por favor, informar o número da opção ou o horário (ex: 07:30)?"
        )
        return {
            "response_to_user": reprompt_text,
            "scheduling_step": "AWAITING_TIME_CHOICE", 
            "messages": AIMessage(content=reprompt_text)
        }

def process_retry_option_choice_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
        """
        Processa a escolha do usuário após ser informado sobre a indisponibilidade
        e perguntado se deseja tentar outro profissional, especialidade, ou verificar mais tarde.
        """
        logger.debug("--- Nó Agendamento: process_retry_option_choice_node ---")
        user_message_content = get_last_user_message_content(state["messages"])
        user_full_name = state.get("user_full_name", "Cliente")
        updates = {}
    
        if not user_message_content:
            logger.warning("Nenhuma resposta do usuário para processar a opção de nova tentativa.")
            updates["response_to_user"] = "Não entendi sua resposta. Gostaria de tentar com outro profissional, outra especialidade, ou verificar mais tarde?"
            updates["scheduling_step"] = "AWAITING_RETRY_OPTION_AFTER_NO_AVAILABILITY" # Mantém o passo
            updates["messages"] = AIMessage(content=updates["response_to_user"])
            return updates
    
        normalized_response = user_message_content.lower()
        
        if "outro profissional" in normalized_response or "outro médico" in normalized_response or "outro dr" in normalized_response:
            logger.info(f"Usuário '{user_full_name}' optou por tentar outro profissional na mesma especialidade.")
            updates["user_chosen_professional_id"] = None
            updates["user_chosen_professional_name"] = None
            updates["user_provided_professional_name"] = None 
            updates["professional_preference_type"] = None 
            updates["user_chosen_turn"] = None
            updates["available_dates_presented"] = None
            updates["user_chosen_date"] = None
            updates["available_times_presented"] = None
            updates["scheduling_step"] = "REQUESTING_PROFESSIONAL_PREFERENCE"
            updates["response_to_user"] = None
    
        elif "outra especialidade" in normalized_response or "mudar especialidade" in normalized_response:
            logger.info(f"Usuário '{user_full_name}' optou por tentar outra especialidade.")
            updates["user_chosen_specialty"] = None
            updates["user_chosen_specialty_id"] = None
            updates["user_chosen_professional_id"] = None
            updates["user_chosen_professional_name"] = None
            updates["user_provided_professional_name"] = None
            updates["professional_preference_type"] = None
            updates["user_chosen_turn"] = None
            updates["available_dates_presented"] = None
            updates["user_chosen_date"] = None
            updates["available_times_presented"] = None
            updates["scheduling_step"] = "VALIDATING_SPECIALTY" 
            updates["response_to_user"] = None 
    
        elif "não" in normalized_response or "cancelar" in normalized_response or "mais tarde" in normalized_response or "nenhum" in normalized_response:
            logger.info(f"Usuário '{user_full_name}' optou por não prosseguir ou verificar mais tarde.")
            updates["response_to_user"] = f"Entendido, {user_full_name}. Se mudar de ideia ou precisar de algo mais, é só chamar!"
            updates["scheduling_step"] = None 
            updates["current_operation"] = None
            updates["messages"] = AIMessage(content=updates["response_to_user"])
        else:
            logger.warning(f"Resposta não clara do usuário ('{user_message_content}') para opção de nova tentativa.")
            updates["response_to_user"] = (
                f"Desculpe, {user_full_name}, não entendi bem. Se não há datas para este profissional, "
                "você gostaria de tentar com 'outro profissional' (na mesma especialidade), "
                "tentar 'outra especialidade', ou 'cancelar' o agendamento por ora?"
            )
            updates["scheduling_step"] = "AWAITING_RETRY_OPTION_AFTER_NO_AVAILABILITY" 
            updates["messages"] = AIMessage(content=updates["response_to_user"])
    
        return updates

def check_cancellation_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Verifica se a última mensagem do usuário indica uma intenção de cancelar o agendamento.
    """
    logger.debug("--- Nó: check_cancellation_node ---")
    user_message_content = get_last_user_message_content(state["messages"])
    current_op = state.get("current_operation")

    if not user_message_content or current_op != "SCHEDULING":
        logger.debug("Nenhuma mensagem de usuário para checar cancelamento ou não está em agendamento. Prosseguindo.")
        return {"cancellation_check_result": "PROCEED"}

    try:
        prompt_messages = CHECK_CANCELLATION_PROMPT_TEMPLATE.format_messages(user_message=user_message_content)
        llm_response = llm_client.invoke(prompt_messages)
        cancellation_intent = llm_response.content.strip().upper()
        logger.info(f"Verificação de cancelamento para '{user_message_content}': LLM respondeu '{cancellation_intent}'")

        if cancellation_intent == "SIM":
            logger.info(f"Intenção de cancelamento detectada para o usuário {state.get('user_full_name', '')}.")
            user_phone_from_state = state.get("user_phone")

            if user_phone_from_state:
                n8n_webhook_url_cancel = f"https://n8n-server.apphealth.com.br/webhook/remove-tag?phone={user_phone_from_state}"
                logger.info(f"Tentando chamar webhook N8N para remover tag (cancelamento): GET {n8n_webhook_url_cancel}")
                try:
                    response = requests.get(n8n_webhook_url_cancel, timeout=10)
                    response.raise_for_status()
                    logger.info(f"Webhook N8N para remover tag (cancelamento) retornou status {response.status_code}")
                except Exception as e:
                    logger.error(f"Erro ao chamar webhook N8N para remover tag (cancelamento): {e}")
            
            return {
                "response_to_user": "Entendido. O processo de agendamento foi cancelado. Se precisar de algo mais, é só chamar!",
                "current_operation": None,
                "scheduling_step": None,
                "user_full_name": state.get("user_full_name"),
                "user_chosen_specialty": None,
                "user_chosen_specialty_id": None,
                "professional_preference_type": None,
                "user_provided_professional_name": None,
                "user_chosen_professional_id": None,
                "user_chosen_professional_name": None,
                "available_professionals_list": None,
                "user_chosen_turn": None,
                "available_dates_presented": None,
                "user_chosen_date": None,
                "available_times_presented": None,
                "user_chosen_time": None,
                "user_chosen_time_fim": None,
                "scheduling_completed": False,
                "scheduling_values_confirmed": None,
                "cancellation_check_result": "CANCELLED"
            }
        else:
            logger.debug("Nenhuma intenção de cancelamento detectada. Prosseguindo com o fluxo normal.")
            return {"cancellation_check_result": "PROCEED"}

    except Exception as e:
        logger.error(f"Erro ao verificar intenção de cancelamento com LLM: {e}", exc_info=True)
        return {"cancellation_check_result": "PROCEED_ λόγω_ERRO"}

def process_fallback_choice_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Processa a escolha do usuário após uma situação de fallback no agendamento, usando LLM para interpretação.
    """
    logger.debug(f"--- Nó: process_fallback_choice_node (LLM). Estado: {state} ---")
    user_message_content = get_last_user_message_content(state.get("messages", []))
    user_full_name = state.get("user_full_name", "Prezado(a) cliente")
    previous_step = state.get("previous_scheduling_step")
    
    # Obter o texto da mensagem que o placeholder_fallback_node enviou (com as opções)
    # Assumimos que a penúltima mensagem no estado é a AIMessage do placeholder_fallback_node
    fallback_prompt_sent_to_user = ""
    messages_history = state.get("messages", [])
    if len(messages_history) >= 2 and isinstance(messages_history[-2], AIMessage):
        fallback_prompt_sent_to_user = messages_history[-2].content
    else:
        logger.warning("Não foi possível recuperar o prompt de fallback original enviado ao usuário.")
        # Como fallback, podemos tentar reconstruir uma descrição genérica, mas é menos ideal
        fallback_prompt_sent_to_user = "O sistema apresentou opções para continuar após um erro."

    updates = {
        "response_to_user": None,
        "scheduling_step": None,
        "current_operation": "SCHEDULING",
        "previous_scheduling_step": None 
    }

    if not user_message_content:
        logger.warning("Nenhuma resposta do usuário para processar escolha de fallback.")
        # Reutiliza a mensagem do placeholder_fallback_node se não houver resposta,
        # mas isso requer que o placeholder_fallback_node seja chamado novamente.
        # Por ora, uma mensagem genérica:
        updates["response_to_user"] = "Não recebi sua escolha. Por favor, me diga o que gostaria de fazer em relação às opções apresentadas."
        updates["scheduling_step"] = "AWAITING_FALLBACK_CHOICE"
        return updates

    # Determinar a descrição da etapa anterior e as opções textuais para o prompt do LLM
    step_map = {
        "VALIDATING_FULL_NAME": "confirmação do seu nome",
        "VALIDATING_SPECIALTY": "escolha da especialidade",
        "CLASSIFYING_PROFESSIONAL_PREFERENCE": "preferência de profissional",
        "PROCESSING_PROFESSIONAL_LOGIC": "validação do profissional",
        "LISTING_AVAILABLE_PROFESSIONALS": "busca por profissionais",
        "VALIDATING_CHOSEN_PROFESSIONAL_FROM_LIST": "escolha do profissional da lista",
        "VALIDATING_TURN_PREFERENCE": "escolha do turno",
        "FETCHING_AVAILABLE_DATES": "busca por datas disponíveis",
        "VALIDATING_CHOSEN_DATE": "confirmação da data",
        "FETCHING_AVAILABLE_TIMES": "busca por horários",
        "AWAITING_TIME_CHOICE": "escolha do horário",
        "AWAITING_FINAL_CONFIRMATION": "confirmação final do agendamento"
    }
    previous_step_description = step_map.get(previous_step, "etapa anterior") if previous_step else "etapa anterior"

    optional_go_to_specialty_text = ""
    cancel_option_text_for_llm = ""
    num_options_presented = 1 # Começa com "tentar novamente"
    
    if previous_step and previous_step not in ["VALIDATING_SPECIALTY", "VALIDATING_FULL_NAME"]:
        num_options_presented +=1
        optional_go_to_specialty_text = f"{num_options_presented}. Voltar para a escolha da especialidade."
    
    num_options_presented +=1
    cancel_option_text_for_llm = f"{num_options_presented}. Cancelar o agendamento."


    llm_classification = "AMBIGUOUS_OR_UNAFFILIATED" # Default
    try:
        prompt_llm = VALIDATE_FALLBACK_CHOICE_PROMPT_TEMPLATE.format_messages(
            user_name=user_full_name,
            fallback_prompt_text_with_options=fallback_prompt_sent_to_user,
            user_response=user_message_content,
            previous_step_description=previous_step_description,
            optional_go_to_specialty_option_text=optional_go_to_specialty_text,
            cancel_option_text=cancel_option_text_for_llm
        )
        logger.debug(f"Prompt para LLM de validação de fallback: {prompt_llm}")
        llm_response_obj = llm_client.invoke(prompt_llm)
        llm_classification = llm_response_obj.content.strip().upper()
        logger.info(f"LLM classificou a escolha de fallback como: '{llm_classification}' para a entrada '{user_message_content}'")
    except Exception as e:
        logger.error(f"Erro ao invocar LLM para classificar escolha de fallback: {e}", exc_info=True)
        # Em caso de erro do LLM, pode-se ter uma lógica de repetição simples ou ir para um estado de erro mais genérico.
        # Por ora, manterá AMBIGUOUS_OR_UNAFFILIATED.
    
    if llm_classification == "RETRY_PREVIOUS_STEP" and previous_step:
        logger.info(f"Usuário escolheu (via LLM) tentar novamente a etapa: '{previous_step}'.")
        updates["scheduling_step"] = previous_step
    elif llm_classification == "GO_TO_SPECIALTY" and (previous_step and previous_step not in ["VALIDATING_SPECIALTY", "VALIDATING_FULL_NAME"]):
        logger.info("Usuário escolheu (via LLM) voltar para a escolha da especialidade.")
        updates["user_chosen_specialty"] = None
        updates["user_chosen_specialty_id"] = None
        updates["professional_preference_type"] = None
        # ... (resetar outros campos do estado como no exemplo anterior)
        updates["user_provided_professional_name"] = None
        updates["user_chosen_professional_id"] = None
        updates["user_chosen_professional_name"] = None
        updates["available_professionals_list"] = None
        updates["user_chosen_turn"] = None
        updates["available_dates_presented"] = None
        updates["user_chosen_date"] = None
        updates["available_times_presented"] = None
        updates["user_chosen_time"] = None
        updates["user_chosen_time_fim"] = None
        updates["scheduling_step"] = "VALIDATING_SPECIALTY"
    elif llm_classification == "CANCEL_SCHEDULING":
        logger.info("Usuário escolheu (via LLM) cancelar o agendamento a partir do fallback.")
        updates["response_to_user"] = f"Entendido, {user_full_name}. O processo de agendamento foi cancelado. Se precisar de algo mais, é só chamar!"
        updates["current_operation"] = None 
        updates["scheduling_step"] = None 
        # ... (resetar estado de agendamento como no exemplo anterior)
        updates["user_full_name"] = state.get("user_full_name") 
        updates["user_phone"] = state.get("user_phone") 
        updates["user_chosen_specialty"] = None
        updates["user_chosen_specialty_id"] = None
        updates["professional_preference_type"] = None
        updates["user_provided_professional_name"] = None
        updates["user_chosen_professional_id"] = None
        updates["user_chosen_professional_name"] = None
        updates["available_professionals_list"] = None
        updates["user_chosen_turn"] = None
        updates["available_dates_presented"] = None
        updates["user_chosen_date"] = None
        updates["available_times_presented"] = None
        updates["user_chosen_time"] = None
        updates["user_chosen_time_fim"] = None
        updates["scheduling_completed"] = False
        updates["scheduling_values_confirmed"] = None
    else: # AMBIGUOUS_OR_UNAFFILIATED ou erro do LLM
        logger.warning(f"Escolha de fallback ambígua ou não afiliada (LLM: '{llm_classification}') para '{user_message_content}'. Solicitando novamente.")
        
        # Reconstrói a mensagem de erro do placeholder_fallback_node para reprompt.
        # Esta é a parte que idealmente rotearia de volta para placeholder_fallback_node.
        # Por agora, vamos apenas construir a mensagem de reprompt.
        error_response_text = f"Desculpe, {user_full_name}, não consegui entender sua escolha '{user_message_content}' claramente em relação às opções que apresentei.\n"
        error_options_list_text = [] # Recriar a lista de opções textuais
        
        current_step_desc_reprompt = step_map.get(previous_step, "etapa anterior") if previous_step else "etapa anterior"
        
        option_counter = 1
        error_options_list_text.append(f"{option_counter}. Tentar novamente a etapa de '{current_step_desc_reprompt}'.")
        
        if previous_step and previous_step not in ["VALIDATING_SPECIALTY", "VALIDATING_FULL_NAME"]:
            option_counter += 1
            error_options_list_text.append(f"{option_counter}. Voltar para a escolha da especialidade.")
        
        option_counter += 1
        error_options_list_text.append(f"{option_counter}. Cancelar o agendamento.")
        
        error_response_text += "Poderia, por favor, escolher uma das seguintes opções (pode dizer o número ou descrever sua escolha):\n"
        error_response_text += "\n".join(error_options_list_text)

        updates["response_to_user"] = error_response_text
        updates["scheduling_step"] = "AWAITING_FALLBACK_CHOICE" 
        updates["previous_scheduling_step"] = previous_step # Mantém para a próxima tentativa

    logger.info(f"Process fallback choice (LLM): Próximo passo definido como '{updates.get('scheduling_step')}'.")
    return updates

# === consultas api ===

def coletar_validar_turno_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Nó para coletar a resposta do usuário sobre o turno e validá-la.
    Inspirado em agentv1.py (processar_input_turno).
    """
    logger.debug("--- Nó Agendamento: coletar_validar_turno_node ---")
    user_response_content = get_last_user_message_content(state["messages"])

    if not user_response_content:
        logger.warning("Nenhuma resposta do usuário para coletar/validar o turno.")
        return {
            "response_to_user": "Não recebi sua preferência de turno. Manhã ou tarde?", 
            "scheduling_step": "VALIDATING_TURN_PREFERENCE", 
            "current_operation": "SCHEDULING"
        }

    logger.info(f"Resposta do usuário sobre turno: '{user_response_content}'")
    prompt_classificacao_turno_str = """
    Analise a seguinte resposta do usuário, que foi perguntado se prefere o período da manhã ou da tarde para um agendamento.
    Classifique a resposta em uma das seguintes categorias: "MANHA", "TARDE", ou "INVALIDO".

    Exemplos:
    - Usuário: "manhã" -> MANHA
    - Usuário: "de manhã, por favor" -> MANHA
    - Usuário: "pode ser de tarde" -> TARDE
    - Usuário: "à tarde" -> TARDE
    - Usuário: "tanto faz" -> INVALIDO
    - Usuário: "sim" -> INVALIDO

    Retorne APENAS a categoria ("MANHA", "TARDE", ou "INVALIDO").

    Resposta do usuário: "{user_response}"
    Categoria:
    """
    prompt_template_turno = ChatPromptTemplate.from_template(prompt_classificacao_turno_str)
    
    try:
        chain_turno = prompt_template_turno | llm_client
        llm_classification_response_str = chain_turno.invoke({"user_response": user_response_content}).content.strip().upper()
        logger.info(f"LLM classificou o turno como: '{llm_classification_response_str}' para o input '{user_response_content}'")

        if llm_classification_response_str == "MANHA":
            return {
                "user_chosen_turn": "MANHA",
                "response_to_user": None, 
                "scheduling_step": "FETCHING_AVAILABLE_DATES", 
                "current_operation": "SCHEDULING"
            }
        elif llm_classification_response_str == "TARDE":
            return {
                "user_chosen_turn": "TARDE",
                "response_to_user": None, 
                "scheduling_step": "FETCHING_AVAILABLE_DATES", 
                "current_operation": "SCHEDULING"
            }
        else: 
            logger.warning(f"LLM retornou categoria de turno não reconhecida ou inválida: '{llm_classification_response_str}'")
            user_name_for_prompt = state.get("user_full_name", "você")
            professional_for_prompt = state.get("user_chosen_professional_name", f"um profissional de {state.get('user_chosen_specialty', 'a especialidade')}")
            reprompt_messages = REQUEST_TURN_PREFERENCE_PROMPT_TEMPLATE.format_messages(
                user_name=user_name_for_prompt,
                professional_name_or_specialty_based=professional_for_prompt
            )
            ai_reprompt_response = llm_client.invoke(reprompt_messages)
            reprompt_text = ai_reprompt_response.content.strip()

            return {
                "response_to_user": reprompt_text,
                "scheduling_step": "VALIDATING_TURN_PREFERENCE", 
                "current_operation": "SCHEDULING",
                "user_chosen_turn": None 
            }

    except Exception as e:
        logger.error(f"Erro inesperado durante classificação de turno: {e}", exc_info=True)
        return {
            "response_to_user": "Ocorreu um erro ao processar sua preferência de turno. Vamos tentar novamente.",
            "scheduling_step": "VALIDATING_TURN_PREFERENCE",
            "current_operation": "SCHEDULING"
        }

def fetch_and_present_available_dates_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Busca datas disponíveis na API para o profissional e apresenta até 3 opções.
    Inspirado em 'consultar_e_apresentar_datas_disponiveis' do agentv1.py.
    """
    logger.debug("--- Nó Agendamento: fetch_and_present_available_dates_node ---")
    
    API_TOKEN = "TXH3xnwI7P4Sh5dS71aRDsQzDrx1GKeW8jXd5eHDpqTOi"
    api_headers = {
        "Authorization": f"{API_TOKEN}",
        "Content-Type": "application/json"
    }

    id_profissional = state.get("user_chosen_professional_id")
    nome_profissional = state.get("user_chosen_professional_name", "O profissional selecionado")
    user_full_name = state.get("user_full_name", "Paciente")
    user_chosen_turn = state.get("user_chosen_turn", "qualquer turno")

    if not id_profissional:
        logger.error("ID do profissional não encontrado no estado para buscar datas.")
        error_prompt = ChatPromptTemplate.from_template(
            "Você é um assistente de agendamento. Informe ao usuário {user_name} que ocorreu um problema "
            "ao tentar identificar o profissional para buscar as datas e que será necessário tentar novamente a seleção do profissional."
        )
        error_response_content = llm_client.invoke(error_prompt.format_messages(user_name=user_full_name)).content.strip()
        return {
            "response_to_user": error_response_content,
            "scheduling_step": "REQUESTING_PROFESSIONAL_PREFERENCE", 
            "current_operation": "SCHEDULING",
            "user_chosen_professional_id": None, 
            "user_chosen_professional_name": None
        }

    hoje = date.today()
    datas_validas_api_format = [] 

    
    for i in range(2):
        data_alvo_consulta = hoje + timedelta(days=30 * i)
        mes_consulta = data_alvo_consulta.strftime("%m")
        ano_consulta = data_alvo_consulta.strftime("%Y")
        
        url = f"https://back.homologacao.apphealth.com.br:9090/api-vizi/agenda/profissionais/{id_profissional}/datas"
        params = {"mes": mes_consulta, "ano": ano_consulta}
        logger.info(f"Consultando API de datas: {url} com params: {params}")

        try:
            import requests
            response = requests.get(url, headers=api_headers, params=params, timeout=10)
            response.raise_for_status() 
            datas_mes_api = response.json() 

            if datas_mes_api and isinstance(datas_mes_api, list):
                for item_data in datas_mes_api:
                    if isinstance(item_data, dict) and "data" in item_data:
                        try:
                            datetime.strptime(item_data["data"], "%Y-%m-%d")
                            datas_validas_api_format.append(item_data["data"])
                        except ValueError:
                            logger.warning(f"Formato de data inválido da API: {item_data['data']}")
            
            datas_validas_api_format = sorted(list(set(datas_validas_api_format)))
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"Erro HTTP ao consultar API de datas: {e.response.status_code} - {e.response.text if e.response else 'Sem corpo de resposta'}")
            error_prompt_http = ChatPromptTemplate.from_template(
                "Você é um assistente de agendamento. Informe ao usuário {user_name} que houve um problema técnico "
                "(código de erro {status_code}) ao tentar buscar as datas disponíveis e peça para tentar mais tarde."
            )
            error_response_content = llm_client.invoke(error_prompt_http.format_messages(user_name=user_full_name, status_code=e.response.status_code if e.response else "desconhecido")).content.strip()
            return {
                "response_to_user": error_response_content,
                "scheduling_step": "REQUESTING_TURN_PREFERENCE",
                "current_operation": "SCHEDULING"
            }
        except requests.exceptions.RequestException as e: 
            logger.error(f"Erro de requisição ao consultar API de datas: {e}")
            error_prompt_req = ChatPromptTemplate.from_template(
                "Você é um assistente de agendamento. Informe ao usuário {user_name} que não foi possível conectar ao sistema "
                "para buscar as datas e sugira verificar a conexão ou tentar mais tarde."
            )
            error_response_content = llm_client.invoke(error_prompt_req.format_messages(user_name=user_full_name)).content.strip()
            return {
                "response_to_user": error_response_content,
                "scheduling_step": "REQUESTING_TURN_PREFERENCE",
                "current_operation": "SCHEDULING"
            }
        except ValueError as json_err: 
            logger.error(f"Erro ao decodificar JSON da API de datas: {json_err}")
            return {
                "response_to_user": "Desculpe, {user_name}, recebi uma resposta inesperada do sistema de agendamento ao buscar as datas. Por favor, tente novamente em alguns instantes.",
                "scheduling_step": "REQUESTING_TURN_PREFERENCE",
                "current_operation": "SCHEDULING"
            }


        if len(datas_validas_api_format) >= 3:
            break
            
    if not datas_validas_api_format:
        logger.info(f"Nenhuma data disponível encontrada para Dr(a). {nome_profissional} nos próximos períodos.")
        no_dates_prompt = ChatPromptTemplate.from_template(
            "Você é um assistente de agendamento. Informe ao usuário {user_name} que, no momento, não foram encontradas datas disponíveis "
            "para o profissional {professional_name}. Pergunte se ele gostaria de tentar com outro profissional ou especialidade, ou verificar mais tarde."
        )
        no_dates_response = llm_client.invoke(no_dates_prompt.format_messages(user_name=user_full_name, professional_name=nome_profissional)).content.strip()
        return {
            "response_to_user": no_dates_response,
            "scheduling_step": "AWAITING_RETRY_OPTION_AFTER_NO_AVAILABILITY",
            "current_operation": "SCHEDULING",
            "available_dates_presented": []
        }

    datas_para_apresentar_api_format = datas_validas_api_format[:3]
    
    datas_formatadas_usuario = [datetime.strptime(d, "%Y-%m-%d").strftime("%d/%m/%Y") for d in datas_para_apresentar_api_format]
    
    present_dates_prompt_str = """
    Você é um assistente de agendamento.
    O usuário {user_name} escolheu o turno da {chosen_turn}.
    Para o profissional {professional_name}, foram encontradas as seguintes datas disponíveis:
    {available_dates_list_str}

    Construa uma mensagem para o usuário apresentando essas datas.
    Peça para ele escolher uma das datas digitando o número da opção ou a data completa.
    Exemplo de formato para cada data na lista: "1. DD/MM/YYYY"
    Sua resposta:
    """
    present_dates_prompt = ChatPromptTemplate.from_template(present_dates_prompt_str)
    
    lista_datas_str_prompt = ""
    for i, data_fmt_usr in enumerate(datas_formatadas_usuario):
        lista_datas_str_prompt += f"{i+1}. {data_fmt_usr}\n"
    
    mensagem_apresentacao_datas = llm_client.invoke(
        present_dates_prompt.format_messages(
            user_name=user_full_name,
            chosen_turn=user_chosen_turn.lower(),
            professional_name=nome_profissional,
            available_dates_list_str=lista_datas_str_prompt.strip()
        )
    ).content.strip()

    return {
        "response_to_user": mensagem_apresentacao_datas,
        "scheduling_step": "VALIDATING_CHOSEN_DATE", 
        "current_operation": "SCHEDULING",
        "available_dates_presented": datas_para_apresentar_api_format 
    }

def process_final_scheduling_confirmation_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Processa a resposta do usuário à pergunta de confirmação final do agendamento.
    """
    logger.debug("--- Nó Agendamento: process_final_scheduling_confirmation_node ---")
    user_response_content = get_last_user_message_content(state["messages"])
    user_full_name = state.get("user_full_name", "Cliente")
    user_phone_from_state = state.get("user_phone")

    telefone_formatado_para_api = user_phone_from_state
    if user_phone_from_state and isinstance(user_phone_from_state, str) and len(user_phone_from_state) > 2 and user_phone_from_state.startswith("55"):
        telefone_formatado_para_api = user_phone_from_state[2:]
        logger.info(f"Número de telefone original '{user_phone_from_state}' formatado para '{telefone_formatado_para_api}' (removido '55' inicial).")
    elif user_phone_from_state:
        logger.info(f"Número de telefone '{user_phone_from_state}' utilizado como está (não inicia com '55' ou é muito curto).")
    else:
        logger.warning("Número de telefone não encontrado no estado. Usando placeholder se necessário para API de agendamento.")
        telefone_formatado_para_api = "00000000000"

    if not user_response_content:
        logger.warning("Nenhuma resposta do usuário para processar a confirmação final.")
        return {
            "response_to_user": "Não recebi sua confirmação. Poderia confirmar o agendamento com 'sim' ou 'não'?",
            "scheduling_step": "AWAITING_FINAL_CONFIRMATION", 
            "current_operation": "SCHEDULING"
        }

    logger.info(f"Resposta do usuário para confirmação final: '{user_response_content}'")

    try:
        prompt_messages = VALIDATE_FINAL_CONFIRMATION_PROMPT_TEMPLATE.format_messages(user_response=user_response_content)
        llm_response = llm_client.invoke(prompt_messages)
        confirmation_status = llm_response.content.strip().upper()
        logger.info(f"Status da confirmação final pelo LLM: {confirmation_status}")

        if confirmation_status == "CONFIRMED":
            profissional_id = state.get("user_chosen_professional_id")
            data_agendamento = state.get("user_chosen_date") 
            hora_inicio_hhmm_str = state.get("user_chosen_time") 
            hora_fim_hhmmss_str = state.get("user_chosen_time_fim")
            nome_paciente = state.get("user_full_name")
            especialidade_id = state.get("user_chosen_specialty_id")

            if not all([profissional_id, data_agendamento, hora_inicio_hhmm_str, hora_fim_hhmmss_str, nome_paciente, especialidade_id]):
                logger.error(f"Dados insuficientes no estado para realizar o agendamento via API. Detalhes: prof_id={profissional_id}, data={data_agendamento}, hora_inicio={hora_inicio_hhmm_str}, hora_fim={hora_fim_hhmmss_str}, nome={nome_paciente}, esp_id={especialidade_id}")
                return {
                    "response_to_user": "Desculpe, ocorreu um problema com os dados do agendamento e não pude confirmar. Por favor, tente refazer o agendamento ou contate o suporte.",
                    "scheduling_completed": False, "current_operation": None, "scheduling_step": None
                }
            
            try:
                hora_inicio_api_format = datetime.strptime(hora_inicio_hhmm_str, "%H:%M").strftime("%H:%M:%S")
            except ValueError:
                logger.error(f"Formato de hora_inicio_hhmm_str inválido: {hora_inicio_hhmm_str}")
                return {"response_to_user": "Erro no formato da hora de início.", "scheduling_step": "AWAITING_TIME_CHOICE"}

            if not re.match(r'^\d{2}:\d{2}:\d{2}$', hora_fim_hhmmss_str):
                logger.error(f"Formato de hora_fim_hhmmss_str inválido: {hora_fim_hhmmss_str}. Esperado HH:MM:SS.")
                return {"response_to_user": "Erro no formato da hora de fim.", "scheduling_step": "AWAITING_TIME_CHOICE"}
            
            telefone_para_payload_agendamento = telefone_formatado_para_api
            if telefone_para_payload_agendamento == "00000000000" and user_phone_from_state is not None: 
                 logger.warning(f"Apesar de user_phone_from_state ('{user_phone_from_state}') existir, telefone_formatado_para_api resultou em placeholder. Verifique a lógica de formatação.")
            elif telefone_para_payload_agendamento == "00000000000":
                 logger.warning(f"UTILIZANDO TELEFONE PLACEHOLDER '{telefone_para_payload_agendamento}' PARA AGENDAMENTO.")
            else:
                logger.info(f"Utilizando telefone '{telefone_para_payload_agendamento}' (formatado de '{user_phone_from_state}') para o agendamento.")

            
            id_unidade_default = 21641

            payload = {
                "data": data_agendamento,
                "horaInicio": hora_inicio_api_format, 
                "horaFim": hora_fim_hhmmss_str,
                "nome": nome_paciente,
                "telefonePrincipal": telefone_para_payload_agendamento,
                "situacao": "AGENDADO",
                "profissionalSaude": {"id": profissional_id},
                "especialidade": {"id": especialidade_id},
                "paciente": {"nome": nome_paciente},
                "unidade": {"id": id_unidade_default}
            }

            api_token = settings.APPHEALTH_API_TOKEN
            headers = {"Authorization": f"{api_token}", "Content-Type": "application/json"}
            api_url = "https://back.homologacao.apphealth.com.br:9090/api-vizi/agendamentos"

            logger.info(f"Tentando realizar agendamento via API. URL: {api_url}, Payload: {json.dumps(payload, indent=2)}")

            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=20)
                response.raise_for_status() 
                api_response_data = response.json()
                agendamento_id_api = api_response_data.get("id", "N/A") 
                logger.info(f"Agendamento CONFIRMADO via API para {user_full_name}. ID: {agendamento_id_api}. Resposta: {api_response_data}")

                if user_phone_from_state:
                    n8n_webhook_url = f"https://n8n-server.apphealth.com.br/webhook/remove-tag?phone={user_phone_from_state}"
                    logger.info(f"Tentando chamar webhook N8N para remover tag: GET {n8n_webhook_url}")
                    try:
                        response_n8n = requests.get(n8n_webhook_url, timeout=10)
                        response_n8n.raise_for_status()
                        logger.info(f"Webhook N8N 'remove-tag' chamado com sucesso para {user_phone_from_state}. Status: {response_n8n.status_code}. Resposta: {response_n8n.text[:200]}")
                    except requests.exceptions.HTTPError as http_err_n8n:
                        logger.error(f"Erro HTTP ao chamar webhook N8N 'remove-tag' para {user_phone_from_state}: {http_err_n8n.response.status_code} - {http_err_n8n.response.text[:200] if http_err_n8n.response else 'Sem corpo'}", exc_info=False)
                    except requests.exceptions.RequestException as req_err_n8n:
                        logger.error(f"Erro de requisição ao chamar webhook N8N 'remove-tag' para {user_phone_from_state}: {req_err_n8n}", exc_info=False)
                    except Exception as e_n8n:
                        logger.error(f"Erro inesperado ao chamar webhook N8N 'remove-tag' para {user_phone_from_state}: {e_n8n}", exc_info=False)
                else:
                    logger.warning("Não foi possível chamar o webhook N8N 'remove-tag' pois o número de telefone não está disponível no estado.")
                
                chosen_specialty = state.get("user_chosen_specialty", "N/A")
                chosen_professional_name = state.get("user_chosen_professional_name", "N/A")
                chosen_date_display = datetime.strptime(data_agendamento, "%Y-%m-%d").strftime("%d/%m/%Y") if data_agendamento else "N/A"
                
                try:
                    success_prompt_messages = SCHEDULING_SUCCESS_MESSAGE_PROMPT_TEMPLATE.format_messages(
                        user_name=user_full_name,
                        chosen_specialty=chosen_specialty,
                        chosen_professional_name=chosen_professional_name,
                        chosen_date_display=chosen_date_display,
                        chosen_time=hora_inicio_hhmm_str,
                        agendamento_id_api=agendamento_id_api
                    )
                    success_response_llm = llm_client.invoke(success_prompt_messages)
                    success_message = success_response_llm.content.strip()
                    logger.info(f"Mensagem de sucesso gerada pelo LLM: '{success_message}'")
                
                except Exception as e_prompt:
                    logger.error(f"Erro ao gerar mensagem de sucesso com LLM: {e_prompt}. Usando fallback.")
                    success_message = (
                        f"Ótimo, {user_full_name}! Seu agendamento para {chosen_specialty} com {chosen_professional_name} "
                        f"no dia {chosen_date_display} às {hora_inicio_hhmm_str} foi confirmado com sucesso (Ticket de confirmação: {agendamento_id_api}). "
                    )
                
                return {
                    "response_to_user": success_message, "scheduling_completed": True,
                    "current_operation": None, "scheduling_step": None, 
                    "scheduling_values_confirmed": { 
                        "name": user_full_name, "specialty": chosen_specialty, 
                        "professional": chosen_professional_name, "date": data_agendamento,
                        "time_start": hora_inicio_hhmm_str, "time_end": hora_fim_hhmmss_str,
                        "api_schedule_id": agendamento_id_api,
                        "phone_original": user_phone_from_state,
                        "phone_formatted_for_api": telefone_formatado_para_api
                    }
                }
            except requests.exceptions.HTTPError as http_err:
                error_content = "N/A"
                if http_err.response is not None:
                    try: error_content = http_err.response.json()
                    except json.JSONDecodeError: error_content = http_err.response.text
                logger.error(f"Erro HTTP ao realizar agendamento: {http_err.response.status_code if http_err.response is not None else 'N/A'} - {error_content}", exc_info=True)
                user_error_message = "Desculpe, não consegui confirmar seu agendamento com nosso sistema. Por favor, tente mais tarde ou contate o suporte."
                if http_err.response is not None and http_err.response.status_code == 400:
                    user_error_message = "Houve um problema com os dados fornecidos para o agendamento. Verifique os detalhes e tente novamente ou contate o suporte."
                return {"response_to_user": user_error_message, "scheduling_completed": False, "current_operation": None, "scheduling_step": None, "error_message": f"API Error: {http_err.response.status_code if http_err.response is not None else 'N/A'} - {error_content}"}
            except requests.exceptions.RequestException as req_err:
                logger.error(f"Erro de conexão/rede ao realizar agendamento: {req_err}", exc_info=True)
                return {"response_to_user": "Desculpe, estou com dificuldades para me conectar ao sistema de agendamentos. Tente novamente em instantes.", "scheduling_completed": False, "current_operation": None, "scheduling_step": None, "error_message": f"Network/Request Error: {str(req_err)}"}

        elif confirmation_status == "CANCELLED":
            logger.info(f"Agendamento CANCELADO por {user_full_name}.")
            return {
                "response_to_user": "Entendido. O agendamento não foi confirmado. Se precisar de algo mais, é só chamar!",
                "scheduling_completed": False, 
                "current_operation": None, 
                "scheduling_step": None
            }
        else: 
            logger.warning(f"Resposta de confirmação ambígua ou não classificada: '{confirmation_status}'")
            return {
                "response_to_user": "Desculpe, não entendi sua resposta. Para confirmar o agendamento, por favor, diga 'sim'. Se não deseja confirmar, diga 'não'.",
                "scheduling_step": "AWAITING_FINAL_CONFIRMATION", 
                "current_operation": "SCHEDULING"
            }
    except Exception as e:
        logger.error(f"Erro ao processar confirmação final do agendamento: {e}", exc_info=True)
        return {
            "response_to_user": "Desculpe, tive um problema ao processar sua confirmação. Poderia tentar confirmar novamente com 'sim' ou 'não'?",
            "scheduling_step": "AWAITING_FINAL_CONFIRMATION",
            "current_operation": "SCHEDULING"
        }

# === consulta api ===

# <<< ADICIONANDO AS FUNÇÕES DOS NÓS FALLBACK >>>
def placeholder_fallback_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Nó de fallback para lidar com erros não recuperáveis ou intenções não claras durante uma operação.
    Oferece opções ao usuário para tentar novamente, mudar de rota ou cancelar.
    """
    logger.debug(f"--- Nó: placeholder_fallback_node. Estado atual: {state} ---")
    user_full_name = state.get("user_full_name", "Prezado(a) cliente")
    current_op = state.get("current_operation")
    current_step = state.get("scheduling_step") 
    fallback_custom_message = state.get("fallback_context_message")

    if fallback_custom_message:
        response_text = f"Desculpe, {user_full_name}, notei um problema: {fallback_custom_message}\n"
    else:
        response_text = f"Desculpe, {user_full_name}, tive dificuldades em processar sua última solicitação.\n"

    options_list = []
    updates_for_state = {
        "fallback_context_message": None
    }

    if current_op == "SCHEDULING" and current_step:
        step_map = {
            "VALIDATING_FULL_NAME": "confirmação do seu nome",
            "VALIDATING_SPECIALTY": "escolha da especialidade",
            "CLASSIFYING_PROFESSIONAL_PREFERENCE": "preferência de profissional",
            "PROCESSING_PROFESSIONAL_LOGIC": "validação do profissional",
            "LISTING_AVAILABLE_PROFESSIONALS": "busca por profissionais",
            "VALIDATING_CHOSEN_PROFESSIONAL_FROM_LIST": "escolha do profissional da lista",
            "VALIDATING_TURN_PREFERENCE": "escolha do turno",
            "FETCHING_AVAILABLE_DATES": "busca por datas disponíveis",
            "VALIDATING_CHOSEN_DATE": "confirmação da data",
            "FETCHING_AVAILABLE_TIMES": "busca por horários",
            "AWAITING_TIME_CHOICE": "escolha do horário",
            "AWAITING_FINAL_CONFIRMATION": "confirmação final do agendamento"
        }
        step_description = step_map.get(current_step, "etapa atual")
        
        response_text += f"Estávamos na etapa de '{step_description}'.\n"
        response_text += "O que você gostaria de fazer?\n"
        
        options_list.append(f"1. Tentar novamente a etapa de '{step_description}'.")
        updates_for_state["previous_scheduling_step"] = current_step

        if current_step not in ["VALIDATING_SPECIALTY", "VALIDATING_FULL_NAME"]:
            options_list.append("2. Voltar para a escolha da especialidade.")
        options_list.append(f"{len(options_list) + 1}. Cancelar o agendamento.")

        updates_for_state["scheduling_step"] = "AWAITING_FALLBACK_CHOICE"
        updates_for_state["current_operation"] = "SCHEDULING"
    
    else:
        last_user_message = get_last_user_message_content(state.get("messages", []))
        if last_user_message:
            response_text += f"Não consegui entender bem '{last_user_message}'. "
        response_text += "Poderia, por favor, reformular sua solicitação ou me dizer como posso te ajudar agora (ex: 'quero fazer um agendamento')?"
        updates_for_state["current_operation"] = None
        updates_for_state["scheduling_step"] = None
        updates_for_state["categoria"] = "Indefinido" 
        updates_for_state["previous_scheduling_step"] = None

    if options_list:
        response_text += "\n".join(options_list)
        response_text += "\nPor favor, digite o número da opção desejada."

    updates_for_state["response_to_user"] = response_text
    logger.info(f"Placeholder fallback: Próximo passo definido como '{updates_for_state.get('scheduling_step')}'. Mensagem: '{response_text}'")
    return updates_for_state

# === NOVO NÓ DISPATCHER ===
def dispatcher_node(state: MainWorkflowState) -> dict:
    """
    Verifica se há uma operação em andamento.
    Este nó não modifica o estado, apenas o lê para o roteamento.
    Retorna um dicionário que pode ser usado pela lógica de roteamento condicional.
    """
    logger.debug("--- Nó: dispatcher_node ---")
    current_op = state.get("current_operation")
    logger.info(f"Dispatcher: Current operation = {current_op}")
    
    # Este nó não altera o estado, apenas retorna o que será usado para rotear.
    # A lógica de roteamento condicional usará o valor de 'current_operation'.
    return {"current_operation": current_op} # Passa o valor para a função de roteamento

# === LÓGICA DE ROTEAMENTO ===
def route_initial_or_ongoing(state: MainWorkflowState) -> str:
    current_op = state.get("current_operation")
    messages = state.get("messages", [])
    last_message_is_human = messages and isinstance(messages[-1], HumanMessage)
    logger.info(f"Roteamento Inicial/Contínuo: Current operation = {current_op}")

    if current_op == "SCHEDULING" and last_message_is_human:
        if state.get("response_to_user") is None or state.get("cancellation_check_result") is None:
             logger.info("Operação de agendamento em progresso com nova mensagem humana. Roteando para 'check_cancellation_node'.")
             return "check_cancellation_node"
        else:
            logger.info("Operação de agendamento em progresso, mas parece ser fluxo interno ou resposta já definida. Roteando para 'route_scheduling_step'.")
            state["cancellation_check_result"] = None
            return "route_scheduling_step"
    elif current_op == "SCHEDULING":
        logger.info("Operação de agendamento em progresso (fluxo interno). Roteando para 'route_scheduling_step'.")
        return "route_scheduling_step"
    else:
        logger.info("Nenhuma operação de agendamento em progresso. Roteando para 'categorize_intent'.")
        return "categorize_intent"

def route_after_cancellation_check(state: MainWorkflowState) -> str:
    """
    Roteia após a verificação de intenção de cancelamento.
    """
    cancellation_result = state.get("cancellation_check_result")
    logger.info(f"Roteamento após verificação de cancelamento: Resultado = {cancellation_result}")
    if cancellation_result == "CANCELLED":
        return END
    return "route_scheduling_step"

def route_after_categorization(state: MainWorkflowState) -> str:
    categoria = state.get("categoria")
    logger.info(f"Roteando com base na categoria: '{categoria}'")

    if categoria == "Saudação ou Despedida":
        return "handle_greeting_farewell"
    elif categoria == "Criar Agendamento": 
        return "solicitar_nome_agendamento_node" 
    elif categoria == "Erro na Categorização" or categoria == "Indefinido":
        return "handle_fallback_placeholder"
    else: 
        return "handle_fallback_placeholder"

def route_scheduling_step(state: MainWorkflowState) -> str:
    step = state.get("scheduling_step")
    logger.info(f"Roteamento do Agendamento: scheduling_step = {step}")

    if step == "VALIDATING_FULL_NAME":
        return "coletar_validar_nome_agendamento_node"
    elif step == "VALIDATING_SPECIALTY": 
        return "coletar_validar_especialidade_node"
    elif step == "REQUESTING_PROFESSIONAL_PREFERENCE":
        return "solicitar_preferencia_profissional_node"
    elif step == "FETCHING_AVAILABLE_DATES": 
        return "fetch_and_present_available_dates_node"
    elif step == "CLASSIFYING_PROFESSIONAL_PREFERENCE": 
        return "coletar_classificar_preferencia_profissional_node"
    elif step == "PROCESSING_PROFESSIONAL_LOGIC": 
        return "processing_professional_logic_node"
    elif step == "LISTING_AVAILABLE_PROFESSIONALS": 
        return "list_available_professionals_node"
    elif step == "VALIDATING_CHOSEN_PROFESSIONAL_FROM_LIST": 
        return "collect_validate_chosen_professional_node"
    elif step == "REQUESTING_TURN_PREFERENCE": 
        return "solicitar_turno_node"
    elif step == "VALIDATING_TURN_PREFERENCE": 
        return "coletar_validar_turno_node"
    elif step == "FETCHING_AVAILABLE_DATES": 
        return "fetch_and_present_available_dates_node"
    elif step == "VALIDATING_CHOSEN_DATE": 
        return "collect_validate_chosen_date_node"
    elif step == "FETCHING_AVAILABLE_TIMES": 
        return "fetch_and_present_available_times_node"
    elif step == "AWAITING_TIME_CHOICE":
        return "coletar_validar_horario_escolhido_node"
    elif step == "AWAITING_FINAL_CONFIRMATION":
        return "process_final_scheduling_confirmation_node"
    elif step == "AWAITING_RETRY_OPTION_AFTER_NO_AVAILABILITY":
        return "process_retry_option_choice_node"
    elif step == "INITIATE_FALLBACK":
        return "placeholder_fallback_node"
    
    logger.warning(f"Roteamento do Agendamento: Passo desconhecido ou não manuseado '{step}'. Finalizando o fluxo de agendamento.")
    return END

def route_after_processing_professional_logic(state: MainWorkflowState) -> str:
        # Se o nó processing_professional_logic_node preparou uma resposta direta ao usuário,
        # ou se o passo seguinte requer uma nova entrada do usuário após uma pergunta,
        # então o fluxo deve parar para enviar essa resposta/pergunta.
        # Casos em que response_to_user é setado por processing_professional_logic_node:
        # - Nome específico validado -> pergunta de turno
        # - Nome específico NÃO validado -> mensagem de erro/reprompt
        # - Usuário quer fornecer nome depois -> pergunta o nome
        # Caso em que response_to_user NÃO é setado por processing_professional_logic_node (fluxo interno):
        # - Preferência por recomendação -> vai listar profissionais (scheduling_step = "LISTING_AVAILABLE_PROFESSIONALS")

        current_scheduling_step = state.get("scheduling_step")
        response_for_user_is_set = bool(state.get("response_to_user"))

        if response_for_user_is_set:
            logger.info(f"route_after_processing_professional_logic: response_to_user foi definido. Próximo passo: {current_scheduling_step}. Indo para END.")
            return END
        elif current_scheduling_step == "LISTING_AVAILABLE_PROFESSIONALS": # Caso de recomendação, fluxo interno
            logger.info(f"route_after_processing_professional_logic: Indo para listar profissionais. Próximo passo: {current_scheduling_step}. Indo para route_scheduling_step.")
            return "route_scheduling_step"
        else:
            # Fallback inesperado, mas seguro é ir para END se não soubermos continuar internamente.
            # Ou poderia ser um erro/log mais específico.
            logger.warning(f"route_after_processing_professional_logic: Condição não prevista. response_to_user: {response_for_user_is_set}, scheduling_step: {current_scheduling_step}. Indo para END como fallback.")
            return END

def route_after_potential_user_prompt(state: MainWorkflowState) -> str:
        if state.get("response_to_user"):
            logger.info(f"Roteador (route_after_potential_user_prompt): 'response_to_user' está definido. Indo para END.")
            return END
        else:
            logger.info(f"Roteador (route_after_potential_user_prompt): 'response_to_user' NÃO definido. Indo para 'route_scheduling_step' para continuar fluxo interno.")
            return "route_scheduling_step"

def route_after_fallback_choice(state: MainWorkflowState) -> str:
    """
    Roteia após o usuário ter feito uma escolha no nó de fallback.
    """
    logger.info(f"Roteamento após escolha de fallback. Estado: {state}")
    next_scheduling_step = state.get("scheduling_step")
    response_to_user = state.get("response_to_user")

    if response_to_user: 
        logger.info(f"Roteamento após fallback: response_to_user definido. Indo para END para enviar mensagem.")
        return END
    elif next_scheduling_step:
        logger.info(f"Roteamento após fallback: próximo passo é '{next_scheduling_step}'. Indo para route_scheduling_step.")
        return "route_scheduling_step"
    else:
        logger.warning("Roteamento após fallback: Nenhuma resposta ao usuário e nenhum próximo passo. Fluxo de cancelamento pode estar incompleto. Indo para END.")
        return END

# === CONSTRUÇÃO DO GRAFO ===
def get_main_conversation_graph_definition() -> StateGraph:
    logger.info("Definindo a estrutura do grafo principal da conversa (sem subgrafos)")
    workflow_builder = StateGraph(MainWorkflowState)
    llm_instance = get_llm_client()

    workflow_builder.add_node("dispatcher", dispatcher_node)
    workflow_builder.add_node("categorize_intent", partial(categorize_node, llm_client=llm_instance))
    workflow_builder.add_node("handle_greeting_farewell", partial(greeting_farewell_node, llm_client=llm_instance))
    workflow_builder.add_node("placeholder_fallback_node", partial(placeholder_fallback_node, llm_client=llm_instance))
    workflow_builder.add_node("solicitar_nome_agendamento_node", partial(solicitar_nome_agendamento_node, llm_client=llm_instance))
    workflow_builder.add_node("coletar_validar_nome_agendamento_node", partial(coletar_validar_nome_agendamento_node, llm_client=llm_instance))
    workflow_builder.add_node("coletar_validar_especialidade_node", partial(coletar_validar_especialidade_node, llm_client=llm_instance)) 
    workflow_builder.add_node("solicitar_preferencia_profissional_node", partial(solicitar_preferencia_profissional_node, llm_client=llm_instance))
    workflow_builder.add_node("coletar_classificar_preferencia_profissional_node", partial(coletar_classificar_preferencia_profissional_node, llm_client=llm_instance))
    workflow_builder.add_node("processing_professional_logic_node", partial(processing_professional_logic_node, llm_client=llm_instance))
    workflow_builder.add_node("list_available_professionals_node", partial(list_available_professionals_node, llm_client=llm_instance))
    workflow_builder.add_node("collect_validate_chosen_professional_node", partial(collect_validate_chosen_professional_node, llm_client=llm_instance))
    workflow_builder.add_node("solicitar_turno_node", partial(solicitar_turno_node, llm_client=llm_instance))
    workflow_builder.add_node("coletar_validar_turno_node", partial(coletar_validar_turno_node, llm_client=llm_instance))
    workflow_builder.add_node("fetch_and_present_available_dates_node", partial(fetch_and_present_available_dates_node, llm_client=llm_instance))
    workflow_builder.add_node("collect_validate_chosen_date_node", partial(collect_validate_chosen_date_node, llm_client=llm_instance))
    workflow_builder.add_node("fetch_and_present_available_times_node", partial(fetch_and_present_available_times_node, llm_client=llm_instance))
    workflow_builder.add_node("process_retry_option_choice_node", partial(process_retry_option_choice_node, llm_client=llm_instance))
    workflow_builder.add_node("coletar_validar_horario_escolhido_node", partial(coletar_validar_horario_escolhido_node, llm_client=llm_instance))
    workflow_builder.add_node("process_final_scheduling_confirmation_node", partial(process_final_scheduling_confirmation_node, llm_client=llm_instance))
    workflow_builder.add_node("route_after_user_interaction", lambda state: state) 
    workflow_builder.add_node("check_cancellation_node", partial(check_cancellation_node, llm_client=llm_instance))
    workflow_builder.add_node("process_fallback_choice_node", partial(process_fallback_choice_node, llm_client=llm_instance))

    workflow_builder.set_entry_point("dispatcher")

    workflow_builder.add_conditional_edges(
        "dispatcher",
        route_initial_or_ongoing,
        {
            "route_scheduling_step": "route_scheduling_step", 
            "categorize_intent": "categorize_intent",
            "check_cancellation_node": "check_cancellation_node",
        }
    )

    workflow_builder.add_conditional_edges(
        "check_cancellation_node",
        route_after_cancellation_check,
        {
            END: END,
            "route_scheduling_step": "route_scheduling_step"
        }
    )

    workflow_builder.add_conditional_edges(
        "categorize_intent",
        route_after_categorization,
        {
            "handle_greeting_farewell": "handle_greeting_farewell",
            "solicitar_nome_agendamento_node": "solicitar_nome_agendamento_node", 
            "handle_fallback_placeholder": "placeholder_fallback_node"
        }
    )
    
    workflow_builder.add_node("route_scheduling_step", lambda state: state) 

    workflow_builder.add_conditional_edges(
        "route_scheduling_step", 
        route_scheduling_step,
        {
            "coletar_validar_nome_agendamento_node": "coletar_validar_nome_agendamento_node",
            "coletar_validar_especialidade_node": "coletar_validar_especialidade_node",
            "solicitar_preferencia_profissional_node": "solicitar_preferencia_profissional_node",
            "coletar_classificar_preferencia_profissional_node": "coletar_classificar_preferencia_profissional_node",
            "processing_professional_logic_node": "processing_professional_logic_node",
            "list_available_professionals_node": "list_available_professionals_node",
            "collect_validate_chosen_professional_node": "collect_validate_chosen_professional_node",
            "solicitar_turno_node": "solicitar_turno_node",
            "coletar_validar_turno_node": "coletar_validar_turno_node",
            "fetch_and_present_available_dates_node": "fetch_and_present_available_dates_node",
            "process_retry_option_choice_node": "process_retry_option_choice_node",
            "collect_validate_chosen_date_node": "collect_validate_chosen_date_node",
            "fetch_and_present_available_times_node": "fetch_and_present_available_times_node",
            "coletar_validar_horario_escolhido_node": "coletar_validar_horario_escolhido_node",
            "process_final_scheduling_confirmation_node": "process_final_scheduling_confirmation_node",
            "process_fallback_choice_node": "process_fallback_choice_node",
            "placeholder_fallback_node": "placeholder_fallback_node",
            END: END 
        }
    )

    workflow_builder.add_conditional_edges(
        "processing_professional_logic_node",
        route_after_processing_professional_logic,
        {
            END: END,
            "route_scheduling_step": "route_scheduling_step"
        }
    )

    workflow_builder.add_conditional_edges(
        "process_retry_option_choice_node",
        lambda state: "route_scheduling_step" if state.get("scheduling_step") and state.get("current_operation") == "SCHEDULING" else END,
        {
            "route_scheduling_step": "route_scheduling_step",
            END: END
        }
    )
    workflow_builder.add_conditional_edges(
        "route_after_user_interaction",
        route_after_potential_user_prompt, 
        {
            END: END,
            "route_scheduling_step": "route_scheduling_step"
        }
    )

    workflow_builder.add_conditional_edges(
        "process_fallback_choice_node",
        route_after_fallback_choice,
        {
            END: END, 
            "route_scheduling_step": "route_scheduling_step"
        }
    )

    # Saídas dos nós
    workflow_builder.add_edge("handle_greeting_farewell", END)
    workflow_builder.add_edge("placeholder_fallback_node", END)
    workflow_builder.add_edge("solicitar_nome_agendamento_node", END) 
    workflow_builder.add_edge("coletar_validar_nome_agendamento_node", "route_after_user_interaction")
    workflow_builder.add_edge("coletar_validar_especialidade_node","route_after_user_interaction")
    workflow_builder.add_edge("solicitar_preferencia_profissional_node", END)
    workflow_builder.add_edge("coletar_classificar_preferencia_profissional_node", "route_scheduling_step")
    workflow_builder.add_edge("list_available_professionals_node", END)
    workflow_builder.add_edge("collect_validate_chosen_professional_node", "route_after_user_interaction") 
    workflow_builder.add_edge("solicitar_turno_node", END) 
    workflow_builder.add_edge("coletar_validar_turno_node", "route_after_user_interaction")
    workflow_builder.add_edge("fetch_and_present_available_dates_node", END)
    workflow_builder.add_edge("collect_validate_chosen_date_node", "route_after_user_interaction")
    workflow_builder.add_edge("fetch_and_present_available_times_node", END)
    workflow_builder.add_edge("coletar_validar_horario_escolhido_node", END)
    workflow_builder.add_edge("process_final_scheduling_confirmation_node", END)
    workflow_builder.add_edge("placeholder_fallback_node", END) 
    return workflow_builder

# === FUNÇÃO DE EXECUÇÃO DO FLUXO ===
async def arun_main_conversation_flow(
    user_text: str,
    session_id: str,
    checkpointer: BaseCheckpointSaver, 
    user_phone: Optional[str] = None,
) -> Optional[str]:
    logger.info(f"Executando arun_main_conversation_flow para session_id: {session_id} com texto: '{user_text}'")
    
    if not checkpointer:
        logger.error("ERRO CRÍTICO: checkpointer não foi fornecido para arun_main_conversation_flow.")
        return "Desculpe, estou com problemas técnicos (Configuração de Checkpoint)."

    workflow_definition = get_main_conversation_graph_definition()
    graph_with_persistence = workflow_definition.compile(checkpointer=checkpointer)
    
    final_state: Optional[MainWorkflowState] = None
    
    initial_input_data = {"messages": [HumanMessage(content=user_text)]} 

    if user_phone:
        initial_input_data["user_phone"] = user_phone
    
    config = {"configurable": {"thread_id": session_id}}
    
    logger.debug(f"Invocando grafo principal para session_id: {session_id} com input_data: {initial_input_data}")

    try:
        async for event_chunk in graph_with_persistence.astream(initial_input_data, config=config, stream_mode="values"):
            final_state = event_chunk 
            logger.debug(f"Chunk do grafo (session_id {session_id}): {event_chunk}")
    except Exception as graph_error:
        logger.error(f"Erro ao invocar o grafo LangGraph para thread {session_id}: {graph_error}", exc_info=True)
        return "Desculpe, ocorreu um erro interno ao processar sua solicitação. - Erro: {graph_error}"

    if final_state:
        logger.info(f"Estado final do grafo (session_id {session_id}): {final_state}")
        
        response_content_to_send = final_state.get("response_to_user")

        if response_content_to_send:
            logger.info(f"Resposta do arun_main_conversation_flow (session_id {session_id}): '{response_content_to_send}'")
            return response_content_to_send
        else:
            all_messages = final_state.get('messages', [])
            if all_messages:
                last_message_obj = all_messages[-1]
                if isinstance(last_message_obj, AIMessage):
                    logger.warning(f"'response_to_user' não definido, mas última mensagem é AIMessage: '{last_message_obj.content}'")
                    return last_message_obj.content 
            logger.warning(f"Nenhum 'response_to_user' definido no estado final e nenhuma AIMessage clara no final do histórico para session_id {session_id}.")
            return None 
            
    logger.warning(f"Nenhum estado final retornado pelo LangGraph para session_id {session_id}.")
    return None