import logging
import json
import re
import requests

from datetime import datetime, date, timedelta, time
from functools import partial
from typing import Optional, List, Dict
from app.core.config import settings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
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
    EXTRACT_FULL_NAME_PROMPT_TEMPLATE
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
            "current_operation": None, # Limpa a operação se a categorização falhar aqui
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
    
    # Usar o prompt para gerar a pergunta sobre o nome
    # (Poderia ser um serviço como em conversation_service.py se a lógica for mais complexa)
    prompt = REQUEST_FULL_NAME_PROMPT_TEMPLATE.format_messages() # Garante que seja uma lista de BaseMessage
    logger.debug(f"Gerando solicitação de nome completo com o LLM. Prompt: {prompt}")
    ai_response = llm_client.invoke(prompt)
    resposta_llm = ai_response.content.strip()
    logger.info(f"Resposta do LLM para solicitação de nome completo: {resposta_llm}")

    return {
        "response_to_user": resposta_llm,
        "scheduling_step": "VALIDATING_FULL_NAME", # Próximo passo é validar a resposta do usuário
        "current_operation": "SCHEDULING" # Mantém a operação
    }

def coletar_validar_nome_agendamento_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Nó para coletar a resposta do usuário, extrair o nome e validá-lo.
    """
    logger.debug("--- Nó Agendamento: coletar_validar_nome_agendamento_node ---")
    user_message_content = get_last_user_message_content(state["messages"])

    if not user_message_content:
        logger.warning("Nenhuma resposta do usuário para coletar/validar o nome.")
        # Reutilizar o prompt de solicitação de nome se o usuário não enviar nada
        reprompt_messages = REQUEST_FULL_NAME_PROMPT_TEMPLATE.format_messages()
        ai_reprompt_response = llm_client.invoke(reprompt_messages)
        return {
            "response_to_user": "Não recebi seu nome. " + ai_reprompt_response.content.strip(),
            "scheduling_step": "VALIDATING_FULL_NAME", 
            "current_operation": "SCHEDULING"
        }

    logger.info(f"Mensagem recebida do usuário para extração de nome: '{user_message_content}'")

    # 1. Usar LLM para extrair o nome da mensagem do usuário
    extracted_name = "NOME_NAO_IDENTIFICADO" # Default
    try:
        extraction_prompt_messages = EXTRACT_FULL_NAME_PROMPT_TEMPLATE.format_messages(user_message=user_message_content)
        llm_extraction_response = llm_client.invoke(extraction_prompt_messages)
        extracted_name = llm_extraction_response.content.strip()
        logger.info(f"Nome extraído pelo LLM: '{extracted_name}' (da entrada: '{user_message_content}')")
    except Exception as e:
        logger.error(f"Erro ao invocar LLM para extração de nome: {e}", exc_info=True)
        # Se houver erro na extração, pedir novamente de forma genérica
        return {
            "response_to_user": "Tive um problema ao entender o nome informado. Poderia, por favor, me dizer seu nome completo novamente?",
            "scheduling_step": "VALIDATING_FULL_NAME",
            "current_operation": "SCHEDULING"
        }

    if not extracted_name or extracted_name == "NOME_NAO_IDENTIFICADO":
        logger.warning(f"LLM não conseguiu extrair um nome válido da entrada: '{user_message_content}'. Retorno do LLM: '{extracted_name}'")
        # Reutilizar o prompt de solicitação de nome
        reprompt_messages = REQUEST_FULL_NAME_PROMPT_TEMPLATE.format_messages()
        ai_reprompt_response = llm_client.invoke(reprompt_messages)
        return {
            "response_to_user": "Não consegui identificar um nome válido na sua resposta. " + ai_reprompt_response.content.strip(),
            "scheduling_step": "VALIDATING_FULL_NAME",
            "current_operation": "SCHEDULING"
        }

    # 2. Validar o nome extraído usando Pydantic
    try:
        validated_name_model = FullNameModel(full_name=extracted_name)
        final_validated_name = validated_name_model.full_name
        logger.info(f"Nome extraído e validado com sucesso: {final_validated_name}")

        prompt_messages_especialidade = REQUEST_SPECIALTY_PROMPT_TEMPLATE.format_messages(user_name=final_validated_name)
        ai_response_especialidade = llm_client.invoke(prompt_messages_especialidade)
        pergunta_especialidade = ai_response_especialidade.content.strip()
        logger.info(f"Pergunta sobre especialidade gerada após validar nome: '{pergunta_especialidade}'")
        
        return {
            "user_full_name": final_validated_name, # Salva o nome realmente validado
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
    except Exception as e: # Outros erros inesperados
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
            "scheduling_step": "VALIDATING_SPECIALTY", # Mantém para reprompt
            "user_chosen_specialty": None,
            "user_chosen_specialty_id": None
        }

    # 1. Usar LLM para validar/normalizar o nome da especialidade
    prompt_validate_specialty_messages = VALIDATE_SPECIALTY_PROMPT_TEMPLATE.format_messages(
        user_input_specialty=last_user_message # Corrigido para user_input_specialty
    )
    try:
        validated_specialty_response = llm_client.invoke(prompt_validate_specialty_messages)
        cleaned_specialty_name = validated_specialty_response.content.strip()
        logger.info(f"Resultado da validação/normalização da especialidade: '{cleaned_specialty_name}' para entrada '{last_user_message}'")

        if cleaned_specialty_name == "ENTRADA_INVALIDA_NAO_EH_ESPECIALIDADE":
            logger.info(f"Entrada '{last_user_message}' não foi considerada uma especialidade válida pelo LLM inicial.")
            # Adicionar uma mensagem mais amigável aqui, talvez listando exemplos ou perguntando novamente
            return {
                "response_to_user": f"Desculpe, não entendi '{last_user_message}' como uma especialidade médica. Poderia tentar informar novamente? Por exemplo: Cardiologia, Ortopedia, etc.",
                "scheduling_step": "VALIDATING_SPECIALTY", # Mantém para reprompt
                "user_chosen_specialty": None,
                "user_chosen_specialty_id": None
            }

    except Exception as e:
        logger.error(f"Erro ao invocar LLM para validar/normalizar especialidade: {e}")
        return {
            "response_to_user": "Desculpe, tive um problema ao tentar entender a especialidade. Poderia tentar novamente?",
            "scheduling_step": "VALIDATING_SPECIALTY",
            "user_chosen_specialty": None,
            "user_chosen_specialty_id": None
        }

    # 2. Buscar TODAS as especialidades da API
    api_token = settings.APPHEALTH_API_TOKEN
    api_headers = {
        "Authorization": f"{api_token}",
        "Content-Type": "application/json"
    }
    url_especialidades_todas = "https://back.homologacao.apphealth.com.br:9090/api-vizi/especialidades"
    
    try:
        response = requests.get(url_especialidades_todas, headers=api_headers, timeout=10)
        response.raise_for_status()
        especialidades_api_list = response.json()
        
        if not especialidades_api_list or not isinstance(especialidades_api_list, list):
            logger.warning(f"API de especialidades ({url_especialidades_todas}) retornou dados vazios ou em formato inesperado.")
            return {
                "response_to_user": "Desculpe, estou com dificuldades para carregar a lista de especialidades no momento. Por favor, tente mais tarde.",
                "scheduling_step": "VALIDATING_SPECIALTY",
                "user_chosen_specialty": cleaned_specialty_name,
                "user_chosen_specialty_id": None
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro ao buscar todas as especialidades da API ({url_especialidades_todas}): {e}")
        return {
            "response_to_user": "Desculpe, estou com dificuldades para acessar nossas especialidades no momento. Por favor, tente novamente em alguns instantes.",
            "scheduling_step": "VALIDATING_SPECIALTY",
            "user_chosen_specialty": cleaned_specialty_name,
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
            "user_chosen_specialty": cleaned_specialty_name,
            "user_chosen_specialty_id": None
        }

    # 3. Formular o prompt para o LLM encontrar a correspondência
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

    # 4. Encontrar o ID correspondente e preparar a resposta
    if nome_especialidade_llm_match != "NENHUMA_CORRESPONDENCIA" and nome_especialidade_llm_match:
        for item in especialidades_api_list:
            if item.get("especialidade") == nome_especialidade_llm_match and isinstance(item.get("id"), int):
                specialty_id_found = item["id"]
                official_specialty_name = item["especialidade"]
                logger.info(f"Sucesso! Entrada original '{last_user_message}' (normalizada para '{cleaned_specialty_name}') correspondeu a '{official_specialty_name}' (ID: {specialty_id_found}).")
                
                # Usar LLM para gerar a próxima pergunta (preferência de profissional)
                # Isso é mais flexível do que uma string fixa.
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
        
        logger.warning(f"Especialidade '{nome_especialidade_llm_match}' sugerida pelo LLM de correspondência não foi encontrada na lista original da API (isso pode indicar um problema no prompt de match ou na lista).")
        # Tratar como NENHUMA_CORRESPONDENCIA se o nome do LLM não estiver na lista.
        
    # Se NENHUMA_CORRESPONDENCIA ou o nome do LLM não foi achado na lista
    logger.info(f"Nenhuma correspondência oficial encontrada para '{cleaned_specialty_name}' (entrada original: '{last_user_message}').")
    
    available_specialties_str = "\n".join([f"- {esp}" for esp in nomes_especialidades_oficiais[:10]]) # Listar até 10
    response_text = (
        f"Desculpe, {user_full_name}, não consegui encontrar a especialidade '{cleaned_specialty_name}' em nossa lista, ou ela não está disponível no momento.\n"
        f"Atualmente, trabalhamos com as seguintes especialidades:\n{available_specialties_str}\n"
        "Por favor, escolha uma da lista ou, se preferir, pode digitar 'cancelar'."
    )
    return {
        "response_to_user": response_text,
        "scheduling_step": "VALIDATING_SPECIALTY", # Volta para o usuário tentar novamente
        "user_chosen_specialty": None,
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
        "scheduling_step": "CLASSIFYING_PROFESSIONAL_PREFERENCE", # Próximo passo
        "current_operation": "SCHEDULING"
    }

def coletar_classificar_preferencia_profissional_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Nó para coletar a resposta do usuário sobre preferência de profissional e classificá-la usando LLM.
    """
    logger.debug("--- Nó Agendamento: coletar_classificar_preferencia_profissional_node ---")
    user_response_content = get_last_user_message_content(state["messages"])
    user_specialty = state.get("user_chosen_specialty", "a especialidade escolhida") # Para o prompt de classificação

    if not user_response_content:
        logger.warning("Nenhuma resposta do usuário para coletar/classificar a preferência de profissional.")
        return {
            "response_to_user": "Não recebi sua preferência sobre o profissional. Poderia me dizer?",
            "scheduling_step": "CLASSIFYING_PROFESSIONAL_PREFERENCE", # Tenta novamente
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
        
        # Tentar extrair o JSON da resposta do LLM
        # O prompt pede "APENAS como um objeto JSON", então idealmente é só parsear.
        # Mas adicionamos uma busca por JSON como fallback.
        match = re.search(r'\{.*\}', llm_classification_response_str, re.DOTALL)
        if not match:
            logger.error(f"Nenhum JSON encontrado na resposta de classificação do LLM: '{llm_classification_response_str}'")
            # Fallback: pedir para o usuário repetir de forma mais clara
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
            # Mensagem inspirada em processar_preferencia_profissional de agentv1.py
            reprompt_message = "Desculpe, não consegui entender sua preferência. Você gostaria que eu buscasse opções de profissionais para você, ou você prefere nomear um específico?"
            return {
                "response_to_user": reprompt_message,
                "scheduling_step": "CLASSIFYING_PROFESSIONAL_PREFERENCE", # Volta para tentar classificar novamente
                "current_operation": "SCHEDULING",
                "professional_preference_type": None, # Limpa
                "user_provided_professional_name": None # Limpa
            }

        # Atualiza o estado com a preferência classificada
        update_dict = {
            "professional_preference_type": preference_type,
            "user_provided_professional_name": extracted_name if preference_type == "SPECIFIC_NAME_PROVIDED" else None,
            "scheduling_step": "PROCESSING_PROFESSIONAL_LOGIC", # Próximo passo genérico para lógica de profissional
            "current_operation": "SCHEDULING",
            "response_to_user": None # O próximo nó (PROCESSING_PROFESSIONAL_LOGIC) gerará a resposta
        }
        
        # Se a preferência for clara e o próximo passo for direto (como pedir o nome se "SPECIFIC_NAME_TO_PROVIDE_LATER"),
        # o nó PROCESSING_PROFESSIONAL_LOGIC cuidará da mensagem.
        # Por enquanto, apenas preparamos o estado.
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
            # Limpeza adicional do nome retornado pelo LLM
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
                # Comparar o nome limpo do LLM com o nome da API
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
        # Remoção da lógica duplicada já foi feita anteriormente.

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

    # Se tivermos um nome de profissional, usamos. Senão, a especialidade.
    professional_for_prompt = professional_name if professional_name else f"um profissional de {specialty_name}"

    prompt_messages = REQUEST_TURN_PREFERENCE_PROMPT_TEMPLATE.format_messages(
        professional_name_or_specialty_based=professional_for_prompt
    )
    ai_response = llm_client.invoke(prompt_messages)
    pergunta_turno = ai_response.content.strip()
    logger.info(f"Pergunta sobre turno gerada: '{pergunta_turno}'")

    return {
        "response_to_user": pergunta_turno,
        "scheduling_step": "VALIDATING_TURN_PREFERENCE", # Próximo passo
        "current_operation": "SCHEDULING"
    }

def coletar_validar_turno_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
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
    Considere variações comuns de escrita (com ou sem acento, maiúsculas/minúsculas) e normalize para a categoria correta.

    Exemplos:
    - Usuário: "manhã" -> MANHA
    - Usuário: "manha" -> MANHA
    - Usuário: "Manhã" -> MANHA
    - Usuário: "MANHA" -> MANHA
    - Usuário: "de manhã, por favor" -> MANHA
    - Usuário: "pela manhã" -> MANHA
    - Usuário: "pode ser de tarde" -> TARDE
    - Usuário: "tarde" -> TARDE
    - Usuário: "Tarde" -> TARDE
    - Usuário: "à tarde" -> TARDE
    - Usuário: "no período da tarde" -> TARDE
    - Usuário: "tanto faz" -> INVALIDO
    - Usuário: "qualquer um" -> INVALIDO
    - Usuário: "sim" -> INVALIDO
    - Usuário: "gostaria" -> INVALIDO
    - Usuário: "não sei" -> INVALIDO

    Retorne APENAS a categoria ("MANHA", "TARDE", ou "INVALIDO").

    Resposta do usuário: "{user_response}"
    Categoria:
    """
    prompt_template_turno = ChatPromptTemplate.from_template(prompt_classificacao_turno_str)
    
    llm_classification_response_str = "INVALIDO" # Default
    try:
        chain_turno = prompt_template_turno | llm_client
        llm_response = chain_turno.invoke({"user_response": user_response_content}).content.strip().upper()
        logger.info(f"LLM classificou o turno como: '{llm_response}' para o input '{user_response_content}'")
        if llm_response in ["MANHA", "TARDE"]:
            llm_classification_response_str = llm_response

    except Exception as e:
        logger.error(f"Erro ao invocar LLM para classificação de turno: {e}", exc_info=True)
        # Não retorna aqui, permite a verificação manual abaixo como fallback

    # Verificação manual como fallback ou reforço, especialmente para casos simples
    normalized_user_response = user_response_content.strip().lower()
    if llm_classification_response_str == "INVALIDO": # Só tenta manual se LLM falhou ou retornou inválido
        if normalized_user_response == "manha" or "manhã" in normalized_user_response :
            logger.info(f"Normalização manual/LLM fallback: '{user_response_content}' -> MANHA")
            llm_classification_response_str = "MANHA"
        elif normalized_user_response == "tarde" or "tarde" in normalized_user_response:
            logger.info(f"Normalização manual/LLM fallback: '{user_response_content}' -> TARDE")
            llm_classification_response_str = "TARDE"

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
    else: # Se ainda inválido após LLM e tentativa manual
        logger.warning(f"LLM e normalização manual falharam em classificar o turno para input '{user_response_content}' (Classificação final: {llm_classification_response_str})")
        professional_for_prompt = state.get("user_chosen_professional_name", f"um profissional de {state.get('user_chosen_specialty', 'a especialidade')}")
        reprompt_messages = REQUEST_TURN_PREFERENCE_PROMPT_TEMPLATE.format_messages(
            professional_name_or_specialty_based=professional_for_prompt
        )
        ai_reprompt_response = llm_client.invoke(reprompt_messages)
        reprompt_text = "Desculpe, não entendi bem sua preferência de turno. " + ai_reprompt_response.content.strip()

        return {
            "response_to_user": reprompt_text,
            "scheduling_step": "VALIDATING_TURN_PREFERENCE",
            "current_operation": "SCHEDULING",
            "user_chosen_turn": None
        }

def list_available_professionals_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Busca profissionais disponíveis para a especialidade escolhida e os apresenta.
    """
    logger.debug("--- Nó Agendamento: list_available_professionals_node ---")
    
    API_TOKEN = "TXH3xnwI7P4Sh5dS71aRDsQzDrx1GKeW8jXd5eHDpqTOi" # Exemplo - NÃO USE EM PRODUÇÃO
    api_headers = {
        "Authorization": f"{API_TOKEN}",
        "Content-Type": "application/json"
    }

    specialty_id = state.get("user_chosen_specialty_id")
    specialty_name = state.get("user_chosen_specialty", "a especialidade escolhida")
    user_full_name = state.get("user_full_name", "Paciente")

    if not specialty_id:
        logger.error(f"ID da especialidade '{specialty_name}' não encontrado no estado para listar profissionais.")
        # Gerar mensagem de erro com LLM
        error_prompt = ChatPromptTemplate.from_template(
            "Você é um assistente de agendamento. Informe ao usuário {user_name} que ocorreu um problema "
            "ao tentar identificar a especialidade para buscar os profissionais e que será necessário tentar novamente a seleção da especialidade."
        )
        error_response_content = llm_client.invoke(error_prompt.format_messages(user_name=user_full_name)).content.strip()
        return {
            "response_to_user": error_response_content,
            "scheduling_step": "VALIDATING_SPECIALTY", # Volta para escolher especialidade
            "current_operation": "SCHEDULING",
            "user_chosen_specialty_id": None,
            "available_professionals_list": None
        }

    url = "https://back.homologacao.apphealth.com.br:9090/api-vizi/profissionais"
    params = {"especialidadeId": specialty_id, "status": "true"} # Busca profissionais ativos
    logger.info(f"Consultando API de profissionais: {url} com params: {params}")

    try:
        response = requests.get(url, headers=api_headers, params=params, timeout=10)
        response.raise_for_status()
        professionals_api_data = response.json() # Espera-se List[Dict] com id, nome, etc.

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
                "scheduling_step": "REQUESTING_SPECIALTY", # Ou REQUESTING_PROFESSIONAL_PREFERENCE
                "current_operation": "SCHEDULING",
                "available_professionals_list": []
            }

        # Limitar a quantidade de profissionais apresentados, e extrair apenas id e nome para o estado e prompt
        # Guardar a lista completa com IDs e nomes para validação posterior
        simplified_professionals_list = []
        for prof in professionals_api_data:
            if prof.get("id") and prof.get("nome"):
                simplified_professionals_list.append({"id": prof.get("id"), "nome": prof.get("nome")})
        
        if not simplified_professionals_list:
             logger.warning(f"Profissionais encontrados para {specialty_name}, mas sem ID/Nome válidos.")
             # Tratar como se não tivesse encontrado
             return {
                "response_to_user": f"Desculpe, {user_full_name}, encontrei registros para {specialty_name} mas estou com dificuldade para obter os nomes. Gostaria de tentar outra especialidade?",
                "scheduling_step": "REQUESTING_SPECIALTY",
                "current_operation": "SCHEDULING",
                "available_professionals_list": []
            }


        # Apresentar até N profissionais (ex: 5)
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
        Se houver mais profissionais do que os listados (total_found > shown_count), mencione isso brevemente.
        Sua resposta:
        """
        present_professionals_prompt = ChatPromptTemplate.from_template(present_professionals_prompt_str)

        total_found = len(simplified_professionals_list)
        shown_count = len(professionals_to_show)
        
        additional_info = ""
        if total_found > shown_count:
            additional_info = f" (e mais {total_found - shown_count} outros)"


        presentation_message = llm_client.invoke(
            present_professionals_prompt.format_messages(
                user_name=user_full_name,
                specialty_name=specialty_name,
                list_of_professional_names_str=names_list_for_prompt.strip() + additional_info
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
    user_response_content = get_last_user_message_content(state["messages"])
    
    # Lista de profissionais que foi apresentada (deve conter ID e Nome)
    professionals_shown_list = state.get("available_professionals_list", []) 
    user_full_name = state.get("user_full_name", "Paciente")

    if not user_response_content:
        logger.warning("Nenhuma resposta do usuário para validar escolha do profissional.")
        # Gerar reprompt com LLM para pedir a escolha novamente
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
        # Erro interno, voltar para listagem
        return {
            "response_to_user": f"Desculpe, {user_full_name}, ocorreu um problema e não consigo ver a lista de profissionais que apresentei. Vamos tentar listar novamente.",
            "scheduling_step": "LISTING_AVAILABLE_PROFESSIONALS",
            "current_operation": "SCHEDULING"
        }

    chosen_prof_id = None
    chosen_prof_name = None

    # Tentar por número da opção
    try:
        choice_num = int(user_response_content)
        if 1 <= choice_num <= len(professionals_shown_list):
            selected = professionals_shown_list[choice_num - 1]
            chosen_prof_id = selected.get("id")
            chosen_prof_name = selected.get("nome")
    except ValueError:
        # Não é um número, tentar por correspondência de nome (simples, pode ser melhorado com LLM/fuzzy)
        for prof_data in professionals_shown_list:
            if prof_data.get("nome") and user_response_content.lower() in prof_data["nome"].lower():
                if chosen_prof_id: # Ambiguidade se mais de um corresponder
                    logger.warning(f"Escolha ambígua de profissional por nome: '{user_response_content}'")
                    # Usar LLM para gerar mensagem de ambiguidade
                    ambiguous_prompt = ChatPromptTemplate.from_template(
                        "Você é um assistente de agendamento. O usuário {user_name} forneceu um nome de profissional '{user_input}' que é ambíguo na lista. "
                        "Peça para ele ser mais específico ou usar o número da opção."
                    ).format_messages(user_name=user_full_name, user_input=user_response_content)
                    ambiguous_response = llm_client.invoke(ambiguous_prompt).content.strip()
                    return {
                        "response_to_user": ambiguous_response,
                        "scheduling_step": "VALIDATING_CHOSEN_PROFESSIONAL_FROM_LIST"
                    }
                chosen_prof_id = prof_data.get("id")
                chosen_prof_name = prof_data.get("nome")

    if chosen_prof_id and chosen_prof_name:
        logger.info(f"Usuário escolheu o profissional: ID={chosen_prof_id}, Nome='{chosen_prof_name}'")
        return {
            "user_chosen_professional_id": chosen_prof_id,
            "user_chosen_professional_name": chosen_prof_name,
            "response_to_user": None, # Próximo nó (solicitar turno) fará a pergunta
            "scheduling_step": "REQUESTING_TURN_PREFERENCE",
            "current_operation": "SCHEDULING",
            "available_professionals_list": None # Limpa a lista do estado
        }
    else:
        logger.warning(f"Escolha do profissional '{user_response_content}' não validada.")
        # Usar LLM para gerar mensagem de "escolha inválida"
        invalid_choice_prompt_str = """
        Você é um assistente de agendamento. O usuário {user_name} fez uma escolha de profissional ('{user_input}') que não corresponde
        a nenhuma das opções apresentadas anteriormente.
        Reapresente as opções de forma concisa (sem listar tudo de novo, apenas lembre que ele precisa escolher da lista)
        e peça para ele tentar novamente informando o nome completo ou o número da opção.
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
        return {
            "response_to_user": invalid_response,
            "scheduling_step": "VALIDATING_CHOSEN_PROFESSIONAL_FROM_LIST",
            "current_operation": "SCHEDULING"
        }

def collect_validate_chosen_date_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    logger.debug("--- Nó Agendamento: collect_validate_chosen_date_node ---")
    user_response_content = get_last_user_message_content(state["messages"])
    
    # Datas que foram apresentadas ao usuário, formato "AAAA-MM-DD"
    available_dates_api_format = state.get("available_dates_presented", [])
    user_full_name = state.get("user_full_name", "Paciente")

    if not user_response_content:
        logger.warning("Nenhuma resposta do usuário para validar data escolhida.")
        # Gerar reprompt com LLM para pedir a escolha novamente
        # (Poderia ser um prompt simples ou um que re-liste as opções de forma concisa)
        return {
            "response_to_user": "Por favor, escolha uma das datas apresentadas.",
            "scheduling_step": "VALIDATING_CHOSEN_DATE",
            "current_operation": "SCHEDULING"
        }

    if not available_dates_api_format:
        logger.error("'available_dates_presented' não encontrado no estado. Não é possível validar a escolha.")
        return {
            "response_to_user": f"Desculpe, {user_full_name}, ocorreu um problema e não consigo ver as opções de data que apresentei. Vamos tentar buscar as datas novamente.",
            "scheduling_step": "FETCHING_AVAILABLE_DATES", # Volta para buscar datas
            "current_operation": "SCHEDULING"
        }

    # Preparar strings para o prompt do LLM
    # Lista para display: "1. DD/MM/YYYY\n2. DD/MM/YYYY ..."
    date_options_display_list = []
    for i, date_str_api in enumerate(available_dates_api_format):
        try:
            dt_obj = datetime.strptime(date_str_api, "%Y-%m-%d")
            date_options_display_list.append(f"{i+1}. {dt_obj.strftime('%d/%m/%Y')}")
        except ValueError:
            logger.warning(f"Data inválida em available_dates_presented: {date_str_api}")
            # Ignorar esta data para o display se estiver mal formatada (improvável se veio da API correta)

    date_options_display_list_str = "\n".join(date_options_display_list)
    date_options_internal_list_str = ", ".join(available_dates_api_format)

    logger.info(f"Validando escolha de data: '{user_response_content}' contra opções (API format): {available_dates_api_format}")

    chosen_date_api_format = "NENHUMA_CORRESPONDENCIA_OU_AMBIGUA" # Default
    try:
        prompt_messages = VALIDATE_CHOSEN_DATE_PROMPT_TEMPLATE.format_messages(
            date_options_display_list_str=date_options_display_list_str,
            user_response=user_response_content,
            date_options_internal_list_str=date_options_internal_list_str
        )
        llm_response = llm_client.invoke(prompt_messages).content.strip()
        logger.info(f"LLM para validação de data retornou: '{llm_response}'")
        
        # Verificar se o LLM retornou uma data válida que está na nossa lista original
        if llm_response in available_dates_api_format:
            chosen_date_api_format = llm_response
        else:
            # Se o LLM não retornou uma data válida da lista, mantemos como não correspondência.
            # Poderíamos adicionar lógica para tentar entender "primeira", "segunda" etc. aqui
            # ou confiar que o LLM no prompt já faz isso.
            # Para este exemplo, confiamos no LLM.
            logger.warning(f"Resposta do LLM '{llm_response}' não é uma das datas válidas apresentadas: {available_dates_api_format}")
            chosen_date_api_format = "NENHUMA_CORRESPONDENCIA_OU_AMBIGUA"


    except Exception as e:
        logger.error(f"Erro ao invocar LLM para validar data escolhida: {e}", exc_info=True)
        # Mantém chosen_date_api_format como "NENHUMA_CORRESPONDENCIA_OU_AMBIGUA" para o reprompt

    if chosen_date_api_format != "NENHUMA_CORRESPONDENCIA_OU_AMBIGUA":
        logger.info(f"Usuário escolheu a data: {chosen_date_api_format}")
        # TODO: Próximo passo é buscar horários para esta data.
        # Por enquanto, vamos confirmar a data e finalizar esta parte.
        # A mensagem de confirmação pode ser gerada pelo próximo nó (de busca de horários).
        
        # Aqui você chamaria a API para buscar horários para chosen_date_api_format
        # e depois formataria a mensagem para o usuário.
        # Por enquanto, vamos simular que o próximo passo é pedir para o usuário confirmar.

        # Exemplo de mensagem que o próximo nó de horários poderia gerar:
        # msg_para_horarios = f"Ótimo! Para a data {datetime.strptime(chosen_date_api_format, '%Y-%m-%d').strftime('%d/%m/%Y')}, estou buscando os horários disponíveis..."
        
        return {
            "user_chosen_date": chosen_date_api_format,
            "response_to_user": None, # O nó de busca de horários gerará a próxima resposta
            "scheduling_step": "FETCHING_AVAILABLE_TIMES", # Próximo passo LÓGICO
            "current_operation": "SCHEDULING",
            "available_dates_presented": None # Limpa as datas apresentadas do estado
        }
    else:
        logger.warning(f"Escolha da data '{user_response_content}' não pôde ser validada ou foi ambígua.")
        # Gerar mensagem de reprompt usando LLM, incluindo as opções novamente.
        # Poderíamos ter um prompt específico para reprompt de data.
        reprompt_text = (
            f"Desculpe, {user_full_name}, não consegui identificar qual data você escolheu a partir de '{user_response_content}'.\n"
            f"As opções eram:\n{date_options_display_list_str}\n"
            "Poderia, por favor, informar o número da opção ou a data completa (dd/mm/aaaa)?"
        )
        return {
            "response_to_user": reprompt_text,
            "scheduling_step": "VALIDATING_CHOSEN_DATE", # Tenta novamente
            "current_operation": "SCHEDULING"
            # Mantém available_dates_presented para o próximo ciclo de validação
        }

def fetch_and_present_available_times_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    logger.info(f"--- Nó: fetch_and_present_available_times_node (Session ID: {state.get('session_id', 'N/A')}) ---")
    user_full_name = state.get("user_full_name", "Usuário")
    professional_id = state.get("user_chosen_professional_id")
    chosen_date_str = state.get("user_chosen_date") # Esperado no formato YYYY-MM-DD
    chosen_turn = state.get("user_chosen_turn") # "MANHA" ou "TARDE"
    professional_name = state.get("user_chosen_professional_name", "o profissional escolhido")

    if not all([professional_id, chosen_date_str, chosen_turn]):
        logger.error(f"Dados insuficientes para buscar horários: professional_id={professional_id}, chosen_date={chosen_date_str}, chosen_turn={chosen_turn}")
        return {
            "response_to_user": "Desculpe, não consegui obter todas as informações necessárias (profissional, data ou turno) para buscar os horários. Poderia tentar novamente?",
            "scheduling_step": "SCHEDULING_ERROR" # Ou um passo de fallback apropriado
        }

    api_url = f"https://back.homologacao.apphealth.com.br:9090/api-vizi/agenda/profissionais/{professional_id}/horarios"
    headers = {"Authorization": settings.APPHEALTH_API_TOKEN} # Usando settings
    params = {"data": chosen_date_str}

    response_to_user = ""
    next_scheduling_step = "AWAITING_TIME_CHOICE"
    # Agora available_times_presented_for_state será uma lista de dicionários
    available_times_presented_for_state: List[Dict[str, str]] = []


    try:
        logger.info(f"Chamando API de horários: GET {api_url} com params: {params}")
        api_response = requests.get(api_url, headers=headers, params=params, timeout=10)
        api_response.raise_for_status()
        available_slots_from_api = api_response.json() # Lista de dicts com horaInicio, horaFim, etc.
        logger.info(f"API de horários retornou {len(available_slots_from_api)} slots.")

        if not available_slots_from_api:
            response_to_user = f"Desculpe, {user_full_name}, parece que não há horários disponíveis para {professional_name} no dia {datetime.strptime(chosen_date_str, '%Y-%m-%d').strftime('%d/%m/%Y')} ({chosen_turn.lower()}). Gostaria de tentar outra data ou turno?"
            next_scheduling_step = "FETCHING_AVAILABLE_DATES" # Volta para escolher data
        else:
            # Filtrar por turno e preparar dados para o estado e display
            filtered_slots_details = []
            for slot in available_slots_from_api:
                if slot.get("horaInicio") and slot.get("horaFim"):
                    try:
                        hora_inicio_obj = datetime.strptime(slot["horaInicio"], "%H:%M:%S").time()
                        
                        # Checar turno
                        if (chosen_turn == "MANHA" and hora_inicio_obj < time(12, 0)) or \
                           (chosen_turn == "TARDE" and hora_inicio_obj >= time(12, 0)):
                            
                            filtered_slots_details.append({
                                "display": slot["horaInicio"][:5],      # HH:MM para mostrar ao usuário
                                "horaInicio_api": slot["horaInicio"],   # HH:MM:SS para API
                                "horaFim_api": slot["horaFim"]          # HH:MM:SS para API
                            })
                    except ValueError as ve:
                        logger.warning(f"Slot de horário com formato de hora inválido da API: {slot}. Erro: {ve}")
                        continue # Pula este slot
            
            # Remover duplicatas baseadas no horário de display (HH:MM) e pegar no máximo X horários
            # Isso evita mostrar "08:00" duas vezes se por acaso a API retornar com segundos diferentes mas o mesmo HH:MM
            unique_filtered_slots_temp = {}
            for slot_detail in filtered_slots_details:
                if slot_detail["display"] not in unique_filtered_slots_temp:
                    unique_filtered_slots_temp[slot_detail["display"]] = slot_detail
            
            # Ordenar pelos horários de display antes de fatiar
            # Convertendo para datetime para ordenar corretamente (ex: "13:00" antes de "08:00" se não for feito)
            sorted_unique_slots = sorted(
                list(unique_filtered_slots_temp.values()), 
                key=lambda x: datetime.strptime(x["display"], "%H:%M").time()
            )

            if not sorted_unique_slots:
                 response_to_user = f"Desculpe, {user_full_name}, após filtrar pelo turno da {chosen_turn.lower()}, não encontramos horários disponíveis para {professional_name} no dia {datetime.strptime(chosen_date_str, '%Y-%m-%d').strftime('%d/%m/%Y')}. Gostaria de tentar outro turno ou data?"
                 next_scheduling_step = "VALIDATING_TURN_PREFERENCE" # Volta para escolher turno ou data
            else:
                available_times_presented_for_state = sorted_unique_slots[:3] # Pega os 3 primeiros
                
                if not available_times_presented_for_state:
                    response_to_user = f"Desculpe, {user_full_name}, não consegui encontrar horários para o turno da {chosen_turn.lower()} no dia {datetime.strptime(chosen_date_str, '%Y-%m-%d').strftime('%d/%m/%Y')} para {professional_name}. Gostaria de tentar outra data ou turno?"
                    next_scheduling_step = "VALIDATING_TURN_PREFERENCE"
                else:
                    # Apenas os horários de início (display) para a lista do prompt
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
    
    # available_times_presented agora é uma lista de dicts: [{'display': 'HH:MM', 'horaInicio_api': 'HH:MM:SS', 'horaFim_api': 'HH:MM:SS'}, ...]
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

    # Preparar listas para o prompt do LLM
    # time_options_display_list_str é a lista numerada "1. HH:MM\n2. HH:MM..."
    time_options_display_list_str = "\n".join([
        f"{i+1}. {slot_details['display']}" for i, slot_details in enumerate(available_times_details_list)
    ])
    # time_options_internal_list_str é uma string dos horários de display "HH:MM, HH:MM, ..."
    time_options_internal_list_str = ", ".join([
        slot_details['display'] for slot_details in available_times_details_list
    ])

    logger.info(f"Validando escolha de horário: '{user_response_content}' contra opções de display: {[s['display'] for s in available_times_details_list]}")

    chosen_time_display_from_llm = "NENHUMA_CORRESPONDENCIA_OU_AMBIGUA"
    try:
        prompt_messages = VALIDATE_CHOSEN_TIME_PROMPT_TEMPLATE.format_messages(
            time_options_display_list_str=time_options_display_list_str,
            user_response=user_response_content,
            time_options_internal_list_str=time_options_internal_list_str # LLM valida contra o formato HH:MM
        )
        llm_response = llm_client.invoke(prompt_messages)
        chosen_time_display_from_llm = llm_response.content.strip() # Espera-se HH:MM
        logger.info(f"LLM para validação de horário (display HH:MM) retornou: '{chosen_time_display_from_llm}'")
    except Exception as e:
        logger.error(f"Erro ao invocar LLM para validar horário escolhido: {e}", exc_info=True)
    
    # Agora, encontrar o slot completo (com horaInicio_api e horaFim_api) correspondente ao horário de display validado pelo LLM
    selected_slot_details: Optional[Dict[str, str]] = None
    if chosen_time_display_from_llm != "NENHUMA_CORRESPONDENCIA_OU_AMBIGUA":
        for slot_details in available_times_details_list:
            if slot_details['display'] == chosen_time_display_from_llm:
                selected_slot_details = slot_details
                break
    
    if selected_slot_details:
        user_chosen_time_hhmm = selected_slot_details['display'] # HH:MM
        user_chosen_time_fim_api = selected_slot_details['horaFim_api'] # HH:MM:SS
        
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
                chosen_time=user_chosen_time_hhmm # Mostrar HH:MM para o usuário
            )
            confirmation_response = llm_client.invoke(confirmation_prompt_messages)
            response_text_for_user = confirmation_response.content.strip()
        except Exception as e:
            logger.error(f"Erro ao gerar mensagem de confirmação final: {e}")
            response_text_for_user = f"Agendamento para {chosen_specialty} com {chosen_professional_name} no dia {chosen_date_display_for_confirmation} às {user_chosen_time_hhmm}. Confirmar?"

        return {
            "user_chosen_time": user_chosen_time_hhmm, # Salva HH:MM (o que o usuário escolheu e viu)
            "user_chosen_time_fim": user_chosen_time_fim_api, # Salva HH:MM:SS (para a API de agendamento)
            "response_to_user": response_text_for_user,
            "scheduling_step": "AWAITING_FINAL_CONFIRMATION", 
            "current_operation": "SCHEDULING", 
            "messages": AIMessage(content=response_text_for_user),
            "available_times_presented": None # Limpa a lista do estado após a escolha
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
            # Mantém available_times_presented para a próxima tentativa
        }

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
            professional_for_prompt = state.get("user_chosen_professional_name", f"um profissional de {state.get('user_chosen_specialty', 'a especialidade')}")
            reprompt_messages = REQUEST_TURN_PREFERENCE_PROMPT_TEMPLATE.format_messages(
                professional_name_or_specialty_based=professional_for_prompt
            )
            ai_reprompt_response = llm_client.invoke(reprompt_messages)
            reprompt_text = "Desculpe, não entendi bem. " + ai_reprompt_response.content.strip()

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
    
    # TODO: Obter API_TOKEN e HEADERS de forma segura (ex: config, secrets)
    # Por enquanto, vamos simular, mas em produção isso precisa ser robusto.
    # Substitua pelo seu token real e mecanismo de obtenção de headers
    API_TOKEN = "TXH3xnwI7P4Sh5dS71aRDsQzDrx1GKeW8jXd5eHDpqTOi" # Exemplo - NÃO USE EM PRODUÇÃO
    api_headers = {
        "Authorization": f"{API_TOKEN}",
        "Content-Type": "application/json"
    }

    id_profissional = state.get("user_chosen_professional_id")
    nome_profissional = state.get("user_chosen_professional_name", "O profissional selecionado")
    user_full_name = state.get("user_full_name", "Paciente")
    user_chosen_turn = state.get("user_chosen_turn", "qualquer turno") # Para a mensagem

    if not id_profissional:
        logger.error("ID do profissional não encontrado no estado para buscar datas.")
        # Gerar mensagem de erro com LLM para ser mais natural
        error_prompt = ChatPromptTemplate.from_template(
            "Você é um assistente de agendamento. Informe ao usuário {user_name} que ocorreu um problema "
            "ao tentar identificar o profissional para buscar as datas e que será necessário tentar novamente a seleção do profissional."
        )
        error_response_content = llm_client.invoke(error_prompt.format_messages(user_name=user_full_name)).content.strip()
        return {
            "response_to_user": error_response_content,
            "scheduling_step": "REQUESTING_PROFESSIONAL_PREFERENCE", # Volta para escolher profissional
            "current_operation": "SCHEDULING",
            "user_chosen_professional_id": None, # Limpa
            "user_chosen_professional_name": None # Limpa
        }

    hoje = date.today()
    datas_validas_api_format = [] # Lista para armazenar "AAAA-MM-DD"

    # Consultar para o mês atual e o próximo para garantir opções (como no agentv1.py)
    for i in range(2): # Consultar 2 meses
        data_alvo_consulta = hoje + timedelta(days=30 * i)
        mes_consulta = data_alvo_consulta.strftime("%m")
        ano_consulta = data_alvo_consulta.strftime("%Y")
        
        # URL da API de datas (ajuste se necessário)
        url = f"https://back.homologacao.apphealth.com.br:9090/api-vizi/agenda/profissionais/{id_profissional}/datas"
        params = {"mes": mes_consulta, "ano": ano_consulta}
        logger.info(f"Consultando API de datas: {url} com params: {params}")

        try:
            # Aqui usaremos 'requests' diretamente como na sua PoC.
            # Em uma arquitetura DDD completa, isso estaria em um serviço de infraestrutura.
            import requests # Adicionar importação no topo do arquivo se não estiver
            response = requests.get(url, headers=api_headers, params=params, timeout=10)
            response.raise_for_status() # Levanta exceção para erros HTTP 4xx/5xx
            datas_mes_api = response.json() # Espera-se uma lista de dicts como [{"data": "YYYY-MM-DD"}, ...]

            if datas_mes_api and isinstance(datas_mes_api, list):
                for item_data in datas_mes_api:
                    if isinstance(item_data, dict) and "data" in item_data:
                        # Validação básica do formato da data (AAAA-MM-DD)
                        try:
                            datetime.strptime(item_data["data"], "%Y-%m-%d")
                            datas_validas_api_format.append(item_data["data"])
                        except ValueError:
                            logger.warning(f"Formato de data inválido da API: {item_data['data']}")
            
            # Remover duplicatas e ordenar
            datas_validas_api_format = sorted(list(set(datas_validas_api_format)))
            
            # Filtrar datas para serem a partir de hoje (ou D+N, conforme regra de negócio)
            # A API já deveria fazer isso, mas uma checagem extra pode ser útil.
            # Por simplicidade, vamos confiar que a API retorna apenas datas futuras válidas.

        except requests.exceptions.HTTPError as e:
            logger.error(f"Erro HTTP ao consultar API de datas: {e.response.status_code} - {e.response.text if e.response else 'Sem corpo de resposta'}")
            # Tentar gerar mensagem de erro amigável com LLM
            error_prompt_http = ChatPromptTemplate.from_template(
                "Você é um assistente de agendamento. Informe ao usuário {user_name} que houve um problema técnico "
                "(código de erro {status_code}) ao tentar buscar as datas disponíveis e peça para tentar mais tarde."
            )
            error_response_content = llm_client.invoke(error_prompt_http.format_messages(user_name=user_full_name, status_code=e.response.status_code if e.response else "desconhecido")).content.strip()
            return {
                "response_to_user": error_response_content,
                "scheduling_step": "REQUESTING_TURN_PREFERENCE", # Volta para pedir turno, talvez o usuário queira mudar algo
                "current_operation": "SCHEDULING"
            }
        except requests.exceptions.RequestException as e: # Outros erros (conexão, timeout)
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
        except ValueError as json_err: # Erro ao decodificar JSON
            logger.error(f"Erro ao decodificar JSON da API de datas: {json_err}")
            # Mensagem genérica para o usuário
            return {
                "response_to_user": "Desculpe, {user_name}, recebi uma resposta inesperada do sistema de agendamento ao buscar as datas. Por favor, tente novamente em alguns instantes.",
                "scheduling_step": "REQUESTING_TURN_PREFERENCE",
                "current_operation": "SCHEDULING"
            }


        if len(datas_validas_api_format) >= 3: # Se já temos 3 ou mais, paramos
            break
            
    if not datas_validas_api_format:
        logger.info(f"Nenhuma data disponível encontrada para Dr(a). {nome_profissional} nos próximos períodos.")
        # Usar LLM para gerar a mensagem de "nenhuma data"
        no_dates_prompt = ChatPromptTemplate.from_template(
            "Você é um assistente de agendamento. Informe ao usuário {user_name} que, no momento, não foram encontradas datas disponíveis "
            "para o profissional {professional_name}. Pergunte se ele gostaria de tentar com outro profissional ou especialidade, ou verificar mais tarde."
        )
        no_dates_response = llm_client.invoke(no_dates_prompt.format_messages(user_name=user_full_name, professional_name=nome_profissional)).content.strip()
        return {
            "response_to_user": no_dates_response,
            "scheduling_step": "REQUESTING_PROFESSIONAL_PREFERENCE", # Volta para escolher profissional/especialidade
            "current_operation": "SCHEDULING",
            "available_dates_presented": []
        }

    # Apresentar até 3 datas
    datas_para_apresentar_api_format = datas_validas_api_format[:3]
    
    # Formatar para o usuário (DD/MM/AAAA)
    datas_formatadas_usuario = [datetime.strptime(d, "%Y-%m-%d").strftime("%d/%m/%Y") for d in datas_para_apresentar_api_format]
    
    # Usar LLM para gerar a mensagem de apresentação das datas
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
    
    # Criar a string da lista de datas para o prompt
    lista_datas_str_prompt = ""
    for i, data_fmt_usr in enumerate(datas_formatadas_usuario):
        lista_datas_str_prompt += f"{i+1}. {data_fmt_usr}\n"
    
    mensagem_apresentacao_datas = llm_client.invoke(
        present_dates_prompt.format_messages(
            user_name=user_full_name,
            chosen_turn=user_chosen_turn.lower(), # "manha" ou "tarde"
            professional_name=nome_profissional,
            available_dates_list_str=lista_datas_str_prompt.strip()
        )
    ).content.strip()

    return {
        "response_to_user": mensagem_apresentacao_datas,
        "scheduling_step": "VALIDATING_CHOSEN_DATE", # Próximo passo lógico
        "current_operation": "SCHEDULING",
        "available_dates_presented": datas_para_apresentar_api_format # Salva no formato AAAA-MM-DD
    }

def process_final_scheduling_confirmation_node(state: MainWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Processa a resposta do usuário à pergunta de confirmação final do agendamento.
    """
    logger.debug("--- Nó Agendamento: process_final_scheduling_confirmation_node ---")
    user_response_content = get_last_user_message_content(state["messages"])
    user_full_name = state.get("user_full_name", "Cliente")

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
            hora_inicio_hhmm_str = state.get("user_chosen_time") # Formato HH:MM
            hora_fim_hhmmss_str = state.get("user_chosen_time_fim") # Deveria ser HH:MM:SS do estado
            nome_paciente = state.get("user_full_name")
            especialidade_id = state.get("user_chosen_specialty_id")

            if not all([profissional_id, data_agendamento, hora_inicio_hhmm_str, hora_fim_hhmmss_str, nome_paciente, especialidade_id]):
                logger.error(f"Dados insuficientes no estado para realizar o agendamento via API. Detalhes: prof_id={profissional_id}, data={data_agendamento}, hora_inicio={hora_inicio_hhmm_str}, hora_fim={hora_fim_hhmmss_str}, nome={nome_paciente}, esp_id={especialidade_id}")
                return {
                    "response_to_user": "Desculpe, ocorreu um problema com os dados do agendamento e não pude confirmar. Por favor, tente refazer o agendamento ou contate o suporte.",
                    "scheduling_completed": False, "current_operation": None, "scheduling_step": None
                }
            
            # Garantir que horaInicio seja HH:MM:SS para a API
            try:
                hora_inicio_api_format = datetime.strptime(hora_inicio_hhmm_str, "%H:%M").strftime("%H:%M:%S")
            except ValueError:
                logger.error(f"Formato de hora_inicio_hhmm_str inválido: {hora_inicio_hhmm_str}")
                # Tratar erro, talvez retornando ao usuário ou um erro genérico
                return {"response_to_user": "Erro no formato da hora de início.", "scheduling_step": "AWAITING_TIME_CHOICE"}


            # Validar/Formatar hora_fim_hhmmss_str se necessário, mas idealmente já está correto
            # Se viesse como HH:MM do estado, precisaria converter para HH:MM:SS aqui também.
            # Como estamos visando que user_chosen_time_fim já seja HH:MM:SS, podemos usá-lo diretamente.
            # Adicionar uma verificação para garantir que não é None ou mal formatado é uma boa prática.
            if not re.match(r'^\d{2}:\d{2}:\d{2}$', hora_fim_hhmmss_str):
                logger.error(f"Formato de hora_fim_hhmmss_str inválido: {hora_fim_hhmmss_str}. Esperado HH:MM:SS.")
                # Fallback ou erro. Por simplicidade, vamos logar e prosseguir, mas idealmente deveria ser tratado.
                # Se a API for estrita, isso pode causar falha.
                # Poderia tentar re-calcular se a duração é sempre fixa, mas é melhor garantir que venha correto.
                # Para este exemplo, vamos assumir que o estado está correto (HH:MM:SS).

            telefone_paciente_placeholder = "00000000000" 
            logger.warning(f"UTILIZANDO TELEFONE PLACEHOLDER '{telefone_paciente_placeholder}' PARA AGENDAMENTO.")
            id_unidade_default = 21641

            payload = {
                "data": data_agendamento,
                "horaInicio": hora_inicio_api_format, 
                "horaFim": hora_fim_hhmmss_str, # Usando o valor do estado
                "nome": nome_paciente,
                "telefonePrincipal": telefone_paciente_placeholder,
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
                
                chosen_specialty = state.get("user_chosen_specialty", "N/A")
                chosen_professional_name = state.get("user_chosen_professional_name", "N/A")
                chosen_date_display = datetime.strptime(data_agendamento, "%Y-%m-%d").strftime("%d/%m/%Y") if data_agendamento else "N/A"
                
                success_message = (
                    f"Ótimo, {user_full_name}! Seu agendamento para {chosen_specialty} com {chosen_professional_name} "
                    f"no dia {chosen_date_display} às {hora_inicio_hhmm_str} foi confirmado com sucesso (ID: {agendamento_id_api}). "
                    "Algo mais em que posso ajudar?"
                )
                return {
                    "response_to_user": success_message, "scheduling_completed": True,
                    "current_operation": None, "scheduling_step": None, 
                    "scheduling_values_confirmed": { 
                        "name": user_full_name, "specialty": chosen_specialty, 
                        "professional": chosen_professional_name, "date": data_agendamento,
                        "time_start": hora_inicio_hhmm_str, "time_end": hora_fim_hhmmss_str, # Log mais detalhado
                        "api_schedule_id": agendamento_id_api
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
            # ... (código de cancelamento permanece o mesmo) ...
            logger.info(f"Agendamento CANCELADO por {user_full_name}.")
            return {
                "response_to_user": "Entendido. O agendamento não foi confirmado. Se precisar de algo mais, é só chamar!",
                "scheduling_completed": False, 
                "current_operation": None, 
                "scheduling_step": None
            }
        else: # AMBIGUOUS
            # ... (código para ambíguo permanece o mesmo) ...
            logger.warning(f"Resposta de confirmação ambígua ou não classificada: '{confirmation_status}'")
            return {
                "response_to_user": "Desculpe, não entendi sua resposta. Para confirmar o agendamento, por favor, diga 'sim'. Se não deseja confirmar, diga 'não'.",
                "scheduling_step": "AWAITING_FINAL_CONFIRMATION", 
                "current_operation": "SCHEDULING"
            }
    except Exception as e:
        # ... (código de exceção geral permanece o mesmo) ...
        logger.error(f"Erro ao processar confirmação final do agendamento: {e}", exc_info=True)
        return {
            "response_to_user": "Desculpe, tive um problema ao processar sua confirmação. Poderia tentar confirmar novamente com 'sim' ou 'não'?",
            "scheduling_step": "AWAITING_FINAL_CONFIRMATION",
            "current_operation": "SCHEDULING"
        }

# === consulta api ===

# <<< ADICIONANDO AS FUNÇÕES DOS NÓS PLACEHOLDER >>>
def placeholder_fallback_node(state: MainWorkflowState) -> dict:
    """Placeholder para intenções fora do escopo ou não tratadas."""
    logger.debug("--- Nó: placeholder_fallback_node ---")
    categoria = state.get("categoria", "desconhecida")
    response_text = f"Sua intenção foi '{categoria}'. No momento, estou aprendendo a lidar com essa solicitação. Posso ajudar com saudações ou com a criação de agendamentos por enquanto."
    return {"response_to_user": response_text, "current_operation": None}

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
    logger.info(f"Roteamento Inicial/Contínuo: Current operation = {current_op}")

    if current_op == "SCHEDULING":
        logger.info("Operação de agendamento em progresso. Roteando para 'route_scheduling_step'.")
        return "route_scheduling_step" # Nova rota para o dispatcher de agendamento
    else: 
        logger.info("Nenhuma operação principal em progresso ou operação concluída. Roteando para 'categorize_intent'.")
        return "categorize_intent"

def route_after_categorization(state: MainWorkflowState) -> str:
    categoria = state.get("categoria")
    logger.info(f"Roteando com base na categoria: '{categoria}'")

    if categoria == "Saudação ou Despedida":
        return "handle_greeting_farewell"
    elif categoria == "Criar Agendamento": 
        # Inicia a operação de agendamento e vai para o primeiro passo
        # O nó de destino (solicitar_nome_agendamento_node) definirá current_operation e scheduling_step
        return "solicitar_nome_agendamento_node" 
    elif categoria == "Erro na Categorização" or categoria == "Indefinido":
        return "handle_fallback_placeholder" # Fallback direto
    else: 
        return "handle_fallback_placeholder" # Fallback para outras categorias

def route_scheduling_step(state: MainWorkflowState) -> str:
    step = state.get("scheduling_step")
    logger.info(f"Roteamento do Agendamento: scheduling_step = {step}")

    if step == "VALIDATING_FULL_NAME":
        return "coletar_validar_nome_agendamento_node"
    elif step == "VALIDATING_SPECIALTY": 
        return "coletar_validar_especialidade_node"
    elif step == "AWAITING_PROFESSIONAL_PREFERENCE":
        return "solicitar_preferencia_profissional_node"
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

# === CONSTRUÇÃO DO GRAFO ===
def get_main_conversation_graph_definition() -> StateGraph:
    logger.info("Definindo a estrutura do grafo principal da conversa (sem subgrafos)")
    workflow_builder = StateGraph(MainWorkflowState)
    llm_instance = get_llm_client()

    # Nós do Grafo Principal
    workflow_builder.add_node("dispatcher", dispatcher_node)
    workflow_builder.add_node("categorize_intent", partial(categorize_node, llm_client=llm_instance))
    workflow_builder.add_node("handle_greeting_farewell", partial(greeting_farewell_node, llm_client=llm_instance))
    workflow_builder.add_node("handle_fallback_placeholder", placeholder_fallback_node)

    # Nós do Fluxo de Agendamento (integrados)
    workflow_builder.add_node("solicitar_nome_agendamento_node", partial(solicitar_nome_agendamento_node, llm_client=llm_instance))
    workflow_builder.add_node("coletar_validar_nome_agendamento_node", partial(coletar_validar_nome_agendamento_node, llm_client=llm_instance))
    workflow_builder.add_node("coletar_validar_especialidade_node", partial(coletar_validar_especialidade_node, llm_client=llm_instance)) 
    workflow_builder.add_node("solicitar_preferencia_profissional_node", partial(solicitar_preferencia_profissional_node, llm_client=llm_instance))
    workflow_builder.add_node("coletar_classificar_preferencia_profissional_node", partial(coletar_classificar_preferencia_profissional_node, llm_client=llm_instance))
    workflow_builder.add_node("processing_professional_logic_node", partial(processing_professional_logic_node, llm_client=llm_instance))

    workflow_builder.add_node("list_available_professionals_node", partial(list_available_professionals_node, llm_client=llm_instance))
    workflow_builder.add_node("collect_validate_chosen_professional_node", partial(collect_validate_chosen_professional_node, llm_client=llm_instance))
    
    # NOVOS NÓS PARA TURNO
    workflow_builder.add_node("solicitar_turno_node", partial(solicitar_turno_node, llm_client=llm_instance))
    workflow_builder.add_node("coletar_validar_turno_node", partial(coletar_validar_turno_node, llm_client=llm_instance))
    workflow_builder.add_node("fetch_and_present_available_dates_node", partial(fetch_and_present_available_dates_node, llm_client=llm_instance))
    workflow_builder.add_node("collect_validate_chosen_date_node", partial(collect_validate_chosen_date_node, llm_client=llm_instance))
    workflow_builder.add_node("fetch_and_present_available_times_node", partial(fetch_and_present_available_times_node, llm_client=llm_instance))
    workflow_builder.add_node("coletar_validar_horario_escolhido_node", partial(coletar_validar_horario_escolhido_node, llm_client=llm_instance))
    workflow_builder.add_node("process_final_scheduling_confirmation_node", partial(process_final_scheduling_confirmation_node, llm_client=llm_instance))


    workflow_builder.set_entry_point("dispatcher")

    # Roteamento a partir do dispatcher inicial
    workflow_builder.add_conditional_edges(
        "dispatcher",
        route_initial_or_ongoing,
        {
            "route_scheduling_step": "route_scheduling_step", 
            "categorize_intent": "categorize_intent"
        }
    )

    # Roteamento após a categorização da intenção
    workflow_builder.add_conditional_edges(
        "categorize_intent",
        route_after_categorization,
        {
            "handle_greeting_farewell": "handle_greeting_farewell",
            "solicitar_nome_agendamento_node": "solicitar_nome_agendamento_node", 
            "handle_fallback_placeholder": "handle_fallback_placeholder"
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
            "collect_validate_chosen_date_node": "collect_validate_chosen_date_node",
            "fetch_and_present_available_times_node": "fetch_and_present_available_times_node",
            "coletar_validar_horario_escolhido_node": "coletar_validar_horario_escolhido_node",
            "process_final_scheduling_confirmation_node": "process_final_scheduling_confirmation_node",
            END: END 
        }
    )

    workflow_builder.add_conditional_edges(
        "processing_professional_logic_node",
        route_after_processing_professional_logic,
        {
            END: END,  # Se response_to_user foi setado por processing_professional_logic_node
            "route_scheduling_step": "route_scheduling_step" # Se for para continuar internamente (ex: para listar profissionais)
        }
    )

    # Saídas dos nós
    workflow_builder.add_edge("handle_greeting_farewell", END)
    workflow_builder.add_edge("handle_fallback_placeholder", END)
    
    # Nós que esperam input do usuário e vão para END (para serem pegos pelo dispatcher na próxima rodada)
    # ou que direcionam para o route_scheduling_step para continuar o fluxo interno.
    workflow_builder.add_edge("solicitar_nome_agendamento_node", END) 
    workflow_builder.add_edge("coletar_validar_nome_agendamento_node", END)
    workflow_builder.add_edge("coletar_validar_especialidade_node", END) # Continua para solicitar preferência
    workflow_builder.add_edge("solicitar_preferencia_profissional_node", END)
    workflow_builder.add_edge("coletar_classificar_preferencia_profissional_node", "route_scheduling_step")

    workflow_builder.add_edge("list_available_professionals_node", END) # Espera escolha do usuário
    workflow_builder.add_edge("collect_validate_chosen_professional_node", "route_scheduling_step") 
    
    # NOVAS ARESTAS PARA TURNO
    workflow_builder.add_edge("solicitar_turno_node", END) # Espera resposta do usuário
    workflow_builder.add_edge("coletar_validar_turno_node", "route_scheduling_step") # Continua para buscar datas
    workflow_builder.add_edge("fetch_and_present_available_dates_node", END)
    workflow_builder.add_edge("collect_validate_chosen_date_node", "route_scheduling_step")
    workflow_builder.add_edge("fetch_and_present_available_times_node", END)
    workflow_builder.add_edge("coletar_validar_horario_escolhido_node", END)
    workflow_builder.add_edge("process_final_scheduling_confirmation_node", END)
    return workflow_builder

# === FUNÇÃO DE EXECUÇÃO DO FLUXO (Permanece a mesma em sua lógica interna) ===
async def arun_main_conversation_flow(
    user_text: str,
    session_id: str,
    checkpointer: BaseCheckpointSaver 
) -> Optional[str]:
    logger.info(f"Executando arun_main_conversation_flow para session_id: {session_id} com texto: '{user_text}'")
    
    if not checkpointer:
        logger.error("ERRO CRÍTICO: checkpointer não foi fornecido para arun_main_conversation_flow.")
        return "Desculpe, estou com problemas técnicos (Configuração de Checkpoint)."

    workflow_definition = get_main_conversation_graph_definition()
    graph_with_persistence = workflow_definition.compile(checkpointer=checkpointer)
    
    final_state: Optional[MainWorkflowState] = None # Tipo explícito
    # A mensagem do usuário é a única nova mensagem a ser processada no início de cada turno.
    # O histórico é gerenciado pelo checkpointer.
    input_data = {"messages": [HumanMessage(content=user_text)]} 
    config = {"configurable": {"thread_id": session_id}}
    
    logger.debug(f"Invocando grafo principal para session_id: {session_id} com input_data: {input_data}")

    try:
        # Usar astream para iterar sobre os chunks de eventos do grafo
        async for event_chunk in graph_with_persistence.astream(input_data, config=config, stream_mode="values"):
            # O último chunk no modo "values" é o estado final do grafo para este passo
            final_state = event_chunk 
            logger.debug(f"Chunk do grafo (session_id {session_id}): {event_chunk}")
    except Exception as graph_error:
        logger.error(f"Erro ao invocar o grafo LangGraph para thread {session_id}: {graph_error}", exc_info=True)
        return "Desculpe, ocorreu um erro interno ao processar sua solicitação."

    # Após o loop, final_state contém o estado completo após a execução do grafo.
    if final_state:
        logger.info(f"Estado final do grafo (session_id {session_id}): {final_state}")
        
        # Recuperar a mensagem de resposta que foi definida pelos nós
        response_content_to_send = final_state.get("response_to_user")

        if response_content_to_send:
            logger.info(f"Resposta do arun_main_conversation_flow (session_id {session_id}): '{response_content_to_send}'")
            return response_content_to_send
        else:
            # Se response_to_user não foi setado, mas há mensagens,
            # pode ser um caso onde o fluxo terminou sem uma resposta explícita para este turno.
            # Verificamos a última mensagem do AI no histórico, mas isso é menos ideal.
            # A boa prática é que todo nó que deve responder ao usuário sete "response_to_user".
            all_messages = final_state.get('messages', [])
            if all_messages:
                last_message_obj = all_messages[-1]
                if isinstance(last_message_obj, AIMessage):
                    logger.warning(f"'response_to_user' não definido, mas última mensagem é AIMessage: '{last_message_obj.content}'")
                    return last_message_obj.content # Não ideal, mas um fallback
            logger.warning(f"Nenhum 'response_to_user' definido no estado final e nenhuma AIMessage clara no final do histórico para session_id {session_id}.")
            return None # Ou uma mensagem de erro padrão
            
    logger.warning(f"Nenhum estado final retornado pelo LangGraph para session_id {session_id}.")
    return None