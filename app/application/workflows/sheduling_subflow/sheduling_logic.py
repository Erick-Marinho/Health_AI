import logging
from typing import Optional
from functools import partial

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .sheduling_state import SchedulingWorkflowState
# Futuramente, importaremos serviços específicos para agendamento aqui
# from app.application.services.scheduling_service import (
#     gerar_pergunta_nome_completo_service,
#     processar_resposta_nome_completo_service
# )
from app.application.prompts.conversation_prompts import REQUEST_FULL_NAME_PROMPT_TEMPLATE
from app.infrastructure.llm_clients import get_llm_client # Para passar ao subgrafo, se necessário

logger = logging.getLogger(__name__)

# === NÓS DO SUBGRAFO DE AGENDAMENTO ===

def internal_scheduling_dispatcher_node(state: SchedulingWorkflowState) -> dict:
    """
    Nó dispatcher INTERNO ao subgrafo de agendamento.
    Decide qual nó executar com base em 'proximo_passo_agendamento'.
    Este nó APENAS lê o estado para roteamento, não o modifica aqui.
    """
    logger.debug(f"--- Subgrafo Dispatcher: Estado recebido: {state} ---")
    proximo_passo = state.get("proximo_passo_agendamento")
    contador_carregado = state.get("contador_teste") # LOGAR O CONTADOR
    logger.info(f"Subgrafo Dispatcher: Próximo passo agendamento = {proximo_passo}, Contador Teste Carregado = {contador_carregado}")
    
    # Simplesmente retorna o estado; o roteamento condicional usará 'proximo_passo_agendamento'.
    # Não precisamos retornar explicitamente 'proximo_passo' no dict, a função de roteamento o acessará do estado.
    return {}

def solicitar_nome_completo_node(state: SchedulingWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Nó inicial do subfluxo de agendamento: solicita o nome completo.
    """
    logger.debug("--- Subgrafo Agendamento: Nó solicitar_nome_completo_node ---")

    # Incrementar o contador de teste
    current_contador = state.get("contador_teste", 0)
    novo_contador = current_contador + 1
    logger.info(f"Contador de teste do subgrafo: {novo_contador}")

    prompt = REQUEST_FULL_NAME_PROMPT_TEMPLATE.format_messages()
    logger.debug(f"Gerando solicitação de nome completo com o LLM. Prompt: {prompt}")
    ai_response = llm_client.invoke(prompt)
    resposta_llm = ai_response.content
    logger.info(f"Resposta do LLM para solicitação de nome completo: {resposta_llm}")

    current_messages = state.get("messages", [])
    updated_messages = current_messages + [AIMessage(content=resposta_llm)]

    return {
        "messages": updated_messages,
        "resposta_para_usuario": resposta_llm,
        "proximo_passo_agendamento": "COLETAR_NOME",
        "agendamento_concluido": False, # Explicitamente não concluído
        "contador_teste": novo_contador
    }
    
def coletar_nome_completo_node(state: SchedulingWorkflowState, llm_client: ChatOpenAI) -> dict:
    """
    Nó para processar a resposta do usuário contendo o nome completo.
    Este nó é chamado DEPOIS que o usuário responde à pergunta do nó anterior.
    A mensagem do usuário estará em state['messages'][-1] (a última da lista DENTRO DESTE SUBGRAFO).
    """
    logger.debug("--- Subgrafo Agendamento: Nó coletar_nome_completo_node ---")
    
    current_messages = state.get("messages", [])
    if not current_messages or not isinstance(current_messages[-1], HumanMessage):
        logger.warning("A última mensagem no subgrafo não é do usuário ou não há mensagens. Isso pode indicar um problema no fluxo ou na primeira passagem.")
        # Em uma situação real, precisaríamos de uma lógica mais robusta aqui.
        return {
            "resposta_para_usuario": "Desculpe, tive um problema ao processar sua resposta. Poderia tentar novamente?",
            "proximo_passo_agendamento": "COLETAR_NOME", # Tenta de novo, mas idealmente, um estado de erro.
            "agendamento_concluido": False
        }

    user_response_content = current_messages[-1].content
    logger.info(f"Nome recebido do usuário para coleta: '{user_response_content}'")

    # TODO: Implementar lógica de validação real do nome (pode usar LLM)
    # nome_validado = processar_resposta_nome_completo_service(user_response_content, llm_client)
    nome_validado = user_response_content # Simulação de validação bem-sucedida

    if nome_validado:
        logger.info(f"Nome validado: {nome_validado}")
        confirmacao = f"Obrigado, {nome_validado}. Agora, por favor, me informe a data e hora que gostaria de agendar."
        updated_messages = current_messages + [AIMessage(content=confirmacao)]
        return {
            "messages": updated_messages,
            "nome_completo_usuario": nome_validado,
            "resposta_para_usuario": confirmacao,
            "proximo_passo_agendamento": "SOLICITAR_DATA_HORA",
            "agendamento_concluido": False # Ainda não concluído
        }
    else:
        logger.warning(f"Não foi possível validar o nome: '{user_response_content}'")
        re_pergunta = "Não consegui identificar um nome válido em sua resposta. Poderia, por gentileza, informar seu nome completo novamente?"
        updated_messages = current_messages + [AIMessage(content=re_pergunta)]
        return {
            "messages": updated_messages,
            "resposta_para_usuario": re_pergunta,
            "proximo_passo_agendamento": "COLETAR_NOME", 
            "agendamento_concluido": False # Ainda não concluído
        }

# === NÓ PLACEHOLDER PARA SOLICITAR DATA E HORA ===

def solicitar_data_hora_node(state: SchedulingWorkflowState, llm_client: ChatOpenAI) -> dict:
    logger.debug("--- Subgrafo Agendamento: Nó solicitar_data_hora_node (Placeholder) ---")
    # Esta mensagem deveria ser gerada por LLM
    pergunta_data_hora = "Qual a melhor data e horário para você?"
    current_messages = state.get("messages", [])
    updated_messages = current_messages + [AIMessage(content=pergunta_data_hora)]
    return {
        "messages": updated_messages,
        "resposta_para_usuario": pergunta_data_hora,
        "proximo_passo_agendamento": "COLETAR_DATA_HORA",
        "agendamento_concluido": False # Ainda não concluído
    }

# === LÓGICA DE ROTEAMENTO DO SUBGRAFO (Exemplo Simples) ===

def route_internal_scheduling_logic(state: SchedulingWorkflowState) -> str:
    proximo_passo = state.get("proximo_passo_agendamento")
    logger.info(f"Subgrafo Roteamento Interno: Próximo passo = {proximo_passo}")

    if proximo_passo == "COLETAR_NOME":
        return "coletar_nome_completo_node"
    elif proximo_passo == "SOLICITAR_DATA_HORA":
        return "solicitar_data_hora_node" # Quando este nó existir de verdade
    # Adicionar mais condições para outros passos (COLETAR_DATA_HORA, CONFIRMAR_AGENDAMENTO, etc.)
    elif proximo_passo is None and not state.get("nome_completo_usuario"): # Se é a primeira vez e não tem próximo passo definido
        return "solicitar_nome_completo_node"
    
    # Se o fluxo chegou ao fim ou a um estado não roteável explicitamente dentro do subgrafo
    logger.info(f"Subgrafo Roteamento Interno: Nenhum nó específico para '{proximo_passo}'. Terminando subgrafo.")
    return END

# === CONSTRUÇÃO DO SUBGRAFO DE AGENDAMENTO ===

def get_scheduling_subgraph_definition(llm_for_nodes: ChatOpenAI) -> StateGraph:
    logger.info("Definindo a estrutura do subgrafo de agendamento com dispatcher interno")
    
    subgraph_builder = StateGraph(SchedulingWorkflowState)

    subgraph_builder.add_node("internal_dispatcher", internal_scheduling_dispatcher_node) # Não precisa de LLM
    subgraph_builder.add_node(
        "solicitar_nome_completo_node", 
        partial(solicitar_nome_completo_node, llm_client=llm_for_nodes)
    )
    subgraph_builder.add_node(
        "coletar_nome_completo_node",
        partial(coletar_nome_completo_node, llm_client=llm_for_nodes)
    )
    subgraph_builder.add_node( # Placeholder
        "solicitar_data_hora_node",
        partial(solicitar_data_hora_node, llm_client=llm_for_nodes)
    )

    subgraph_builder.set_entry_point("internal_dispatcher")

    subgraph_builder.add_conditional_edges(
        "internal_dispatcher",
        route_internal_scheduling_logic,
        {
            "solicitar_nome_completo_node": "solicitar_nome_completo_node",
            "coletar_nome_completo_node": "coletar_nome_completo_node",
            "solicitar_data_hora_node": "solicitar_data_hora_node", # Adicionar outros nós aqui
            END: END 
        }
    )
    
    # Após cada nó de ação, o subgrafo termina para esperar a próxima entrada do usuário.
    # O dispatcher interno na próxima invocação direcionará corretamente.
    subgraph_builder.add_edge("solicitar_nome_completo_node", END)
    subgraph_builder.add_edge("coletar_nome_completo_node", END) # Termina para esperar input de data/hora
    subgraph_builder.add_edge("solicitar_data_hora_node", END) # Termina para esperar input de data/hora

    logger.info("Subgrafo de agendamento (com dispatcher interno) definido.")
    return subgraph_builder

# Função para ser chamada pelo grafo principal
# Ela compila o subgrafo e o retorna.
# O checkpointer do grafo principal será propagado automaticamente.
def get_compiled_scheduling_subgraph(llm_client_from_main: Optional[ChatOpenAI] = None) -> StateGraph:
    # Se o subgrafo não precisar do llm_client diretamente nos nós (porque usa serviços que o recebem),
    # não precisamos passá-lo aqui. Mas se os nós usarem, como no exemplo acima, passamos.
    # A instância do llm_client pode ser gerenciada no grafo principal e passada para cá.
    definition = get_scheduling_subgraph_definition(llm_for_nodes=llm_client_from_main)
    # O checkpointer é herdado do grafo pai, não precisamos definir um aqui
    # a menos que queiramos um checkpointer *independente* para o subgrafo.
    return definition.compile()
