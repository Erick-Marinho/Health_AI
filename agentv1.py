# %%
from typing import Annotated, Optional
from typing_extensions import TypedDict, Dict
from dotenv import load_dotenv
import requests
import re
import gradio as gr
import nest_asyncio
import logging
import json
from datetime import datetime, date, timedelta, time

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain

from IPython.display import display, Image
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel, field_validator, ValidationError



# %%
API_TOKEN = "TXH3xnwI7P4Sh5dS71aRDsQzDrx1GKeW8jXd5eHDpqTOi" # Substitua pelo seu token real
api_headers = {
    "Authorization": f"{API_TOKEN}",
    "Content-Type": "application/json" # Adicione outros cabeçalhos necessários
}

# %%
# Configuração básica do logger (ajuste o nível conforme necessário)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s]: %(message)s') # Ou logging.INFO, logging.WARNING etc.
logger = logging.getLogger(__name__)

# %%
load_dotenv()
nest_asyncio.apply()
memory = MemorySaver()
# No início do seu script, após os imports e load_dotenv()
llm = ChatOpenAI(temperature=0) # Ou o modelo de sua escolha

# %%
class NomeCompletoModel(BaseModel):
    nome_completo: str

    @field_validator('nome_completo', mode='before')
    @classmethod
    def validar_formato_nome(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Entrada para nome deve ser uma string")
        nome_limpo = v.strip()

        if not nome_limpo:
            raise ValueError("O nome completo não pode ser vazio")
        
        if len(nome_limpo.split()) < 2:
            raise ValueError("O nome completo deve conter pelo menos um nome e um sobrenome")
        
        if not re.match(r'^[a-zA-Z\s]+$', nome_limpo):
            raise ValueError("O nome completo deve conter apenas letras e espaços")
        
        return nome_limpo

# %%
class EspecialidadeModel(BaseModel):
    especialidade: str

    @field_validator('especialidade', mode='before')
    @classmethod
    def validar_especialidade(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("A especialidade deve ser uma string")
        
        especialidade_limpa = v.strip()
        
        if not especialidade_limpa:
            raise ValueError("A especialidade não pode ser vazia")
        
        if len(especialidade_limpa) < 3:
            raise ValueError("A especialidade deve ter pelo menos 3 caracteres")
        
        return especialidade_limpa

# %%
class ProfissionalNomeModel(BaseModel):
    nome_profissional: str

    @field_validator('nome_profissional', mode='before')
    @classmethod
    def validar_formato_nome_profissional(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Nome do profissional deve ser uma string")
        nome_limpo = v.strip()
        if not nome_limpo:
            raise ValueError("O nome do profissional não pode ser vazio")
        if len(nome_limpo.split()) < 2: # Exigir pelo menos nome e sobrenome
            raise ValueError("O nome do profissional parece incompleto. Por favor, informe nome e sobrenome.")
        # Outras validações podem ser adicionadas (ex: apenas letras e espaços)
        return nome_limpo

# %%
class State(TypedDict):
    messages: Annotated[list, add_messages]
    categoria: Optional[str]
    dados_unidades_api: dict

    fase_agendamento: Optional[str]
    nome_para_agendamento: Optional[str]
    erro_coleta_agendamento: Optional[str]

    especialidade_para_agendamento: Optional[str]
    especialidade_id_para_agendamento: Optional[int]

    preferencia_profissional: Optional[str]
    profissional_para_agendamento: Optional[str]
    nome_profissional_sugerido_usuario: Optional[str]
    id_profissional_selecionado: Optional[int]
    profissionais_encontrados: Optional[list] 
    erro_api_profissionais: Optional[str] 
    turno_para_agendamento: Optional[str]

    datas_disponiveis_apresentadas: Optional[list[str]] # As até 3 datas mostradas ao usuário
    data_agendamento_escolhida: Optional[str]
    horarios_disponiveis_apresentados: Optional[list[str]] # Os até 2 horários mostrados
    horario_agendamento_escolhido: Optional[str]

# %%
def get_unidades() -> Dict[str, str]:
    """
    Consulta a informação clínica
    """

    url = "https://back.homologacao.apphealth.com.br:9090/api-vizi/unidades"
    headers = {
        "Authorization": "TXH3xnwI7P4Sh5dS71aRDsQzDrx1GKeW8jXd5eHDpqTOi"
    }

    try:
        resposta = requests.get(
            url,
            headers=headers,
            timeout=10 
        )
        
        resposta.raise_for_status()
        
        return resposta.json()
        
    except requests.exceptions.HTTPError as http_err:
        return {
            "error": "Erro na resposta da API",
            "status_code": resposta.status_code if hasattr(resposta, 'status_code') else 'Não disponível',
            "details": str(http_err)
        }
    except requests.exceptions.ConnectionError as conn_err:
        return {"error": "Falha na conexão", "details": str(conn_err)}
    except requests.exceptions.Timeout as timeout_err:
        return {"error": "Timeout na requisição", "details": str(timeout_err)}
    except requests.exceptions.RequestException as req_err:
        return {"error": "Erro na requisição", "details": str(req_err)}
    
def get_especialidade_id_por_nome(nome_especialidade_usuario: str, headers: Dict[str, str]) -> Optional[int]:
    """
    Busca todas as especialidades da API (usando os headers fornecidos) e usa o LLM 
    para encontrar a correspondência mais próxima para a especialidade fornecida 
    pelo usuário, retornando seu ID.
    """
    logger.debug(f"Iniciando busca de ID para especialidade '{nome_especialidade_usuario}' usando LLM, com headers.")
    url_especialidades_todas = "https://back.homologacao.apphealth.com.br:9090/api-vizi/especialidades"
    
    try:
        # 1. Buscar TODAS as especialidades da API usando os headers
        # Não passamos params aqui, pois queremos a lista completa para o LLM.
        response = requests.get(url_especialidades_todas, headers=headers, timeout=10) 
        response.raise_for_status() 
        especialidades_api = response.json()
        
        if not especialidades_api or not isinstance(especialidades_api, list):
            logger.warning(f"API de especialidades ({url_especialidades_todas}) retornou dados vazios ou em formato inesperado usando os headers.")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro ao buscar todas as especialidades da API ({url_especialidades_todas}) usando headers: {e}")
        return None

    # Extrair apenas os nomes para o prompt do LLM
    nomes_especialidades_oficiais = [
        item["especialidade"] for item in especialidades_api if "especialidade" in item and isinstance(item.get("especialidade"), str)
    ]
    if not nomes_especialidades_oficiais:
        logger.warning("Não foi possível extrair nomes de especialidades válidos da resposta da API (lista completa).")
        return None

    # 2. Formular o prompt para o LLM
    prompt_text = """Com base na especialidade informada pelo usuário e na lista de especialidades médicas válidas abaixo, determine qual item da lista melhor corresponde à entrada do usuário.
Usuário informou: "{especialidade_usuario}"
Lista de especialidades válidas: {lista_especialidades}

Retorne APENAS o nome EXATO da especialidade da lista que melhor corresponde. Se nenhuma corresponder bem, retorne 'Nenhuma correspondência'.
Sua resposta deve ser apenas o nome da especialidade ou 'Nenhuma correspondência'.
"""
    
    prompt = PromptTemplate(
        input_variables=["especialidade_usuario", "lista_especialidades"],
        template=prompt_text
    )
    
    chain = LLMChain(llm=llm, prompt=prompt) # 'llm' deve estar definido e acessível
    
    try:
        # 3. Invocar o LLM
        resultado_llm = chain.invoke({
            "especialidade_usuario": nome_especialidade_usuario,
            "lista_especialidades": ", ".join(nomes_especialidades_oficiais)
        })
        
        if not resultado_llm or 'text' not in resultado_llm:
            logger.error("Resposta do LLM inválida ou não contém a chave 'text'.")
            return None
            
        nome_especialidade_llm = resultado_llm['text'].strip()
        logger.debug(f"LLM sugeriu a especialidade: '{nome_especialidade_llm}' para a entrada '{nome_especialidade_usuario}'")

        if nome_especialidade_llm == "Nenhuma correspondência" or not nome_especialidade_llm:
            logger.info(f"LLM indicou nenhuma correspondência ou retornou vazio para '{nome_especialidade_usuario}'.")
            return None

        # 4. Encontrar o ID correspondente na lista original da API
        for item in especialidades_api:
            if "especialidade" in item and "id" in item and \
               item["especialidade"] == nome_especialidade_llm and \
               isinstance(item["id"], int):
                logger.info(f"LLM associou '{nome_especialidade_usuario}' com '{nome_especialidade_llm}' (ID: {item['id']})")
                return item["id"]
        
        logger.warning(f"Especialidade '{nome_especialidade_llm}' sugerida pelo LLM não foi encontrada na lista original da API ou o item correspondente é inválido.")
        return None

    except Exception as e:
        logger.error(f"Erro durante a invocação do LLM ou processamento da sua resposta: {e}")
        return None
    
def consultar_api_profissionais(state: State) -> dict:
    """
    Consulta a API para obter a lista de profissionais.
    Pode buscar por ID de especialidade ou todos os profissionais ativos
    para validação posterior.
    """
    logger.debug("[consultar_api_profissionais] Iniciando consulta à API de profissionais.")
    messages = state.get("messages", []).copy()
    modo_consulta = state.get("modo_consulta_profissionais", "POR_ESPECIALIDADE") # Assume que esta chave será definida corretamente antes
    especialidade_id = state.get("especialidade_id_para_agendamento")
    nome_profissional_para_validar = state.get("nome_profissional_para_validar")

    # Verificar ID apenas se o modo for POR_ESPECIALIDADE
    if modo_consulta == "POR_ESPECIALIDADE":
        if not especialidade_id:
            logger.error("[consultar_api_profissionais] MODO_CONSULTA='POR_ESPECIALIDADE', mas ID da especialidade não encontrado no estado.")
            messages.append(
                AIMessage(content="Desculpe, houve um problema interno ao identificar a especialidade. Não consigo prosseguir com a busca de profissionais no momento.")
            )
            return {
                "messages": messages,
                "fase_agendamento": "ERRO_INTERNO_BUSCA_PROFISSIONAIS",
                "profissionais_encontrados": [], # Garante a chave mesmo em erro
                "erro_api_profissionais": "ID da especialidade ausente"
            }

    url = "https://back.homologacao.apphealth.com.br:9090/api-vizi/profissionais"
    params = {"status": "true"} # Sempre buscar profissionais ativos

    if modo_consulta == "POR_ESPECIALIDADE":
        params["especialidadeId"] = especialidade_id
        logger.info(f"[consultar_api_profissionais] Consultando API por ESPECIALIDADE: URL='{url}', Params='{params}'")
    elif modo_consulta == "TODOS_PARA_VALIDACAO":
        logger.info(f"[consultar_api_profissionais] Consultando TODOS os profissionais ativos para validação: URL='{url}', Params='{params}'")
        # Não adiciona especialidadeId aos params
    else:
        logger.error(f"[consultar_api_profissionais] Modo de consulta desconhecido: {modo_consulta}")
        messages.append(AIMessage(content="Ocorreu um erro interno ao tentar buscar profissionais. Por favor, tente novamente."))
        return {
            "messages": messages,
            "fase_agendamento": "ERRO_INTERNO_GERAL",
            "profissionais_encontrados": [], # Garante a chave
            "erro_api_profissionais": f"Modo de consulta inválido: {modo_consulta}"
        }

    try:
        response = requests.get(url, headers=api_headers, params=params, timeout=10)
        response.raise_for_status()
        profissionais_api = response.json()
        logger.debug(f"[consultar_api_profissionais] API retornou {len(profissionais_api) if isinstance(profissionais_api, list) else 'dados inválidos'}.")

        # A lógica de processamento da resposta agora depende do modo_consulta
        if modo_consulta == "POR_ESPECIALIDADE":
            if profissionais_api and isinstance(profissionais_api, list) and len(profissionais_api) > 0:
                nomes_profissionais = [p.get("nome") for p in profissionais_api if p.get("nome")]
                if nomes_profissionais:
                    max_nomes_display = 5
                    nomes_display = nomes_profissionais[:max_nomes_display]
                    mensagem_resposta = f"Encontrei os seguintes profissionais para a especialidade desejada: {', '.join(nomes_display)}"
                    if len(nomes_profissionais) > max_nomes_display:
                        mensagem_resposta += f" e mais {len(nomes_profissionais) - max_nomes_display}."
                    mensagem_resposta += " Qual deles você prefere, ou gostaria de mais detalhes sobre algum?"

                    messages.append(AIMessage(content=mensagem_resposta))

                    # --- CORREÇÃO AQUI ---
                    update_dict = {
                        "messages": messages, # Atualiza as mensagens
                        "profissionais_encontrados": profissionais_api, # Usa a chave correta 'profissionais_encontrados'
                        "fase_agendamento": "PROFISSIONAIS_APRESENTADOS_AGUARDANDO_ESCOLHA",
                        "erro_api_profissionais": None # Limpa erros anteriores
                    }
                    # --- FIM DA CORREÇÃO ---

                    logger.debug(f"PREPARANDO RETORNO (sucesso POR_ESPECIALIDADE): update_dict chaves = {list(update_dict.keys())}")
                    logger.debug(f"RETORNANDO DE consultar_api_profissionais (sucesso POR_ESPECIALIDADE): profissionais_encontrados = {profissionais_api}")
                    return update_dict
                else: # Profissionais encontrados, mas sem nomes válidos
                    mensagem_resposta = "Encontrei alguns registros de profissionais, mas não consegui obter os nomes. Por favor, tente novamente mais tarde ou contate o suporte."
                    messages.append(AIMessage(content=mensagem_resposta))
                    logger.debug(f"RETORNANDO DE consultar_api_profissionais (sem nomes válidos): profissionais_encontrados = {profissionais_api}")
                    return {
                        "messages": messages,
                         # --- CORREÇÃO AQUI ---
                        "profissionais_encontrados": profissionais_api, # Usa a chave correta, mesmo que os dados internos sejam problemáticos
                        # --- FIM DA CORREÇÃO ---
                        "fase_agendamento": "ERRO_PROCESSAMENTO_NOMES_PROFISSIONAIS",
                    }
            else: # Nenhum profissional encontrado para a especialidade
                mensagem_resposta = "No momento, não encontrei profissionais disponíveis para a especialidade selecionada. Gostaria de tentar outra especialidade ou verificar mais tarde?"
                profissionais_api = []
                messages.append(AIMessage(content=mensagem_resposta))
                logger.debug("RETORNANDO DE consultar_api_profissionais (nenhum encontrado POR_ESPECIALIDADE): profissionais_encontrados = []")
                return {
                    "messages": messages,
                    # --- CORREÇÃO AQUI ---
                    "profissionais_encontrados": profissionais_api, # Usa a chave correta com lista vazia
                     # --- FIM DA CORREÇÃO ---
                    "fase_agendamento": "NENHUM_PROFISSIONAL_ENCONTRADO_PARA_ESPECIALIDADE", # Nome da fase corrigido (era NENHUM_PROFISSIONAL_ENCONTRADO)
                }

        elif modo_consulta == "TODOS_PARA_VALIDACAO":
            if not profissionais_api:
                 logger.warning("[consultar_api_profissionais] MODO_CONSULTA='TODOS_PARA_VALIDACAO', mas nenhum profissional foi retornado pela API.")
                 profissionais_api = []

            logger.info(f"[consultar_api_profissionais] Lista de {len(profissionais_api)} profissionais obtida para validação (nome alvo: '{nome_profissional_para_validar}').")
            logger.debug(f"RETORNANDO DE consultar_api_profissionais (TODOS_PARA_VALIDACAO): profissionais_encontrados = {profissionais_api}")
            return {
                "messages": messages,
                "fase_agendamento": "LISTA_PROFISSIONAIS_PARA_VALIDACAO_OBTIDA",
                # --- CORREÇÃO AQUI ---
                "profissionais_encontrados": profissionais_api, # Usa a chave correta
                # --- FIM DA CORREÇÃO ---
                "erro_api_profissionais": None
            }

    except requests.exceptions.HTTPError as e:
        logger.error(f"[consultar_api_profissionais] Erro HTTP: {e.response.status_code} - {e.response.text} (Modo: {modo_consulta})")
        mensagem_erro = f"Desculpe, houve um erro ao buscar os profissionais (Código: {e.response.status_code}). Por favor, tente novamente mais tarde."
        messages.append(AIMessage(content=mensagem_erro))
        logger.debug("RETORNANDO DE consultar_api_profissionais (HTTPError): profissionais_encontrados = []")
        return {
            "messages": messages,
            "profissionais_encontrados": [], # Usa a chave correta
            "fase_agendamento": "ERRO_API_PROFISSIONAIS",
            "erro_api_profissionais": f"Erro HTTP: {e.response.status_code}"
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"[consultar_api_profissionais] Erro de Requisição: {e} (Modo: {modo_consulta})")
        mensagem_erro = "Desculpe, não consegui me conectar ao sistema para buscar os profissionais. Verifique sua conexão ou tente novamente mais tarde."
        messages.append(AIMessage(content=mensagem_erro))
        logger.debug("RETORNANDO DE consultar_api_profissionais (RequestException): profissionais_encontrados = []")
        return {
            "messages": messages,
            "profissionais_encontrados": [], # Usa a chave correta
            "fase_agendamento": "ERRO_CONEXAO_API_PROFISSIONAIS",
            "erro_api_profissionais": f"Erro de Conexão: {e}"
        }
    except Exception as e:
        logger.exception(f"[consultar_api_profissionais] Erro Inesperado (Modo: {modo_consulta}):")
        mensagem_erro = "Desculpe, ocorreu um erro inesperado ao processar sua solicitação de busca por profissionais. Por favor, tente novamente."
        messages.append(AIMessage(content=mensagem_erro))
        logger.debug(f"RETORNANDO DE consultar_api_profissionais (Exception - {type(e).__name__}): profissionais_encontrados = []")
        return {
            "messages": messages,
            "profissionais_encontrados": [], # Usa a chave correta
            "fase_agendamento": "ERRO_INESPERADO_BUSCA_PROFISSIONAIS",
            "erro_api_profissionais": f"Erro Inesperado: {e}"
        }
    
def buscar_profissional_por_nome_na_lista_api(nome_buscado: str, lista_profissionais_api: list, especialidade_nome: str) -> Optional[dict]:
    """
    Usa o LLM para encontrar a melhor correspondência para nome_buscado dentro da lista_profissionais_api.
    Retorna o objeto do profissional se uma correspondência única for encontrada, senão None.
    """
    if not lista_profissionais_api:
        return None

    nomes_profissionais_api = [prof.get("nome") for prof in lista_profissionais_api if prof.get("nome")]
    if not nomes_profissionais_api:
        return None

    prompt_template = ChatPromptTemplate.from_template(
        """
        Você precisa encontrar a melhor correspondência para um nome de profissional fornecido por um usuário dentro de uma lista de nomes de profissionais de uma API.
        A especialidade é {especialidade}.

        Nome fornecido pelo usuário: "{nome_usuario}"
        Lista de nomes de profissionais disponíveis da API para esta especialidade: {lista_api_nomes}

        Analise o nome fornecido pelo usuário e retorne APENAS o nome EXATO da lista_api_nomes que melhor corresponde.
        Se houver uma correspondência clara e única, retorne o nome exato da lista.
        Se não houver uma correspondência clara, ou se o nome do usuário for muito diferente de qualquer opção na lista, retorne "NENHUMA_CORRESPONDENCIA_CLARA".
        Se o nome do usuário puder corresponder a múltiplos itens da lista de forma ambígua, retorne "MULTIPLAS_CORRESPONDENCIAS_AMBIGUAS".

        Exemplos:
        - Usuário: "Doutora Clara Joaquina", Lista: ["Clara Joaquina", "Carlos Andrade"] -> Clara Joaquina
        - Usuário: "Clara J.", Lista: ["Clara Joaquina", "Carlos Andrade"] -> Clara Joaquina
        - Usuário: "Dr. Silva", Lista: ["João Silva", "Maria Silva", "Carlos Andrade"] -> MULTIPLAS_CORRESPONDENCIAS_AMBIGUAS
        - Usuário: "Dr. Pereira", Lista: ["Clara Joaquina", "Carlos Andrade"] -> NENHUMA_CORRESPONDENCIA_CLARA

        Nome exato da lista, "NENHUMA_CORRESPONDENCIA_CLARA", ou "MULTIPLAS_CORRESPONDENCIAS_AMBIGUAS":
        """
    )
    chain = prompt_template | llm

    try:
        resultado_llm = chain.invoke({
            "nome_usuario": nome_buscado,
            "lista_api_nomes": ", ".join(nomes_profissionais_api),
            "especialidade": especialidade_nome
        }).content.strip()

        logger.debug(f"[buscar_profissional_por_nome_na_lista_api] LLM para '{nome_buscado}' vs lista API: resultado '{resultado_llm}'")

        if resultado_llm == "NENHUMA_CORRESPONDENCIA_CLARA" or resultado_llm == "MULTIPLAS_CORRESPONDENCIAS_AMBIGUAS":
            logger.info(f"LLM indicou: {resultado_llm} para o nome '{nome_buscado}'")
            return {"status": resultado_llm} # Retorna um dict com status para o chamador tratar

        # Se o LLM retornou um nome, encontrar o objeto profissional correspondente
        for prof in lista_profissionais_api:
            if prof.get("nome") == resultado_llm:
                return {"status": "SUCESSO", "profissional": prof}
        
        logger.warning(f"LLM retornou '{resultado_llm}', mas não foi encontrado na lista original da API.")
        return {"status": "ERRO_INTERNO_LLM_NAO_ENCONTRADO_NA_LISTA"}

    except Exception as e:
        logger.error(f"Erro na chamada LLM para correspondência de nome de profissional: {e}")
        return {"status": "ERRO_CHAMADA_LLM"}


# %%
# nodes functions

def get_last_user_message_content(state_messages: list) -> Optional[str]:
    """Helper para obter o conteúdo da última mensagem do usuário."""
    if not state_messages:
        return None
    
    last_message = state_messages[-1]
    content = None
    role = None

    if hasattr(last_message, 'type') and hasattr(last_message, 'content'): # Formato Langchain Message (e.g. HumanMessage)
        role = last_message.type
        content = last_message.content
    elif isinstance(last_message, tuple) and len(last_message) == 2: # Formato ("role", "content")
        role, content = last_message
    elif isinstance(last_message, dict) and "role" in last_message and "content" in last_message: # Formato dict
        role = last_message["role"]
        content = last_message["content"]
    
    if role == "user" or (hasattr(last_message, 'type') and last_message.type == "human"): # Verifica se é do usuário
        return content
    elif role == "ai" or (hasattr(last_message, 'type') and last_message.type == "ai"):
        print(f"DEBUG [get_last_user_message_content]: Última mensagem é do AI: '{content}'. Esperando input do usuário.")
        return None 

    print(f"AVISO [get_last_user_message_content]: Formato de última mensagem não reconhecido ou não é do usuário: {last_message}")
    return None

def checar_fase_ou_categorizar(state: State) -> dict:
    """
    Decide se continua um fluxo de agendamento ou categoriza uma nova consulta.
    Este nó agora será o ponto de partida lógico após a entrada do usuário.
    """
    print(f"DEBUG [checar_fase_ou_categorizar]: Estado atual: fase_agendamento='{state.get('fase_agendamento')}'")
    if state.get("fase_agendamento"):
        # Se estamos em alguma fase de agendamento, o input do usuário é para essa fase
        return {"next_node": "processar_input_agendamento"}
    else:
        # Caso contrário, é uma nova consulta para categorizar
        return {"next_node": "categorize"}

def categorize(state: State) -> dict:
    """
    Categoriza a mensagem do usuário.
    """
    user_message_content = get_last_user_message_content(state["messages"])

    if not user_message_content:
        print("AVISO [categorize]: Não foi possível obter a última mensagem do usuário para categorização.")
        # Decide o que fazer: pode ser erro, ou categoria padrão
        return {"categoria": "Indefinido", "messages": [("ai", "Não consegui processar sua última mensagem. Pode repetir?")]}


    prompt = ChatPromptTemplate.from_template(
        """
        Categorize a seguinte consulta do cliente em uma das seguintes categorias:
        - "Inserir Agendamento"
        - "Consultar Agendamento"
        - "Atualizar Agendamento"
        - "Cancelar Agendamento"
        - "Informações Unidade"
        - "Saudação ou Despedida"
        - "Fora do Escopo"

        Retorne APENAS o nome da categoria escolhida, sem nenhum texto explicativo, aspas literais ou prefixos como "Categoria:".
        Se a consulta não se encaixar claramente ou for apenas uma saudação/despedida, use "Saudação ou Despedida" ou "Fora do Escopo".

        Consulta do cliente: {user_query}
        Categoria Selecionada:"""
    )

    chain = prompt | ChatOpenAI(temperature=0) # Especificar modelo é bom
    categoria_llm = chain.invoke({"user_query": user_message_content}).content
    categoria_limpa = categoria_llm.replace('Categoria: ', '').replace('"', '').strip()
    
    print(f"DEBUG [categorize]: User Message='{user_message_content}', LLM_Output='{categoria_llm}', Cleaned_Category='{categoria_limpa}'")
    return {"categoria": categoria_limpa}

def iniciar_coleta_nome_agendamento(state: State) -> dict:
    """
    Inicia a coleta do nome para um novo agendamento.
    """
    print("DEBUG [iniciar_coleta_nome_agendamento]: Iniciando coleta de nome.")

    prompt_humanizado = ChatPromptTemplate.from_template(
        "Você é um assistente virtual muito simpático de uma clínica médica. "
        "Gere uma mensagem curta e acolhedora para iniciar o processo de agendamento, "
        "pedindo o nome completo do paciente. "
        "Exemplos de tom: 'Claro, vamos começar! Para dar início ao seu agendamento, pode me dizer seu nome completo, por favor?', "
        "'Com prazer! Para que eu possa registrar seu pedido de agendamento, qual é o seu nome completo?'. "
        "Retorne APENAS a mensagem gerada, sem introduções ou despedidas."
    )

    chain_geradora_mensagem = prompt_humanizado | llm

    try:
        # Invoca o LLM para gerar a mensagem
        resposta_llm = chain_geradora_mensagem.invoke({}) # Não precisa de input específico aqui
        mensagem_para_usuario = resposta_llm.content.strip()
        logger.info(f"[iniciar_coleta_nome_agendamento]: Mensagem gerada pelo LLM: '{mensagem_para_usuario}'")

    except Exception as e:
        logger.error(f"[iniciar_coleta_nome_agendamento]: Erro ao gerar mensagem com LLM: {e}. Usando mensagem padrão.")
        # Fallback para a mensagem original em caso de erro
        mensagem_para_usuario = "Entendido! Para iniciar o seu agendamento, por favor, informe o seu nome completo."

    return {
        "fase_agendamento": "AGUARDANDO_NOME",
        "messages": [("ai", mensagem_para_usuario)], # Usa a mensagem gerada ou o fallback
        "erro_coleta_agendamento": None # Limpa erros anteriores
    }

def processar_input_nome_agendamento(state: State) -> dict:
    """
    Processa a entrada do usuário quando a fase é AGUARDANDO_NOME.
    """
    print(f"DEBUG [processar_input_nome_agendamento]: Processando nome. Fase: {state.get('fase_agendamento')}")
    user_input = get_last_user_message_content(state["messages"])

    if not user_input:
        print("AVISO [processar_input_nome_agendamento]: Nenhuma entrada do usuário para processar como nome.")
        return {
            "messages": [("ai", "Não recebi seu nome. Poderia informar, por favor?")],
            "erro_coleta_agendamento": "Entrada não recebida."
        }

    try:
        nome_validado_model = NomeCompletoModel(nome_completo=user_input)
        nome_validado = nome_validado_model.nome_completo
        
        print(f"DEBUG [processar_input_nome_agendamento]: Nome validado: '{nome_validado}'")
        # PRÓXIMO PASSO: Mudar fase e pedir próxima informação ou confirmar
        # Por enquanto, vamos apenas confirmar o nome e finalizar o fluxo de agendamento para teste
        mensagem_confirmacao = f"Obrigado, {nome_validado}! Seu nome foi registrado."
        # Futuramente: self.fase_agendamento = "AGUARDANDO_SERVICO"
        # mensagem_confirmacao = f"Obrigado, {nome_validado}! Qual serviço você gostaria de agendar?"
        
        return {
            "nome_para_agendamento": nome_validado,
            "fase_agendamento": "NOME_COLETADO", # Ou "AGUARDANDO_SERVICO" no futuro
            "messages": [("ai", mensagem_confirmacao)],
            "erro_coleta_agendamento": None
        }
    except ValidationError as e:
        # Extração da mensagem de erro de forma mais limpa
        if e.errors():
            # Acessa a mensagem do primeiro erro.
            # A estrutura exata pode depender da versão do Pydantic,
            # mas 'msg' geralmente contém a mensagem da exceção original.
            raw_error_msg = e.errors()[0].get('msg', "Entrada inválida.")
            
            # Remove o prefixo "Value error, " se estiver presente,
            # ou outros prefixos que Pydantic possa adicionar dependendo do tipo de erro.
            # Esta é uma forma simples de limpar; para casos mais complexos,
            # você poderia ter mapeamentos de tipos de erro para mensagens customizadas.
            if isinstance(raw_error_msg, str): # Garantir que é uma string
                if raw_error_msg.lower().startswith("value error, "):
                    error_detail = raw_error_msg[len("value error, "):].strip()
                elif "assertion failed," in raw_error_msg.lower(): # Exemplo para outro tipo de erro Pydantic
                    # Tratar especificamente se necessário, ou apenas usar a msg
                    error_detail = e.errors()[0].get('ctx', {}).get('error', raw_error_msg)
                    if hasattr(error_detail, 'message'): # Se ctx.error for uma exceção
                        error_detail = str(error_detail.message)
                    else:
                        error_detail = str(error_detail)

                else:
                    error_detail = raw_error_msg
            else:
                error_detail = "Entrada inválida." # Fallback

        else:
            error_detail = "Ocorreu um erro de validação não especificado."
            
        mensagem_erro_usuario = f"Houve um problema com o nome informado: {error_detail}. Por favor, tente novamente."
        print(f"ERRO [processar_input_nome_agendamento]: Erro de validação do nome: {error_detail}")
        return {
            "messages": [("ai", mensagem_erro_usuario)],
            "erro_coleta_agendamento": error_detail,
            "fase_agendamento": "AGUARDANDO_NOME" # Mantém na mesma fase para nova tentativa
        }
    
def iniciar_coleta_especialidade(state: State) -> dict:
    """
    Inicia a coleta da especialidade após o nome ter sido coletado.
    """
    print("DEBUG [iniciar_coleta_especialidade]: Iniciando coleta de especialidade.")
    nome = state.get("nome_para_agendamento", "cliente")
    mensagem_para_usuario = f"Obrigado, {nome}! Qual especialidade médica você gostaria de agendar?"
    
    return {
        "fase_agendamento": "AGUARDANDO_ESPECIALIDADE",
        "messages": [("ai", mensagem_para_usuario)],
        "erro_coleta_agendamento": None
    }

def processar_input_especialidade(state: State) -> dict:
    """
    Processa a entrada do usuário quando a fase é AGUARDANDO_ESPECIALIDADE.
    """
    print(f"DEBUG [processar_input_especialidade]: Processando especialidade. Fase: {state.get('fase_agendamento')}")
    user_input = get_last_user_message_content(state["messages"])

    if not user_input:
        print("AVISO [processar_input_especialidade]: Nenhuma entrada do usuário para processar como especialidade.")
        return {
            "messages": [("ai", "Não recebi a especialidade. Poderia informar, por favor?")],
            "erro_coleta_agendamento": "Entrada não recebida.",
            "fase_agendamento": "AGUARDANDO_ESPECIALIDADE"
        }

    try:
        especialidade_validada_model = EspecialidadeModel(especialidade=user_input)
        especialidade_validada = especialidade_validada_model.especialidade
        
        print(f"DEBUG [processar_input_especialidade]: Especialidade validada: '{especialidade_validada}'")
        
        return {
            "especialidade_para_agendamento": especialidade_validada,
            "fase_agendamento": "ESPECIALIDADE_COLETADA_AGUARDANDO_PREFERENCIA_PROFISSIONAL",
            "messages": [("ai", f"Ótimo! Você selecionou a especialidade: {especialidade_validada}. Agora, gostaria de escolher um profissional específico ou prefere que eu indique os disponíveis?")],
            "erro_coleta_agendamento": None
        }
    except ValidationError as e:
        if e.errors():
            raw_error_msg = e.errors()[0].get('msg', "Entrada inválida.")
            
            if isinstance(raw_error_msg, str):
                if raw_error_msg.lower().startswith("value error, "):
                    error_detail = raw_error_msg[len("value error, "):].strip()
                else:
                    error_detail = raw_error_msg
            else:
                error_detail = "Entrada inválida."

        else:
            error_detail = "Ocorreu um erro de validação não especificado."
            
        mensagem_erro_usuario = f"Houve um problema com a especialidade informada: {error_detail}. Por favor, tente novamente."
        print(f"ERRO [processar_input_especialidade]: Erro de validação da especialidade: {error_detail}")
        return {
            "messages": [("ai", mensagem_erro_usuario)],
            "erro_coleta_agendamento": error_detail,
            "fase_agendamento": "AGUARDANDO_ESPECIALIDADE"
        }
    
def processar_preferencia_profissional(state: State) -> dict:
    logger.debug(f"Iniciando processamento de preferência de profissional com LLM. Fase atual: '{state.get('fase_agendamento')}'")
    user_input = get_last_user_message_content(state["messages"])
    messages = state.get("messages", [])
    # Não precisamos mais de especialidade_id ou especialidade_nome aqui diretamente para chamadas API
    # especialidade_id = state.get("especialidade_id_para_agendamento") # REMOVER ou COMENTAR
    # especialidade_nome = state.get("especialidade_para_agendamento", "a especialidade selecionada") # REMOVER ou COMENTAR

    if not user_input:
        # ... (lógica existente para input vazio - manter como está)
        logger.warning("[processar_preferencia_profissional] Nenhuma entrada do usuário.")
        # A mensagem de reprompt já deve ter sido enviada pela fase anterior se necessário,
        # ou o usuário simplesmente não respondeu. Manter na mesma fase para aguardar.
        return {
            "messages": messages, 
            "fase_agendamento": "ESPECIALIDADE_COLETADA_AGUARDANDO_PREFERENCIA_PROFISSIONAL"
        }

    prompt_classificacao_template = ChatPromptTemplate.from_template(
        """
        Analise a seguinte mensagem do usuário que acabou de informar uma especialidade médica e agora precisa decidir sobre o profissional.
        O usuário pode:
        1. Querer uma recomendação de profissionais (Ex: "me indique", "quais as opções?", "pode ser qualquer um").
        2. Querer nomear um profissional específico (Ex: "quero o Dr. Silva", "prefiro a Dra. Ana", "já tenho um nome").
        3. Fornecer diretamente o nome de um profissional (Ex: "Dra. Clara Joaquina", "Carlos Andrade").
        4. Indicar que quer nomear um profissional, mas sem ainda dizer o nome (Ex: "eu gostaria de escolher", "prefiro nomear").

        Sua tarefa é:
        A. Classificar a intenção do usuário em uma das seguintes categorias: "PEDIR_RECOMENDACAO", "ESCOLHER_ESPECIFICO_COM_NOME", "ESCOLHER_ESPECIFICO_SEM_NOME", "APENAS_NOME_PROFISSIONAL", "AMBIGUO_OU_NAO_ENTENDIDO".
        B. Se a categoria for "ESCOLHER_ESPECIFICO_COM_NOME" ou "APENAS_NOME_PROFISSIONAL", extrair o nome completo do profissional mencionado.

        Retorne sua resposta APENAS como um objeto JSON com as chaves "categoria_preferencia" e "nome_extraido_profissional" (use null se nenhum nome for extraído ou aplicável).

        Exemplos de saída:
        - Usuário: "me indique um, por favor" -> {{"categoria_preferencia": "PEDIR_RECOMENDACAO", "nome_extraido_profissional": null}}
        - Usuário: "quero agendar com a Doutora Clara Joaquina" -> {{"categoria_preferencia": "ESCOLHER_ESPECIFICO_COM_NOME", "nome_extraido_profissional": "Doutora Clara Joaquina"}}
        - Usuário: "Doutor João" -> {{"categoria_preferencia": "APENAS_NOME_PROFISSIONAL", "nome_extraido_profissional": "Doutor João"}}
        - Usuário: "prefiro escolher" -> {{"categoria_preferencia": "ESCOLHER_ESPECIFICO_SEM_NOME", "nome_extraido_profissional": null}}
        - Usuário: "não sei" -> {{"categoria_preferencia": "AMBIGUO_OU_NAO_ENTENDIDO", "nome_extraido_profissional": null}}

        Mensagem do usuário: "{user_query}"
        JSON de saída (APENAS o objeto JSON, sem qualquer outro texto):
        """
    )

    
    chain_classificacao = prompt_classificacao_template | llm

    # DEBUGGING EXTRA:
    logger.debug(f"DEBUG_TEMPLATE_VARS: Template input_variables: {prompt_classificacao_template.input_variables}")
    # logger.debug(f"DEBUG_TEMPLATE_STR: Template string: {prompt_classificacao_template.template}")

    if hasattr(prompt_classificacao_template, 'messages') and prompt_classificacao_template.messages:
        # Para ChatPromptTemplate.from_template, a string original geralmente está no prompt da primeira mensagem
        first_message_prompt = getattr(prompt_classificacao_template.messages[0], 'prompt', None)
        if first_message_prompt and hasattr(first_message_prompt, 'template'):
            logger.debug(f"DEBUG_TEMPLATE_STR: Template string: {first_message_prompt.template}")
        else:
            logger.debug("DEBUG_TEMPLATE_STR: Não foi possível extrair a string do template da primeira mensagem.")
    else:
        logger.debug("DEBUG_TEMPLATE_STR: O prompt_classificacao_template não possui o atributo 'messages' ou está vazio.")

    try:
        user_input_for_llm = get_last_user_message_content(state["messages"]) # Para confirmar o que está sendo enviado
        print(f"DEBUG_PROCESSAR_PREFERENCIA_USER_INPUT: '{user_input_for_llm}'") # Print para confirmar a entrada

        llm_response_str = chain_classificacao.invoke({"user_query": user_input}).content

        print(f"DEBUG_LLM_RESPONSE_STR_RAW: '{llm_response_str}'") # Print direto da saída do LLM
        logger.debug(f"LLM_RESPONSE_STR_RAW: '{llm_response_str}'")

        match = re.search(r'\{.*\}', llm_response_str, re.DOTALL)
        if not match:
            print(f"DEBUG_REGEX_NAO_ENCONTROU_JSON. Resposta completa do LLM: '{llm_response_str}'")
            logger.error(f"[processar_preferencia_profissional] Regex não encontrou JSON. Resposta completa do LLM: '{llm_response_str}'") 
            raise json.JSONDecodeError("Nenhum JSON na resposta do LLM Classificação.", llm_response_str, 0)
        
        parsed_llm_response = json.loads(match.group(0))
        categoria_llm = parsed_llm_response.get("categoria_preferencia")
        nome_extraido_pelo_llm_inicial = parsed_llm_response.get("nome_extraido_profissional")

        logger.info(f"[processar_preferencia_profissional] LLM Classificou: Categoria='{categoria_llm}', Nome Extraído Inicial='{nome_extraido_pelo_llm_inicial}'")

        # Limpar qualquer nome de profissional sugerido anteriormente para evitar confusão
        updated_state = {
            "nome_profissional_sugerido_usuario": None,
            "profissional_para_agendamento": None, # Limpar também este para garantir
            "id_profissional_selecionado": None   # e este
        }

        if categoria_llm == "PEDIR_RECOMENDACAO":
            # A mensagem para o usuário será dada pelo nó que efetivamente busca as recomendações.
            # Não precisamos adicionar uma mensagem aqui.
            logger.info("[processar_preferencia_profissional] Preferência: PEDIR_RECOMENDACAO.")
            updated_state.update({
                "messages": messages,
                "preferencia_profissional": "pedir_recomendacao",
                "fase_agendamento": "PREFERENCIA_PROFISSIONAL_PROCESSADA",
                "erro_coleta_agendamento": None
            })
            return updated_state
        
        elif categoria_llm == "ESCOLHER_ESPECIFICO_SEM_NOME":
            # O usuário quer escolher, mas ainda não disse o nome.
            # A mensagem para pedir o nome será dada pelo nó iniciar_coleta_nome_profissional.
            logger.info("[processar_preferencia_profissional] Preferência: ESCOLHER_ESPECIFICO_SEM_NOME.")
            updated_state.update({
                "messages": messages,
                "preferencia_profissional": "escolher_especifico_sem_nome_ainda", # Nova preferência para clareza
                "fase_agendamento": "PREFERENCIA_PROFISSIONAL_PROCESSADA",
                "erro_coleta_agendamento": None
            })
            return updated_state

        elif categoria_llm in ["ESCOLHER_ESPECIFICO_COM_NOME", "APENAS_NOME_PROFISSIONAL"] and nome_extraido_pelo_llm_inicial:
            logger.info(f"[processar_preferencia_profissional] Preferência: {categoria_llm} com nome: '{nome_extraido_pelo_llm_inicial}'.")
            # Apenas armazenamos o nome. A validação e busca na API ocorrerão no próximo nó.
            # Não adicionamos mensagem ao usuário aqui; o próximo nó cuidará disso após a tentativa de busca/validação.
            updated_state.update({
                "messages": messages,
                "preferencia_profissional": "escolher_especifico_com_nome_fornecido", # Nova preferência para clareza
                "nome_profissional_sugerido_usuario": nome_extraido_pelo_llm_inicial, # NOVA CHAVE NO ESTADO
                "fase_agendamento": "PREFERENCIA_PROFISSIONAL_PROCESSADA", 
                "erro_coleta_agendamento": None
            })
            return updated_state
        
        else: # AMBIGUO_OU_NAO_ENTENDIDO ou LLM não extraiu nome quando deveria
            logger.warning(f"[processar_preferencia_profissional] Preferência ambígua ou LLM não extraiu nome. Categoria: {categoria_llm}, Nome: {nome_extraido_pelo_llm_inicial}")
            mensagem_reprompt = "Desculpe, não consegui entender sua preferência ou o nome do profissional. Você gostaria de nomear um profissional específico ou prefere que eu busque opções para você?"
            messages.append(AIMessage(content=mensagem_reprompt))
            updated_state.update({
                "messages": messages,
                "fase_agendamento": "ESPECIALIDADE_COLETADA_AGUARDANDO_PREFERENCIA_PROFISSIONAL", 
                "preferencia_profissional": None # Garante que a preferência seja reavaliada
            })
            return updated_state

    except json.JSONDecodeError as e:
        logger.error(f"[processar_preferencia_profissional] Erro ao decodificar JSON da resposta do LLM: {e}")
        mensagem_erro_tecnico = "Desculpe, tive um problema técnico ao processar sua preferência. Poderia tentar novamente, por favor?"
        messages.append(AIMessage(content=mensagem_erro_tecnico))
        # Retornar ao estado anterior para nova tentativa do usuário
        return {
            "messages": messages,
            "fase_agendamento": "ESPECIALIDADE_COLETADA_AGUARDANDO_PREFERENCIA_PROFISSIONAL",
            "preferencia_profissional": None
        }
    except Exception as e:
        logger.exception("[processar_preferencia_profissional] Erro inesperado:")
        mensagem_erro_tecnico = "Desculpe, ocorreu um erro inesperado ao processar sua preferência. Poderia repetir, por favor?"
        messages.append(AIMessage(content=mensagem_erro_tecnico))
        return {
            "messages": messages,
            "fase_agendamento": "ESPECIALIDADE_COLETADA_AGUARDANDO_PREFERENCIA_PROFISSIONAL", 
            "preferencia_profissional": None
        }

def gerar_resposta_informacao_unidade(state: State) -> dict:
    user_message_content = get_last_user_message_content(state["messages"])
    if not user_message_content: # Adicionando verificação similar
        print("AVISO [gerar_resposta_informacao_unidade]: Sem mensagem do usuário.")
        return {"messages": [("ai", "Sobre o que você gostaria de informações da unidade?")]}

    dados_unidades_api = get_unidades()
    if "error" in dados_unidades_api: # Tratar erro da API
        print(f"ERRO [gerar_resposta_informacao_unidade]: Falha ao obter dados da unidade: {dados_unidades_api}")
        return {"messages": [("ai", f"Desculpe, estou com problemas para acessar as informações da unidade no momento. (Detalhe: {dados_unidades_api.get('error')})")]}

    prompt = ChatPromptTemplate.from_template(
        """
        Você é um assistente virtual de uma clínica médica.
        Use os seguintes dados para responder à consulta do usuário:

        Dados da clínica: {dados_unidades_api}

        Consulta: {user_query}

        Formato telefone: (DDI) (DDD) Número. Se houver ramal, mencione-o.
        Responda de forma completa e amigável, formatada em markdown.
        """
    )

    chain = prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2) # Um pouco de temperatura para naturalidade
    resposta_content = chain.invoke({
        "dados_unidades_api": str(dados_unidades_api), # Convertido para string para o prompt
        "user_query": user_message_content
    }).content
    
    return {"messages": [("ai", resposta_content)], "dados_unidades_api": dados_unidades_api}

def gerar_resposta_saudacao_despedida(state: State) -> dict:
    """Gera uma resposta para saudações ou despedidas."""
    user_message_content = get_last_user_message_content(state["messages"])
    # Poderia ter lógicas diferentes para saudação vs despedida baseado no histórico ou análise simples
    resposta = "Olá! Como posso te ajudar hoje?"
    if user_message_content and any(term in user_message_content.lower() for term in ["tchau", "até mais", "obrigado", "adeus"]):
        resposta = "De nada! Se precisar de mais alguma coisa, é só chamar. Até logo!"
    
    print(f"DEBUG [gerar_resposta_saudacao_despedida]: Categoria Saudação/Despedida. User: '{user_message_content}' AI: '{resposta}'")
    return {"messages": [("ai", resposta)]}

def gerar_resposta_fora_escopo(state: State) -> dict:
    user_message_content = get_last_user_message_content(state["messages"])
    # Se não houver user_message_content, uma mensagem genérica
    if not user_message_content:
        user_message_content = "sua solicitação" 

    prompt = ChatPromptTemplate.from_template(
        """
        Você é um assistente virtual da Clínica HealthAI.
        Sua função é auxiliar com informações sobre a clínica e agendamentos.
        Informe ao usuário de forma educada que você não pode ajudar com a seguinte solicitação, pois está fora do seu escopo de atuação.
        Se a solicitação for vaga ou não compreendida, peça para reformular ou dar mais detalhes sobre o que precisa em relação à clínica.

        Solicitação do usuário: {user_query}

        Resposta formatada em markdown:
        """
    )

    chain = prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    resposta_content = chain.invoke({"user_query": user_message_content}).content
    
    return {"messages": [("ai", resposta_content)]}
    
def processar_input_nome_profissional_indicado(state: State) -> dict:
    """
    Processa o nome do profissional indicado pelo usuário e pede confirmação.
    """
    print(f"DEBUG [processar_input_nome_profissional_indicado]: Fase: {state.get('fase_agendamento')}")
    user_input_nome_profissional = get_last_user_message_content(state["messages"])

    if not user_input_nome_profissional:
        print("AVISO [processar_input_nome_profissional_indicado]: Nenhuma entrada do usuário para processar como nome do profissional.")
        return {
            "messages": [("ai", "Não recebi o nome do profissional. Poderia informar, por favor?")],
            "erro_coleta_agendamento": "Nome do profissional não recebido.",
            "fase_agendamento": "AGUARDANDO_NOME_PROFISSIONAL_INDICADO" 
        }

    try: # DESCOMENTE ESTE BLOCO TRY
        # Validacao Pydantic (opcional, mas recomendada)
        profissional_validado_model = ProfissionalNomeModel(nome_profissional=user_input_nome_profissional) # DESCOMENTE
        nome_profissional_validado = profissional_validado_model.nome_profissional # DESCOMENTE
        
        # nome_profissional_para_confirmar = user_input_nome_profissional.strip() # REMOVA OU COMENTE ESTA LINHA
        nome_profissional_para_confirmar = nome_profissional_validado # USE O NOME VALIDADO

        # A checagem básica abaixo torna-se redundante se a validação Pydantic estiver ativa e funcionando
        # if not nome_profissional_para_confirmar: # COMENTE OU REMOVA ESTE BLOCO IF
        #     return {
        #         "messages": [("ai", "O nome do profissional não pode ser vazio. Poderia informar novamente?")],
        #         "erro_coleta_agendamento": "Nome do profissional vazio.",
        #         "fase_agendamento": "AGUARDANDO_NOME_PROFISSIONAL_INDICADO"
        #     }
            
        print(f"DEBUG [processar_input_nome_profissional_indicado]: Nome do profissional validado e para confirmar: '{nome_profissional_para_confirmar}'")
        
        mensagem_confirmacao = f"Entendido. Só para confirmar, o profissional que você deseja é: {nome_profissional_para_confirmar}?"
        
        return {
            "profissional_para_agendamento": nome_profissional_para_confirmar, 
            "fase_agendamento": "NOME_PROFISSIONAL_INDICADO_AGUARDANDO_CONFIRMACAO", 
            "messages": [("ai", mensagem_confirmacao)],
            "erro_coleta_agendamento": None
        }
    except ValidationError as e:
        if e.errors():
            raw_error_msg = e.errors()[0].get('msg', "Entrada inválida.")
            # Sua lógica de extração de 'error_detail' aqui. Exemplo:
            if isinstance(raw_error_msg, str):
                if raw_error_msg.lower().startswith("value error, "):
                    error_detail = raw_error_msg[len("value error, "):].strip()
                else:
                    error_detail = raw_error_msg
            else:
                error_detail = "Entrada inválida."
        else:
            error_detail = "Ocorreu um erro de validação não especificado."
            
        mensagem_erro_usuario = f"Houve um problema com o nome do profissional informado: {error_detail}. Por favor, tente novamente."
        print(f"ERRO [processar_input_nome_profissional_indicado]: Erro de validação: {error_detail}")
        return {
            "messages": [("ai", mensagem_erro_usuario)],
            "erro_coleta_agendamento": error_detail,
            "fase_agendamento": "AGUARDANDO_NOME_PROFISSIONAL_INDICADO"
        }

def iniciar_coleta_nome_profissional(state: State) -> dict:
    """
    Pede ao usuário o nome do profissional específico que ele deseja agendar.
    """
    print("DEBUG [iniciar_coleta_nome_profissional]: Solicitando nome do profissional específico.")
    
    mensagem_para_usuario = "Entendido! Por favor, informe o nome completo do profissional com quem você gostaria de agendar."
    
    return {
        "messages": [("ai", mensagem_para_usuario)],
        "fase_agendamento": "AGUARDANDO_NOME_PROFISSIONAL_INDICADO", # Define a fase esperada pelo processar_input_nome_profissional_indicado
        "erro_coleta_agendamento": None 
    }

def processar_confirmacao_nome_profissional(state: State) -> dict:
    logger.debug(f"[processar_confirmacao_nome_profissional]: Fase atual: {state.get('fase_agendamento')}")
    messages = state.get("messages", []).copy()
    user_input_confirmacao = get_last_user_message_content(messages) # Pega a ÚLTIMA mensagem, que deve ser do usuário
    
    nome_profissional_confirmado_anteriormente = state.get("profissional_para_agendamento")
    especialidade_nome = state.get("especialidade_para_agendamento", "a especialidade selecionada") # Pegar para a mensagem

    if not user_input_confirmacao:
        logger.warning("[processar_confirmacao_nome_profissional]: Nenhuma entrada do usuário para processar a confirmação.")
        # Adiciona à lista existente de mensagens
        messages.append(AIMessage(content=f"Não recebi sua confirmação sobre o profissional {nome_profissional_confirmado_anteriormente}. É este o profissional correto?"))
        return {
            "messages": messages,
            "erro_coleta_agendamento": "Confirmação não recebida.",
            "fase_agendamento": "NOME_PROFISSIONAL_INDICADO_AGUARDANDO_CONFIRMACAO" 
        }

    lower_input = user_input_confirmacao.lower()
    
    # Usar LLM para classificação Sim/Não seria mais robusto que keywords
    # Por enquanto, mantendo sua lógica de keywords:
    resposta_classificada = "INCERTA"
    keywords_sim = ["sim", "s", "isso", "correto", "confirmo", "é esse", "positivo", "exato"]
    keywords_nao = ["não", "n", "errado", "incorreto", "outro", "nao é"]

    if any(keyword in lower_input for keyword in keywords_sim):
        resposta_classificada = "SIM"
    elif any(keyword in lower_input for keyword in keywords_nao):
        resposta_classificada = "NAO"

    if resposta_classificada == "SIM":
        logger.info(f"[processar_confirmacao_nome_profissional]: Usuário confirmou o profissional: {nome_profissional_confirmado_anteriormente}")
        mensagem_para_usuario = f"Ótimo! Agendamento com {nome_profissional_confirmado_anteriormente} confirmado. Agora, qual turno você prefere para o agendamento: manhã ou tarde?"
        messages.append(AIMessage(content=mensagem_para_usuario))
        return {
            "messages": messages,
            # MUITO IMPORTANTE: Esta fase deve ser a que dispara a coleta de turno!
            "fase_agendamento": "AGUARDANDO_TURNO", 
            "erro_coleta_agendamento": None,
            # nome_profissional_sugerido_usuario e preferencia_profissional podem ser mantidos ou limpos.
            # Se limpos, garante que não interfiram em fluxos futuros se algo der errado.
            # "nome_profissional_sugerido_usuario": None, 
            # "preferencia_profissional": None,
        }
    elif resposta_classificada == "NAO":
        logger.info(f"[processar_confirmacao_nome_profissional]: Usuário NÃO confirmou o profissional: {nome_profissional_confirmado_anteriormente}")
        mensagem_para_usuario = (f"Entendido. O profissional não era {nome_profissional_confirmado_anteriormente}. "
                                 f"Você gostaria de tentar informar outro nome para {especialidade_nome}, "
                                 "ou prefere que eu apresente algumas opções de indicação?")
        messages.append(AIMessage(content=mensagem_para_usuario))
        return {
            "messages": messages,
            "profissional_para_agendamento": None, 
            "id_profissional_selecionado": None,
            "nome_profissional_sugerido_usuario": None, # Limpa a sugestão anterior
            "preferencia_profissional": None, # Força reavaliação da preferência no próximo passo
            "fase_agendamento": "ESPECIALIDADE_COLETADA_AGUARDANDO_PREFERENCIA_PROFISSIONAL", 
            "erro_coleta_agendamento": "Usuário indicou que o nome do profissional estava incorreto."
        }
    else: # INCERTA
        logger.warning(f"[processar_confirmacao_nome_profissional]: Resposta de confirmação não clara: '{user_input_confirmacao}'")
        mensagem_para_usuario = (f"Desculpe, não entendi sua resposta sobre o profissional "
                                 f"{nome_profissional_confirmado_anteriormente}. Ele está correto, sim ou não?")
        messages.append(AIMessage(content=mensagem_para_usuario))
        return {
            "messages": messages,
            "fase_agendamento": "NOME_PROFISSIONAL_INDICADO_AGUARDANDO_CONFIRMACAO"
        }

def preparar_consulta_api_profissionais(state: State) -> dict:
    logger.debug("[preparar_consulta_api_profissionais] Iniciando preparação para consulta API ou validação de nome.")
    messages = state.get("messages", [])
    especialidade_nome = state.get("especialidade_para_agendamento")
    current_especialidade_id = state.get("especialidade_id_para_agendamento")
    nome_sugerido_pelo_usuario = state.get("nome_profissional_sugerido_usuario")
    preferencia = state.get("preferencia_profissional") # Para sabermos se devemos listar todos ou validar nome

    # api_headers é global

    if not especialidade_nome:
        logger.error("[preparar_consulta_api_profissionais] Nome da especialidade não encontrado no estado.")
        messages.append(AIMessage(content="Desculpe, ocorreu um erro interno e não consegui identificar a especialidade. Por favor, poderia tentar o agendamento desde o início?"))
        return {
            "messages": messages,
            "fase_agendamento": None,
            "erro_api_profissionais": "Nome da especialidade crítico ausente."
        }

    # Passo 1: Garantir que temos o ID da especialidade
    actual_especialidade_id = None
    if current_especialidade_id and isinstance(current_especialidade_id, int):
        logger.debug(f"[preparar_consulta_api_profissionais] ID da especialidade ({current_especialidade_id}) já existe para '{especialidade_nome}'.")
        actual_especialidade_id = current_especialidade_id
    else:
        logger.info(f"[preparar_consulta_api_profissionais] Buscando ID para especialidade '{especialidade_nome}'...")
        id_encontrado_api = get_especialidade_id_por_nome(especialidade_nome, api_headers)
        if id_encontrado_api is not None:
            logger.info(f"[preparar_consulta_api_profissionais] ID {id_encontrado_api} obtido para '{especialidade_nome}'.")
            actual_especialidade_id = id_encontrado_api
        else:
            logger.error(f"[preparar_consulta_api_profissionais] Falha ao obter ID para '{especialidade_nome}'.")
            mensagem_erro_usuario = (f"Desculpe, não consegui encontrar um ID correspondente para a especialidade '{especialidade_nome}'. "
                                     "Gostaria de tentar informar a especialidade novamente ou escolher outra?")
            messages.append(AIMessage(content=mensagem_erro_usuario))
            return {
                "messages": messages,
                "fase_agendamento": "AGUARDANDO_ESPECIALIDADE",
                "erro_api_profissionais": f"Falha ao obter ID da API para {especialidade_nome}",
                "especialidade_para_agendamento": None,
                "especialidade_id_para_agendamento": None,
                "preferencia_profissional": None
            }
    
    # Se chegamos aqui, temos actual_especialidade_id
    # Guardar o ID obtido/confirmado no estado é crucial
    updated_state_dict = {"especialidade_id_para_agendamento": actual_especialidade_id}

    # Passo 2: Verificar se o usuário já forneceu um nome de profissional
    if preferencia == "escolher_especifico_com_nome_fornecido" and nome_sugerido_pelo_usuario:
        logger.info(f"[preparar_consulta_api_profissionais] Usuário forneceu nome '{nome_sugerido_pelo_usuario}'. Validando com API para especialidade ID {actual_especialidade_id}.")
        
        # Buscar TODOS os profissionais da especialidade para poder usar buscar_profissional_por_nome_na_lista_api
        url_prof_todos = "https://back.homologacao.apphealth.com.br:9090/api-vizi/profissionais"
        params_prof_todos = {"status": "true", "especialidadeId": actual_especialidade_id}
        
        try:
            logger.debug(f"[preparar_consulta_api_profissionais] Consultando API de profissionais (lista completa): URL='{url_prof_todos}', Params='{params_prof_todos}'")
            response_prof_todos = requests.get(url_prof_todos, headers=api_headers, params=params_prof_todos, timeout=10)
            response_prof_todos.raise_for_status()
            lista_profissionais_da_especialidade = response_prof_todos.json()

            if not lista_profissionais_da_especialidade or not isinstance(lista_profissionais_da_especialidade, list):
                logger.warning(f"[preparar_consulta_api_profissionais] Nenhum profissional retornado pela API para especialidade ID {actual_especialidade_id} ao tentar validar nome '{nome_sugerido_pelo_usuario}'.")
                messages.append(AIMessage(content=f"Para a especialidade {especialidade_nome}, não encontrei nenhum profissional cadastrado. Gostaria de tentar outra especialidade ou que eu procure recomendações gerais (se aplicável)?"))
                updated_state_dict.update({
                    "messages": messages,
                    "fase_agendamento": "ESPECIALIDADE_COLETADA_AGUARDANDO_PREFERENCIA_PROFISSIONAL", # Volta para o usuário decidir
                    "nome_profissional_sugerido_usuario": None # Limpa o nome
                })
                return updated_state_dict

        except requests.RequestException as e:
            logger.error(f"[preparar_consulta_api_profissionais] Erro ao buscar lista completa de profissionais da API para validar nome: {e}")
            messages.append(AIMessage(content="Desculpe, tive um problema ao consultar a lista de profissionais para validar o nome. Poderia tentar novamente em instantes?"))
            updated_state_dict.update({
                "messages": messages,
                "fase_agendamento": "ESPECIALIDADE_COLETADA_AGUARDANDO_PREFERENCIA_PROFISSIONAL", # Volta para o usuário decidir
                "nome_profissional_sugerido_usuario": None
            })
            return updated_state_dict

        # Agora, com a lista, usar o LLM para encontrar a melhor correspondência
        resultado_busca_obj = buscar_profissional_por_nome_na_lista_api(
            nome_sugerido_pelo_usuario, 
            lista_profissionais_da_especialidade,
            especialidade_nome # Passar o nome da especialidade para o prompt do LLM
        )

        status_busca_llm = resultado_busca_obj.get("status") if resultado_busca_obj else "ERRO_INESPERADO_LLM_BUSCA"

        if status_busca_llm == "SUCESSO":
            profissional_encontrado_api = resultado_busca_obj["profissional"]
            nome_oficial_api = profissional_encontrado_api.get("nome")
            id_oficial_api = profissional_encontrado_api.get("id")

            if nome_oficial_api and id_oficial_api:
                logger.info(f"[preparar_consulta_api_profissionais] Nome '{nome_sugerido_pelo_usuario}' correspondido com API: '{nome_oficial_api}' (ID: {id_oficial_api}).")
                mensagem_confirmacao = f"Entendido. Encontrei o(a) profissional: {nome_oficial_api} para {especialidade_nome}. É este(a) mesmo quem você deseja?"
                messages.append(AIMessage(content=mensagem_confirmacao))
                updated_state_dict.update({
                    "messages": messages,
                    "profissional_para_agendamento": nome_oficial_api, # Nome oficial para confirmação
                    "id_profissional_selecionado": id_oficial_api,
                    "fase_agendamento": "NOME_PROFISSIONAL_INDICADO_AGUARDANDO_CONFIRMACAO",
                    # nome_profissional_sugerido_usuario pode ser mantido ou limpo aqui. Manter pode ser útil se o usuário disser "não".
                    "erro_api_profissionais": None
                })
                return updated_state_dict
            else:
                logger.error("[preparar_consulta_api_profissionais] Profissional da API retornado por LLM sem nome ou ID válidos.")
                status_busca_llm = "ERRO_INTERNO_LLM_DADOS_INVALIDOS" # Tratar como falha

        # Se a busca LLM falhou ou não encontrou correspondência clara
        if status_busca_llm == "NENHUMA_CORRESPONDENCIA_CLARA":
            msg_usuario_falha = f"Desculpe, não consegui encontrar um profissional chamado '{nome_sugerido_pelo_usuario}' para a especialidade {especialidade_nome} na nossa lista. Gostaria de tentar um nome diferente ou prefere que eu liste as opções disponíveis?"
        elif status_busca_llm == "MULTIPLAS_CORRESPONDENCIAS_AMBIGUAS":
            msg_usuario_falha = f"Encontrei alguns profissionais com nomes parecidos com '{nome_sugerido_pelo_usuario}' para {especialidade_nome}. Para evitar confusão, poderia fornecer o nome completo e correto, ou prefere que eu liste as opções?"
        else: # ERRO_CHAMADA_LLM, ERRO_INTERNO_LLM_NAO_ENCONTRADO_NA_LISTA, ERRO_INTERNO_LLM_DADOS_INVALIDOS, ERRO_INESPERADO_LLM_BUSCA
            logger.warning(f"[preparar_consulta_api_profissionais] Falha na busca/validação do nome '{nome_sugerido_pelo_usuario}' via LLM. Status: {status_busca_llm}")
            msg_usuario_falha = f"Tive um problema ao tentar verificar o nome '{nome_sugerido_pelo_usuario}'. Gostaria de tentar novamente com outro nome, ou prefere que eu liste as opções para {especialidade_nome}?"
        
        messages.append(AIMessage(content=msg_usuario_falha))
        updated_state_dict.update({
            "messages": messages,
            "fase_agendamento": "ESPECIALIDADE_COLETADA_AGUARDANDO_PREFERENCIA_PROFISSIONAL",
            "preferencia_profissional": None, # Para forçar nova pergunta de preferência
            "nome_profissional_sugerido_usuario": None, # Limpa sugestão anterior
            "profissional_para_agendamento": None,
            "id_profissional_selecionado": None,
            "erro_api_profissionais": f"Falha na validação do nome: {status_busca_llm}"
        })
        return updated_state_dict

    elif preferencia == "pedir_recomendacao":
        # Usuário pediu recomendação, apenas preparamos o ID da especialidade.
        logger.info(f"[preparar_consulta_api_profissionais] Usuário pediu recomendação para especialidade ID {actual_especialidade_id}. Pronto para consultar API.")
        # A mensagem ao usuário será dada pelo próximo nó (consultar_api_profissionais)
        updated_state_dict.update({
            "messages": messages, # Mensagens não são alteradas aqui neste caso
            "fase_agendamento": "PRONTO_PARA_CONSULTAR_API_PROFISSIONAIS",
            "erro_api_profissionais": None
        })
        return updated_state_dict
        
    elif preferencia == "escolher_especifico_sem_nome_ainda":
        # O usuário quer escolher, mas ainda não forneceu o nome.
        # O roteamento após processar_preferencia_profissional já deveria ter direcionado
        # para iniciar_coleta_nome_profissional.
        # Se chegou aqui, é um pouco inesperado, mas podemos apenas preparar o ID e deixar
        # o próximo roteamento decidir.
        logger.info(f"[preparar_consulta_api_profissionais] Usuário quer escolher nome (ainda não fornecido) para esp. ID {actual_especialidade_id}. ID pronto.")
        updated_state_dict.update({
            "messages": messages,
             # A fase aqui poderia ser algo que o próximo nó de roteamento (route_after_preparar_consulta_api)
             # use para ir para iniciar_coleta_nome_profissional, ou apenas PRONTO_PARA_CONSULTAR_API_PROFISSIONAIS
             # e a lógica de roteamento trata. Por simplicidade, manteremos PRONTO... e a rota decide.
            "fase_agendamento": "PRONTO_PARA_CONSULTAR_API_PROFISSIONAIS", # Ou uma fase mais específica
            "erro_api_profissionais": None
        })
        return updated_state_dict

    else:
        # Caso de fallback ou preferência não tratada explicitamente acima
        logger.warning(f"[preparar_consulta_api_profissionais] Preferência não tratada ou estado inesperado: '{preferencia}'. Fase será PRONTO_PARA_CONSULTAR_API.")
        updated_state_dict.update({
            "messages": messages,
            "fase_agendamento": "PRONTO_PARA_CONSULTAR_API_PROFISSIONAIS", # Default para busca geral
            "erro_api_profissionais": "Preferência não mapeada em preparar_consulta_api_profissionais"
        })
        return updated_state_dict
        
def executar_consulta_profissionais_e_apresentar(state: State) -> dict:
    """
    Consulta a API de profissionais usando o ID da especialidade e apresenta os resultados.
    """
    logger.info("DEBUG [executar_consulta_profissionais_e_apresentar]: Iniciando consulta à API de profissionais.") # Usar logger.info ou logger.debug
    
    current_messages = state.get("messages", []).copy() # Obter uma cópia das mensagens atuais para evitar modificar o estado diretamente aqui
    
    especialidade_id = state.get("especialidade_id_para_agendamento")
    especialidade_nome = state.get("especialidade_para_agendamento", "a especialidade solicitada")

    if not isinstance(especialidade_id, int):
        logger.critical(f"[executar_consulta_profissionais_e_apresentar]: ID da especialidade ({especialidade_id}) é inválido ou não encontrado no estado.") # Usar logger.critical ou logger.error
        error_message = f"Desculpe, ocorreu um erro interno e não consegui identificar o ID da especialidade para a busca. Por favor, poderia tentar o agendamento desde o início?"
        return {
            "messages": current_messages + [AIMessage(content=error_message)], # Adicionar ao histórico
            "fase_agendamento": None, 
            "erro_api_profissionais": f"ID da especialidade crítico ausente ou inválido ({especialidade_id}) no estado para consulta de profissionais."
        }

    api_headers = {"Authorization": "TXH3xnwI7P4Sh5dS71aRDsQzDrx1GKeW8jXd5eHDpqTOi"} 
    url = f"https://back.homologacao.apphealth.com.br:9090/api-vizi/profissionais"
    params = {"status": True, "especialidadeId": especialidade_id}

    logger.debug(f"[executar_consulta_profissionais_e_apresentar]: Consultando URL: {url} com params: {params}")

    try:
        resposta = requests.get(url, headers=api_headers, params=params, timeout=15)
        resposta.raise_for_status()
        profissionais_encontrados_api_response = resposta.json()

        if profissionais_encontrados_api_response and isinstance(profissionais_encontrados_api_response, list) and len(profissionais_encontrados_api_response) > 0:
            logger.debug(f"[executar_consulta_profissionais_e_apresentar]: {len(profissionais_encontrados_api_response)} profissional(is) encontrado(s) para ID {especialidade_id}.")
            
            nomes_profissionais = [prof.get("nome") for prof in profissionais_encontrados_api_response if prof.get("nome")]
            
            if not nomes_profissionais: 
                logger.warning("[executar_consulta_profissionais_e_apresentar]: Profissionais encontrados, mas sem nomes válidos na resposta da API.")
                mensagem_usuario = (f"Encontrei alguns profissionais para {especialidade_nome}, mas estou com dificuldade para listar os nomes. "
                                    "Gostaria de tentar novamente ou escolher outra especialidade?")
                return {
                    "messages": current_messages + [AIMessage(content=mensagem_usuario)], # Adicionar ao histórico
                    "profissionais_encontrados": profissionais_encontrados_api_response, 
                    "fase_agendamento": "ERRO_PROCESSAMENTO_NOMES_PROFISSIONAIS", 
                    "erro_api_profissionais": "Profissionais retornados sem nomes válidos."
                }

            lista_nomes_str = ", ".join(nomes_profissionais)
            mensagem_usuario = (f"Para {especialidade_nome}, encontrei os seguintes profissionais disponíveis: {lista_nomes_str}. "
                                "Algum deles te interessa? Se sim, por favor, diga o nome.")
            
            # Log para depurar o que está sendo armazenado
            logger.debug(f"[executar_consulta_profissionais_e_apresentar] Armazenando profissionais: {profissionais_encontrados_api_response}")
            logger.debug(f"[executar_consulta_profissionais_e_apresentar] Definindo fase para: PROFISSIONAIS_APRESENTADOS_AGUARDANDO_ESCOLHA")

            return_dict = {
                "messages": current_messages + [AIMessage(content=mensagem_usuario)], 
                "profissionais_encontrados": profissionais_encontrados_api_response, 
                "fase_agendamento": "PROFISSIONAIS_APRESENTADOS_AGUARDANDO_ESCOLHA",
                "erro_api_profissionais": None
            }
            logger.info(f"[executar_consulta_profissionais_e_apresentar] VALOR SENDO RETORNADO para 'profissionais_encontrados': {return_dict.get('profissionais_encontrados')}")
            return return_dict
        else:
            logger.info(f"[executar_consulta_profissionais_e_apresentar]: Nenhum profissional encontrado para ID {especialidade_id} ({especialidade_nome}). Resposta: {profissionais_encontrados_api_response}")
            mensagem_usuario = (f"No momento, não encontrei profissionais disponíveis para {especialidade_nome}. "
                                "Gostaria de tentar outra especialidade ou verificar novamente mais tarde?")
            
            logger.debug(f"[executar_consulta_profissionais_e_apresentar] Nenhum profissional encontrado. Armazenando lista vazia.")
            logger.debug(f"[executar_consulta_profissionais_e_apresentar] Definindo fase para: NENHUM_PROFISSIONAL_ENCONTRADO")
            
            return_dict = {
                "messages": current_messages + [AIMessage(content=mensagem_usuario)], 
                "profissionais_encontrados": [], 
                "fase_agendamento": "NENHUM_PROFISSIONAL_ENCONTRADO",
                "erro_api_profissionais": None
            }
            logger.info(f"[executar_consulta_profissionais_e_apresentar] VALOR SENDO RETORNADO para 'profissionais_encontrados': {return_dict.get('profissionais_encontrados')}")
            return return_dict

    except requests.exceptions.HTTPError as http_err:
        err_msg = f"Erro HTTP ao consultar profissionais para ID {especialidade_id}: {http_err}. Status: {http_err.response.status_code if http_err.response else 'N/A'}"
        logger.error(f"[executar_consulta_profissionais_e_apresentar]: {err_msg}") # Usar logger
        # Construir mensagem de erro para o usuário e adicionar ao histórico
        user_error_message = f"Desculpe, tive um problema ao buscar os profissionais para {especialidade_nome}. Por favor, tente novamente mais tarde."
        return {
            "messages": current_messages + [AIMessage(content=user_error_message)],
            "fase_agendamento": "ERRO_CONSULTA_API_PROFISSIONAIS",
            "erro_api_profissionais": err_msg,
            "profissionais_encontrados": None # Limpar em caso de erro
        }
    except requests.exceptions.Timeout as timeout_err:
        err_msg = f"Timeout ao consultar profissionais para ID {especialidade_id}: {timeout_err}"
        logger.error(f"[executar_consulta_profissionais_e_apresentar]: {err_msg}") # Usar logger
        user_error_message = f"A busca por profissionais para {especialidade_nome} está demorando muito. Por favor, tente novamente mais tarde."
        return {
            "messages": current_messages + [AIMessage(content=user_error_message)],
            "fase_agendamento": "ERRO_CONSULTA_API_PROFISSIONAIS",
            "erro_api_profissionais": err_msg,
            "profissionais_encontrados": None # Limpar em caso de erro
        }
    except requests.exceptions.RequestException as req_err:
        err_msg = f"Erro de requisição ao consultar profissionais para ID {especialidade_id}: {req_err}"
        logger.error(f"[executar_consulta_profissionais_e_apresentar]: {err_msg}") # Usar logger
        user_error_message = f"Ocorreu um erro de comunicação ao buscar os profissionais para {especialidade_nome}. Por favor, tente mais tarde."
        return {
            "messages": current_messages + [AIMessage(content=user_error_message)],
            "fase_agendamento": "ERRO_CONSULTA_API_PROFISSIONAIS",
            "erro_api_profissionais": err_msg,
            "profissionais_encontrados": None # Limpar em caso de erro
        }
    except ValueError as json_err: # Se a resposta não for JSON válido
        # Importante: a variável 'profissionais_encontrados_api_response' pode não estar definida se o erro ocorreu antes de sua atribuição
        err_msg = f"Erro de decodificação JSON ao consultar profissionais para ID {especialidade_id}: {json_err}"
        logger.error(f"[executar_consulta_profissionais_e_apresentar]: {err_msg}") # Usar logger
        user_error_message = f"Recebi uma resposta inesperada ao buscar os profissionais para {especialidade_nome}. Por favor, informe nossa equipe técnica."
        return {
            "messages": current_messages + [AIMessage(content=user_error_message)],
            # "profissionais_encontrados": profissionais_encontrados_api_response, # REMOVER ESTA LINHA, variável pode não existir
            "profissionais_encontrados": None, # Definir como None ou lista vazia
            "fase_agendamento": "ERRO_CONSULTA_API_PROFISSIONAIS",
            "erro_api_profissionais": err_msg
        }
    
def placeholder_processar_escolha_profissional_recomendado(state: State) -> dict:
    user_choice = get_last_user_message_content(state["messages"])
    available_professionals = state.get("profissionais_disponiveis_api", [])
    nomes_disponiveis = [prof.get("nome", "").lower() for prof in available_professionals]
    
    if user_choice and user_choice.lower() in nomes_disponiveis:
        # Encontrou o profissional na lista!
        # Pode pegar o objeto completo do profissional se precisar de mais dados dele.
        profissional_escolhido_obj = next((prof for prof in available_professionals if prof.get("nome", "").lower() == user_choice.lower()), None)
        
        print(f"DEBUG [placeholder_processar_escolha_profissional_recomendado]: Usuário escolheu profissional recomendado: {user_choice}, Objeto: {profissional_escolhido_obj}")
        return {
            "profissional_para_agendamento": user_choice, # Ou profissional_escolhido_obj.get("nome")
            "fase_agendamento": "PROFISSIONAL_RECOMENDADO_ESCOLHIDO", # Nova fase final
            "messages": [("ai", f"Ótimo! Agendamento com {user_choice} selecionado. (Próximos passos do agendamento a implementar)")]
        }
    else:
        # Não entendeu a escolha ou nome não está na lista
        nomes_str = ", ".join([prof.get("nome") for prof in available_professionals if prof.get("nome")])
        return {
            "messages": [("ai", f"Desculpe, não encontrei '{user_choice}' na lista de recomendações ({nomes_str}). Por favor, informe um dos nomes listados ou peça para ver outras opções se houver.")],
            "fase_agendamento": "RECOMENDACOES_APRESENTADAS_AGUARDANDO_ESCOLHA" # Volta para a mesma fase
        }
    
def processar_escolha_profissional(state: State) -> dict:
    """
    Processa a escolha do profissional feita pelo usuário a partir da lista de recomendações.
    """
    logger.debug("Iniciando processamento da escolha do profissional recomendado.")
    logger.debug(f"[processar_escolha_profissional] Estado recebido (chaves): {list(state.keys())}")

    # --- CORREÇÃO PRINCIPAL AQUI ---
    # Ler da chave correta definida no State: 'profissionais_encontrados'
    lista_prof_no_estado = state.get("profissionais_encontrados")
    logger.debug(f"[processar_escolha_profissional] Conteúdo de state['profissionais_encontrados']: {lista_prof_no_estado}")
    # --- FIM DA CORREÇÃO ---

    current_messages = state.get("messages", [])
    user_input_escolha = get_last_user_message_content(current_messages)

    # --- CORREÇÃO PRINCIPAL AQUI ---
    # Usar a chave correta também aqui: 'profissionais_encontrados'
    profissionais_encontrados = state.get("profissionais_encontrados", []) # Renomeado localmente para clareza
    # --- FIM DA CORREÇÃO ---

    if not user_input_escolha:
        logger.warning("[processar_escolha_profissional] Nenhuma entrada do usuário para processar como escolha.")
        return {
            "messages": current_messages,
            "fase_agendamento": "PROFISSIONAIS_APRESENTADOS_AGUARDANDO_ESCOLHA"
        }

    # --- CORREÇÃO NA MENSAGEM DE ERRO ---
    if not profissionais_encontrados: # Verifica a variável local que leu da chave correta
        logger.error("[processar_escolha_profissional] Nenhum profissional encontrado no estado ('profissionais_encontrados' está vazio ou ausente). Isso não deveria acontecer aqui.")
        # --- FIM DA CORREÇÃO NA MENSAGEM DE ERRO ---
        current_messages.append(AIMessage(content="Desculpe, ocorreu um erro interno e não consigo encontrar a lista de profissionais que apresentei. Poderíamos tentar a busca novamente?"))
        return {
            "messages": current_messages,
            "fase_agendamento": "ESPECIALIDADE_COLETADA_AGUARDANDO_PREFERENCIA_PROFISSIONAL",
            "profissionais_encontrados": None,
            "preferencia_profissional": None
        }

    profissional_selecionado_obj = None
    nome_profissional_selecionado = None
    user_input_lower = user_input_escolha.lower()

    # --- CORREÇÃO NO LOOP ---
    # Iterar sobre a variável local correta: 'profissionais_encontrados'
    for profissional_obj in profissionais_encontrados:
    # --- FIM DA CORREÇÃO ---
        nome_api = profissional_obj.get("nome")
        if nome_api:
            nome_api_lower = nome_api.lower()
            if user_input_lower == nome_api_lower or user_input_lower in nome_api_lower:
                if profissional_selecionado_obj:
                    logger.warning(f"[processar_escolha_profissional] Escolha ambígua: '{user_input_escolha}' corresponde a múltiplos profissionais.")
                    # --- CORREÇÃO NA LIST COMPREHENSION ---
                    nomes_correspondentes = [p.get("nome") for p in profissionais_encontrados if user_input_lower in p.get("nome","").lower()]
                    # --- FIM DA CORREÇÃO ---
                    mensagem_ambiguidade = (f"Sua escolha '{user_input_escolha}' corresponde a mais de um profissional: {', '.join(nomes_correspondentes)}. "
                                            "Poderia ser mais específico ou fornecer o nome completo de quem deseja?")
                    current_messages.append(AIMessage(content=mensagem_ambiguidade))
                    return {
                        "messages": current_messages,
                        "fase_agendamento": "PROFISSIONAIS_APRESENTADOS_AGUARDANDO_ESCOLHA"
                    }
                profissional_selecionado_obj = profissional_obj
                nome_profissional_selecionado = nome_api

    if profissional_selecionado_obj and nome_profissional_selecionado:
        logger.info(f"[processar_escolha_profissional] Usuário escolheu o profissional: '{nome_profissional_selecionado}'")
        mensagem_sucesso = f"Ótimo! Selecionei {nome_profissional_selecionado} para você. Agora, qual turno você prefere para o agendamento: manhã ou tarde?" # Modificado para seguir para o turno
        current_messages.append(AIMessage(content=mensagem_sucesso))
        return {
            "messages": current_messages,
            "profissional_para_agendamento": nome_profissional_selecionado,
            "id_profissional_selecionado": profissional_selecionado_obj.get("id"),
            "fase_agendamento": "AGUARDANDO_TURNO", # Fase que leva para coleta de turno
            "erro_coleta_agendamento": None,
            "profissionais_encontrados": None # Limpa a lista após a escolha
        }
    else:
        logger.warning(f"[processar_escolha_profissional] Escolha do usuário ('{user_input_escolha}') não corresponde a nenhum profissional na lista.")
        # --- CORREÇÃO NA LIST COMPREHENSION ---
        nomes_profissionais_str = ", ".join([p.get("nome") for p in profissionais_encontrados if p.get("nome")])
        # --- FIM DA CORREÇÃO ---
        mensagem_nao_encontrado = (f"Desculpe, não encontrei '{user_input_escolha}' na lista de profissionais que apresentei ({nomes_profissionais_str}). "
                                   "Por favor, verifique o nome ou escolha um da lista.")
        current_messages.append(AIMessage(content=mensagem_nao_encontrado))
        return {
            "messages": current_messages,
            "fase_agendamento": "PROFISSIONAIS_APRESENTADOS_AGUARDANDO_ESCOLHA"
        }

def lidar_com_nenhum_profissional_encontrado_placeholder(state: State) -> dict:
    logger.info("Nó 'lidar_com_nenhum_profissional_encontrado_placeholder' atingido. TODO: Implementar lógica.")
    messages = state.get("messages", [])
    # A mensagem de "nenhum profissional encontrado" já foi enviada pelo nó anterior.
    # Aqui, poderíamos perguntar: "Gostaria de tentar uma especialidade diferente ou verificar mais tarde?"
    # Por ora, vamos apenas simular que o fluxo pode terminar aqui ou aguardar nova interação.
    # Se quiséssemos ser proativos:
    # messages.append(AIMessage(content="Gostaria de buscar por outra especialidade ou posso ajudar em algo mais?"))
    # fase_agendamento = "AGUARDANDO_DECISAO_APOS_NENHUM_PROFISSIONAL"
    # Por simplicidade no placeholder, vamos apenas registrar.
    return {
        "messages": messages,
        # A fase já foi definida como "NENHUM_PROFISSIONAL_ENCONTRADO"
    }

def processar_input_turno(state: State) -> dict:
        """
        Processa a entrada do usuário para o turno (manhã/tarde) usando LLM.
        """
        logger.debug(f"Iniciando processamento do input do turno. Fase: {state.get('fase_agendamento')}")
        messages = state.get("messages", [])
        user_input = get_last_user_message_content(messages)

        if not user_input:
            logger.warning("[processar_input_turno] Nenhuma entrada do usuário para processar como turno.")
            # A mensagem anterior do AI já perguntou o turno.
            # Mantém a fase para nova tentativa do usuário.
            return {
                "messages": messages, # Mantém as mensagens atuais
                "fase_agendamento": "AGUARDANDO_TURNO"
            }

        # Prompt para o LLM
        prompt_template = ChatPromptTemplate.from_template(
            """
            Analise a seguinte resposta do usuário, que foi perguntado se prefere o período da manhã ou da tarde para um agendamento.
            Classifique a resposta em uma das seguintes categorias: "MANHA", "TARDE", ou "INVALIDO".

            Exemplos:
            - Usuário: "manhã" -> MANHA
            - Usuário: "de manhã, por favor" -> MANHA
            - Usuário: "pode ser de tarde" -> TARDE
            - Usuário: "à tarde" -> TARDE
            - Usuário: "tanto faz" -> INVALIDO
            - Usuário: "sim" -> INVALIDO
            - Usuário: "quero o dia todo" -> INVALIDO
            - Usuário: "quero cancelar" -> INVALIDO (tratar intenções de cancelamento em outro fluxo)

            Retorne APENAS a categoria ("MANHA", "TARDE", ou "INVALIDO").

            Resposta do usuário: "{user_response}"
            Categoria:
            """
        )

        chain = prompt_template | llm # Seu llm já configurado

        try:
            llm_response_str = chain.invoke({"user_response": user_input}).content.strip().upper()
            logger.info(f"[processar_input_turno] LLM classificou o turno como: '{llm_response_str}' para o input '{user_input}'")

            if llm_response_str == "MANHA":
                mensagem_confirmacao = "Entendido, preferência pelo período da manhã. Vou verificar as agendas disponíveis a partir de três dias de hoje."
                messages.append(AIMessage(content=mensagem_confirmacao))
                return {
                    "messages": messages,
                    "turno_para_agendamento": "MANHA",
                    "fase_agendamento": "TURNO_COLETADO_AGUARDANDO_HORARIOS",
                    "erro_coleta_agendamento": None
                }
            elif llm_response_str == "TARDE":
                mensagem_confirmacao = "Entendido, preferência pelo período da tarde. Vou verificar as agendas disponíveis a partir de três dias de hoje."
                messages.append(AIMessage(content=mensagem_confirmacao))
                return {
                    "messages": messages,
                    "turno_para_agendamento": "TARDE",
                    "fase_agendamento": "TURNO_COLETADO_AGUARDANDO_HORARIOS",
                    "erro_coleta_agendamento": None
                }
            else: # Inclui "INVALIDO" ou qualquer outra resposta inesperada do LLM
                logger.warning(f"[processar_input_turno] LLM retornou categoria de turno não reconhecida ou inválida: '{llm_response_str}'")
                mensagem_reprompt = "Desculpe, não entendi bem. Você prefere o período da manhã ou da tarde para o seu agendamento?"
                messages.append(AIMessage(content=mensagem_reprompt))
                return {
                    "messages": messages,
                    "turno_para_agendamento": None, # Limpa qualquer tentativa anterior
                    "fase_agendamento": "AGUARDANDO_TURNO", # Volta para o usuário responder novamente
                    "erro_coleta_agendamento": "Resposta de turno não compreendida."
                }

        except Exception as e:
            logger.exception("[processar_input_turno] Erro inesperado ao processar input do turno com LLM:")
            mensagem_erro_tecnico = "Desculpe, ocorreu um erro técnico ao processar sua preferência de turno. Poderia tentar novamente?"
            messages.append(AIMessage(content=mensagem_erro_tecnico))
            return {
                "messages": messages,
                "fase_agendamento": "AGUARDANDO_TURNO",
                "erro_coleta_agendamento": "Erro técnico no processamento do turno."
            }
        
def consultar_e_apresentar_datas_disponiveis(state: State) -> dict:
    """
    Consulta a API de datas disponíveis para o profissional no mês corrente 
    (e próximo, se necessário para obter opções) e apresenta até 3 opções ao usuário.
    A API já deve retornar apenas datas válidas (ex: a partir do próximo dia útil ou D+3).
    """
    logger.debug("Iniciando consulta e apresentação de datas disponíveis.")
    messages = state.get("messages", [])
    id_profissional = state.get("id_profissional_selecionado")
    nome_profissional = state.get("profissional_para_agendamento", "o profissional selecionado")

    if not id_profissional:
        logger.error("[consultar_e_apresentar_datas_disponiveis] ID do profissional não encontrado no estado.")
        messages.append(AIMessage(content="Desculpe, não consegui identificar o profissional para buscar as datas. Poderíamos tentar a seleção do profissional novamente?"))
        return {
            "messages": messages,
            "fase_agendamento": "ESPECIALIDADE_COLETADA_AGUARDANDO_PREFERENCIA_PROFISSIONAL", 
            "erro_coleta_agendamento": "ID do profissional ausente para busca de datas."
        }

    # 1. Determinar mês/ano para consulta (começando pelo mês atual)
    hoje = date.today()
    
    datas_validas_api = []
    # Consultar para o mês atual e o próximo para garantir opções
    for i in range(2): # Consultar 2 meses
        data_alvo_consulta = hoje + timedelta(days=30 * i) # Aproximação para o próximo mês se i=1
        mes_consulta = data_alvo_consulta.strftime("%m")
        ano_consulta = data_alvo_consulta.strftime("%Y")
        
        url = f"https://back.homologacao.apphealth.com.br:9090/api-vizi/agenda/profissionais/{id_profissional}/datas"
        params = {"mes": mes_consulta, "ano": ano_consulta}
        logger.info(f"Consultando API de datas: {url} com params: {params}")

        try:
            response = requests.get(url, headers=api_headers, params=params, timeout=10)
            response.raise_for_status()
            datas_mes_api = response.json()

            if datas_mes_api and isinstance(datas_mes_api, list):
                for item_data in datas_mes_api:
                    if isinstance(item_data, dict) and "data" in item_data:
                        # A API já deve retornar datas válidas, então não precisamos mais do filtro data_api_obj >= data_inicio_busca
                        datas_validas_api.append(item_data["data"])
            
            # Remover duplicatas (caso a API retorne para múltiplos perfis/unidades na mesma data) e ordenar
            datas_validas_api = sorted(list(set(datas_validas_api)))

        except requests.exceptions.HTTPError as e:
            logger.error(f"Erro HTTP ao consultar API de datas: {e.response.status_code} - {e.response.text}")
            messages.append(AIMessage(content=f"Desculpe, tive um problema ao buscar as datas disponíveis (código {e.response.status_code}). Por favor, tente mais tarde."))
            return {
                "messages": messages,
                "fase_agendamento": "ERRO_API_DATAS",
                "erro_coleta_agendamento": "Erro HTTP na API de datas."
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro de requisição ao consultar API de datas: {e}")
            messages.append(AIMessage(content="Desculpe, não consegui me conectar ao sistema para buscar as datas. Verifique sua conexão ou tente mais tarde."))
            return {
                "messages": messages,
                "fase_agendamento": "ERRO_API_DATAS",
                "erro_coleta_agendamento": "Erro de conexão na API de datas."
            }
        
        # Se já temos 3 ou mais datas, podemos parar de consultar meses seguintes
        # Isso é importante para não fazer chamadas desnecessárias se o primeiro mês já deu opções suficientes.
        if len(datas_validas_api) >= 3:
            break
            
    if not datas_validas_api:
        logger.info(f"Nenhuma data disponível encontrada para o Dr(a). {nome_profissional} nos próximos períodos consultados.")
        # A mensagem pode ser genérica, já que não temos mais uma "data de início" específica para mencionar.
        messages.append(AIMessage(content=f"No momento, não encontrei datas disponíveis para o(a) Dr(a). {nome_profissional}. Gostaria de tentar com outro profissional ou especialidade?"))
        return {
            "messages": messages,
            "fase_agendamento": "NENHUMA_DATA_DISPONIVEL",
            "datas_disponiveis_apresentadas": []
        }

    # 2. Apresentar até 3 datas
    datas_para_apresentar = datas_validas_api[:3]
    datas_formatadas_usuario = [datetime.strptime(d, "%Y-%m-%d").strftime("%d/%m/%Y") for d in datas_para_apresentar]
    
    # A mensagem não precisa mais mencionar "a partir de..."
    mensagem_usuario = f"Encontrei as seguintes datas disponíveis para o(a) Dr(a). {nome_profissional}:\n"
    for i, data_fmt in enumerate(datas_formatadas_usuario):
        mensagem_usuario += f"{i+1}. {data_fmt}\n"
    mensagem_usuario += "Qual data você prefere? (Digite o número ou a data completa)"

    messages.append(AIMessage(content=mensagem_usuario))
    
    return {
        "messages": messages,
        "fase_agendamento": "DATAS_APRESENTADAS_AGUARDANDO_ESCOLHA",
        "datas_disponiveis_apresentadas": datas_para_apresentar,
        "erro_coleta_agendamento": None
    }

    # agent.ipynb
    # ... (código existente) ...
    
def processar_escolha_data_e_buscar_horarios(state: State) -> dict:
    logger.debug("[processar_escolha_data_e_buscar_horarios] Iniciando.")
    messages = state.get("messages", []).copy()
    user_input_data_escolha = get_last_user_message_content(messages)
    
    datas_apresentadas_api_format = state.get("datas_disponiveis_apresentadas", []) # Formato "AAAA-MM-DD"
    id_profissional = state.get("id_profissional_selecionado")
    turno_escolhido = state.get("turno_para_agendamento") # "MANHA" ou "TARDE"
    nome_profissional = state.get("profissional_para_agendamento", "o profissional selecionado")

    if not user_input_data_escolha:
        logger.warning("[processar_escolha_data_e_buscar_horarios] Nenhuma entrada do usuário para escolha da data.")
        # A mensagem anterior já pedia a data, então apenas aguardamos.
        # Não precisa adicionar nova mensagem, manter a fase.
        return {"messages": messages, "fase_agendamento": "DATAS_APRESENTADAS_AGUARDANDO_ESCOLHA"}

    if not datas_apresentadas_api_format or not id_profissional or not turno_escolhido:
        logger.error("[processar_escolha_data_e_buscar_horarios] Estado incompleto: datas apresentadas, ID do profissional ou turno ausente.")
        messages.append(AIMessage(content="Desculpe, ocorreu um erro interno e não consigo processar sua escolha de data. Poderia tentar o agendamento desde o início?"))
        return {"messages": messages, "fase_agendamento": None, "erro_coleta_agendamento": "Estado incompleto para buscar horários."}

    data_selecionada_api_format = None # "AAAA-MM-DD"

    # Validar escolha do usuário (número da opção ou data)
    try:
        escolha_int = int(user_input_data_escolha)
        if 1 <= escolha_int <= len(datas_apresentadas_api_format):
            data_selecionada_api_format = datas_apresentadas_api_format[escolha_int - 1]
        else:
            raise ValueError("Número da opção inválido.")
    except ValueError: # Não é um número, tentar parsear como data DD/MM/AAAA
        try:
            data_obj_usuario = datetime.strptime(user_input_data_escolha, "%d/%m/%Y")
            data_str_usuario_api_format = data_obj_usuario.strftime("%Y-%m-%d")
            if data_str_usuario_api_format in datas_apresentadas_api_format:
                data_selecionada_api_format = data_str_usuario_api_format
            else:
                logger.warning(f"[processar_escolha_data_e_buscar_horarios] Data '{user_input_data_escolha}' não estava na lista apresentada.")
                # Não considerar como válida, mesmo que seja uma data real.
                data_selecionada_api_format = None 
        except ValueError:
            logger.warning(f"[processar_escolha_data_e_buscar_horarios] Entrada '{user_input_data_escolha}' não é número de opção válido nem data DD/MM/AAAA reconhecida.")
            data_selecionada_api_format = None

    if not data_selecionada_api_format:
        datas_formatadas_usuario_reprompt = [datetime.strptime(d, "%Y-%m-%d").strftime("%d/%m/%Y") for d in datas_apresentadas_api_format]
        msg_reprompt = f"Desculpe, não entendi sua escolha. Por favor, digite o número da opção ou a data completa de uma das seguintes datas:\n"
        for i, data_fmt in enumerate(datas_formatadas_usuario_reprompt):
            msg_reprompt += f"{i+1}. {data_fmt}\n"
        messages.append(AIMessage(content=msg_reprompt))
        return {
            "messages": messages,
            "fase_agendamento": "DATAS_APRESENTADAS_AGUARDANDO_ESCOLHA"
        }

    logger.info(f"[processar_escolha_data_e_buscar_horarios] Usuário escolheu data: {data_selecionada_api_format}")
    
    # 2. Chamar API de Horários
    url_horarios = f"https://back.homologacao.apphealth.com.br:9090/api-vizi/agenda/profissionais/{id_profissional}/horarios"
    params_horarios = {"data": data_selecionada_api_format}
    
    logger.info(f"Consultando API de horários: {url_horarios} com params: {params_horarios}")
    try:
        response_horarios = requests.get(url_horarios, headers=api_headers, params=params_horarios, timeout=10)
        response_horarios.raise_for_status()
        horarios_da_api = response_horarios.json() # Lista de {"horaInicio": "HH:MM:SS", ...}
    except requests.exceptions.HTTPError as e:
        logger.error(f"Erro HTTP ao consultar API de horários para data {data_selecionada_api_format}: {e.response.status_code} - {e.response.text}")
        messages.append(AIMessage(content=f"Desculpe, tive um problema ao buscar os horários para {datetime.strptime(data_selecionada_api_format, '%Y-%m-%d').strftime('%d/%m/%Y')} (código {e.response.status_code}). Gostaria de tentar outra data?"))
        # Volta para escolher outra data
        return {
            "messages": messages, 
            "fase_agendamento": "DATAS_APRESENTADAS_AGUARDANDO_ESCOLHA", # Permite escolher outra data da lista anterior
            "data_agendamento_escolhida": None # Limpa a tentativa
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro de requisição ao consultar API de horários: {e}")
        messages.append(AIMessage(content=f"Desculpe, não consegui buscar os horários para {datetime.strptime(data_selecionada_api_format, '%Y-%m-%d').strftime('%d/%m/%Y')}. Verifique sua conexão ou tente mais tarde. Gostaria de escolher outra data?"))
        return {
            "messages": messages, 
            "fase_agendamento": "DATAS_APRESENTADAS_AGUARDANDO_ESCOLHA",
            "data_agendamento_escolhida": None
        }

    # 3. Filtrar Horários pelo Turno
    horarios_filtrados_turno = []
    if horarios_da_api and isinstance(horarios_da_api, list):
        for horario_obj_api in horarios_da_api:
            hora_inicio_str = horario_obj_api.get("horaInicio")
            if hora_inicio_str:
                try:
                    hora_inicio_time = datetime.strptime(hora_inicio_str, "%H:%M:%S").time()
                    
                    if turno_escolhido == "MANHA" and hora_inicio_time < time(12, 0):
                        horarios_filtrados_turno.append(hora_inicio_time.strftime("%H:%M"))
                    elif turno_escolhido == "TARDE" and time(12, 0) <= hora_inicio_time < time(18,0): # Ex: Tarde até 18h
                        horarios_filtrados_turno.append(hora_inicio_time.strftime("%H:%M"))
                except ValueError:
                    logger.warning(f"Formato de horaInicio inválido da API: {hora_inicio_str}")
        
        horarios_filtrados_turno = sorted(list(set(horarios_filtrados_turno))) # Ordena e remove duplicatas

    # 4. Apresentar Horários Filtrados
    data_escolhida_fmt_usuario = datetime.strptime(data_selecionada_api_format, '%Y-%m-%d').strftime('%d/%m/%Y')
    
    if not horarios_filtrados_turno:
        logger.info(f"Nenhum horário disponível para {data_escolhida_fmt_usuario} no período da {turno_escolhido.lower()}.")
        messages.append(AIMessage(content=f"Para a data {data_escolhida_fmt_usuario}, não encontrei horários disponíveis no período da {turno_escolhido.lower()}. Gostaria de escolher outra data da lista anterior?"))
        return {
            "messages": messages,
            "fase_agendamento": "DATAS_APRESENTADAS_AGUARDANDO_ESCOLHA", # Volta para escolher outra data
            "data_agendamento_escolhida": None, # Limpa tentativa
            "horarios_disponiveis_apresentados": []
        }
    else:
        horarios_para_apresentar = horarios_filtrados_turno[:5] # Limitar a 5 opções
        
        msg_horarios = f"Para {data_escolhida_fmt_usuario} no período da {turno_escolhido.lower()}, encontrei estes horários disponíveis para o(a) Dr(a). {nome_profissional}:\n"
        for i, hora_fmt in enumerate(horarios_para_apresentar):
            msg_horarios += f"{i+1}. {hora_fmt}\n"
        msg_horarios += "Qual horário você prefere? (Digite o número)"
        
        messages.append(AIMessage(content=msg_horarios))
        return {
            "messages": messages,
            "data_agendamento_escolhida": data_selecionada_api_format, # Guarda a data que gerou esses horários
            "horarios_disponiveis_apresentados": horarios_para_apresentar, # Lista de strings "HH:MM"
            "fase_agendamento": "HORARIOS_APRESENTADOS_AGUARDANDO_ESCOLHA",
            "erro_coleta_agendamento": None
        }

def processar_escolha_horario(state: State) -> State:
    """
    Processa a escolha do horário feita pelo usuário.
    Valida a entrada, armazena o horário escolhido e prepara para a confirmação final.
    """
    user_input = ""
    if state.get("messages") and isinstance(state["messages"][-1], HumanMessage):
        user_input = state["messages"][-1].content.strip()

    horarios_apresentados = state.get("horarios_disponiveis_apresentados")
    profissional_nome = state.get("profissional_para_agendamento", "o profissional selecionado") # Nome do profissional
    data_escolhida_formatada = state.get("data_agendamento_escolhida_formatada", state.get("data_agendamento_escolhida"))


    if not user_input:
        logger.warning("[processar_escolha_horario] Nenhuma entrada do usuário para processar.")
        # Adiciona uma mensagem para o usuário pedindo para tentar novamente
        ai_message = AIMessage(content="Por favor, digite o número do horário que você deseja.")
        return {
            "messages": state.get("messages", []) + [ai_message],
            "fase_agendamento": "HORARIOS_APRESENTADOS_AGUARDANDO_ESCOLHA" # Mantém na mesma fase
        }

    if not horarios_apresentados:
        logger.error("[processar_escolha_horario] Não há horários apresentados no estado para escolher.")
        ai_message = AIMessage(content="Desculpe, houve um problema e não consigo ver os horários que te mostrei. Poderíamos tentar buscar os horários novamente?")
        # Idealmente, aqui você poderia ter uma lógica para voltar para uma fase anterior ou pedir ajuda.
        return {
            "messages": state.get("messages", []) + [ai_message],
            "fase_agendamento": "TURNO_COLETADO_AGUARDANDO_HORARIOS" # Ou uma fase de erro
        }

    try:
        escolha_idx = int(user_input) - 1 # Ajusta para índice baseado em zero
        if 0 <= escolha_idx < len(horarios_apresentados):
            horario_escolhido = horarios_apresentados[escolha_idx]
            logger.info(f"[processar_escolha_horario] Usuário escolheu o horário: {horario_escolhido} (entrada: '{user_input}')")

            # Preparar para a próxima fase: confirmação final
            # (Você pode querer um nó separado para a chamada de API de agendamento)
            nova_fase = 'PROCESSAR_CONFIRMACAO_FINAL_AGENDAMENTO' # Defina esta fase

            profissional_nome = state.get("profissional_para_agendamento", "o profissional selecionado")
            data_escolhida_str = state.get("data_agendamento_escolhida") # Formato "YYYY-MM-DD"
            data_escolhida_formatada_usuario = "data não encontrada"

            if data_escolhida_str:
                try:
                    data_escolhida_formatada_usuario = datetime.strptime(data_escolhida_str, '%Y-%m-%d').strftime('%d/%m/%Y')
                except ValueError:
                    logger.error(f"Erro ao formatar data_agendamento_escolhida: {data_escolhida_str}")

            confirmacao_mensagem = (
                f"Ótimo! Seu agendamento com {profissional_nome} para o dia {data_escolhida_formatada_usuario} às {horario_escolhido} está quase pronto. "
                f"Posso confirmar?"
            )

            ai_message = AIMessage(content=confirmacao_mensagem)

            return {
                "messages": state.get("messages", []) + [ai_message],
                "horario_agendamento_escolhido": horario_escolhido,
                "fase_agendamento": nova_fase,
                "erro_coleta_agendamento": None
            }
        else:
            logger.warning(f"[processar_escolha_horario] Escolha inválida: '{user_input}'. Não está na faixa de opções.")
            ai_message = AIMessage(content=f"Opção '{user_input}' inválida. Por favor, escolha um número da lista de horários que apresentei.")
            return {
                "messages": state.get("messages", []) + [ai_message],
                "fase_agendamento": "HORARIOS_APRESENTADOS_AGUARDANDO_ESCOLHA", # Mantém na mesma fase
                "erro_coleta_agendamento": "Escolha de horário inválida"
            }
    except ValueError:
        logger.warning(f"[processar_escolha_horario] Entrada não numérica: '{user_input}'.")
        ai_message = AIMessage(content="Por favor, digite apenas o número correspondente ao horário desejado.")
        return {
            "messages": state.get("messages", []) + [ai_message],
            "fase_agendamento": "HORARIOS_APRESENTADOS_AGUARDANDO_ESCOLHA", # Mantém na mesma fase
            "erro_coleta_agendamento": "Entrada de horário não numérica"
        }
    except Exception as e:
        logger.error(f"[processar_escolha_horario] Erro inesperado: {e}")
        ai_message = AIMessage(content="Desculpe, ocorreu um erro inesperado ao processar sua escolha de horário. Vamos tentar novamente.")
        return {
            "messages": state.get("messages", []) + [ai_message],
            "fase_agendamento": "HORARIOS_APRESENTADOS_AGUARDANDO_ESCOLHA",
            "erro_coleta_agendamento": "Erro inesperado no processamento do horário"
        }
    
def processar_confirmacao_final_agendamento(state: State) -> dict:
        """
        Placeholder para processar a resposta do usuário à confirmação final.
        Aqui viria a lógica para chamar a API de agendamento se 'sim',
        ou voltar/cancelar se 'não'.
        """
        logger.debug("[processar_confirmacao_final_agendamento] Iniciando.")
        messages = state.get("messages", [])
        user_input = get_last_user_message_content(messages)
        horario = state.get("horario_agendamento_escolhido")
        data = state.get("data_agendamento_escolhida") # Formato AAAA-MM-DD
        profissional = state.get("profissional_para_agendamento")

        # Simplificação: Assume "sim" e finaliza
        if user_input and any(term in user_input.lower() for term in ["sim", "s", "confirmo", "pode"]):
            logger.info(f"Usuário confirmou agendamento para {data} às {horario} com {profissional}.")
            # Aqui você chamaria a API real de agendamento.
            mensagem_final = f"Perfeito! Seu agendamento para {datetime.strptime(data, '%Y-%m-%d').strftime('%d/%m/%Y')} às {horario} com {profissional} foi confirmado com sucesso!"
            messages.append(AIMessage(content=mensagem_final))
            # Limpa estado relevante para não interferir em futuras interações
            return {
                "messages": messages,
                "fase_agendamento": "AGENDAMENTO_CONCLUIDO", # Fase final
                "horario_agendamento_escolhido": None,
                "data_agendamento_escolhida": None,
                # Manter nome, especialidade, profissional para possível consulta futura no histórico
            }
        elif user_input and any(term in user_input.lower() for term in ["não", "n", "cancela", "mudei"]):
             logger.info("Usuário cancelou a confirmação final.")
             messages.append(AIMessage(content="Entendido. O agendamento não foi confirmado. Posso ajudar com mais alguma coisa?"))
             # Limpa estado relevante
             return {
                "messages": messages,
                "fase_agendamento": None, # Volta ao início do fluxo
                # Limpar todos os dados coletados
                "nome_para_agendamento": None,
                "especialidade_para_agendamento": None,
                "especialidade_id_para_agendamento": None,
                "preferencia_profissional": None,
                "profissional_para_agendamento": None,
                "id_profissional_selecionado": None,
                "turno_para_agendamento": None,
                "data_agendamento_escolhida": None,
                "horario_agendamento_escolhido": None,
                "datas_disponiveis_apresentadas": None,
                "horarios_disponiveis_apresentados": None,
             }
        else:
            logger.warning(f"Resposta de confirmação final não entendida: '{user_input}'")
            messages.append(AIMessage(content="Desculpe, não entendi. Posso confirmar o agendamento (sim ou não)?"))
            return {
                "messages": messages,
                "fase_agendamento": 'PROCESSAR_CONFIRMACAO_FINAL_AGENDAMENTO' # Mantém na fase de confirmação
            }

# %%
# Lógica de Roteamento Principal (após categorização)
def route_after_categorization(state: State) -> str:
    categoria_atual = state.get("categoria")
    print(f"DEBUG [route_after_categorization]: Categoria='{categoria_atual}'")

    if categoria_atual == "Informações Unidade":
        return "ROTA_CONSULTA_UNIDADE"
    elif categoria_atual == "Inserir Agendamento":
        # Verifica se já estamos em um fluxo de agendamento (pouco provável aqui se 'checar_fase' já direcionou)
        # Mas é uma boa salvaguarda ou se o fluxo for diferente.
        if state.get("fase_agendamento"):
             # Se por algum motivo categorizou de novo no meio de um agendamento, vai para o processamento de input
            return "ROTA_PROCESSAR_INPUT_AGENDAMENTO"
        return "ROTA_INICIAR_COLETA_NOME_AGENDAMENTO"
    elif categoria_atual == "Saudação ou Despedida":
        return "ROTA_SAUDACAO_DESPEDIDA"
    else: # "Consultar Agendamento", "Atualizar Agendamento", "Cancelar Agendamento", "Fora do Escopo", "Indefinido"
        # Todos os outros, por enquanto, vão para fora do escopo ou uma resposta padrão.
        # No futuro, cada um teria sua própria rota.
        return "ROTA_FORA_ESCOPO"

def route_entry_logic(state: State) -> str:
        fase = state.get("fase_agendamento")
        logger.debug(f"DEBUG [route_entry_logic]: Fase Agendamento='{fase}'")

        if fase == "AGUARDANDO_NOME":
            return "CONTINUAR_AGENDAMENTO"
        elif fase == "AGUARDANDO_ESPECIALIDADE":
            return "PROCESSAR_ESPECIALIDADE"
        elif fase == "ESPECIALIDADE_COLETADA_AGUARDANDO_PREFERENCIA_PROFISSIONAL":
            return "PROCESSAR_PREFERENCIA_PROFISSIONAL"
        elif fase == "AGUARDANDO_NOME_PROFISSIONAL_INDICADO":
            return "PROCESSAR_NOME_PROFISSIONAL_INDICADO"
        elif fase == "NOME_PROFISSIONAL_INDICADO_AGUARDANDO_CONFIRMACAO":
            return "PROCESSAR_CONFIRMACAO_NOME_PROFISSIONAL"
        # --- REMOVER CONDIÇÕES ABAIXO ---
        # elif fase == "PROFISSIONAL_ESPECIFICO_CONFIRMADO_AGUARDANDO_TURNO": # Fase após confirmar nome específico
        #      logger.info(f"Fase {fase}. Profissional específico confirmado. Indo para coleta de turno.")
        #      return "INICIAR_COLETA_TURNO" # REMOVIDO
        # --- FIM DA REMOÇÃO ---
        elif fase == "PROFISSIONAIS_APRESENTADOS_AGUARDANDO_ESCOLHA":
            return "PROCESSAR_ESCOLHA_PROFISSIONAL"
        # --- REMOVER CONDIÇÃO ABAIXO ---
        # elif fase == "PROFISSIONAL_RECOMENDADO_ESCOLHIDO": # Fase após escolher profissional recomendado
        #     logger.info(f"Fase {fase}. Profissional recomendado escolhido. Indo para coleta de turno.")
        #     return "INICIAR_COLETA_TURNO" # REMOVIDO
        # --- FIM DA REMOÇÃO ---
        elif fase == "AGUARDANDO_TURNO":
                return "PROCESSAR_TURNO" # Esta rota agora será usada após escolher/confirmar profissional
        elif fase == "TURNO_COLETADO_AGUARDANDO_HORARIOS":
            logger.info(f"Fase {fase}. Turno coletado. Indo para consulta de datas disponíveis.")
            return "CONSULTAR_DATAS_DISPONIVEIS"
        elif fase == "DATAS_APRESENTADAS_AGUARDANDO_ESCOLHA":
            logger.info(f"Fase {fase}. Datas apresentadas. Aguardando escolha da data.")
            return "PROCESSAR_ESCOLHA_DATA"
        elif fase == 'HORARIOS_APRESENTADOS_AGUARDANDO_ESCOLHA':
            logger.info(f"Fase {fase}. Horários apresentados. Aguardando escolha do horário.")
            return "PROCESSAR_ESCOLHA_HORARIO"
        elif fase == 'PROCESSAR_CONFIRMACAO_FINAL_AGENDAMENTO':
            logger.info(f"Fase {fase}. Pronto para confirmação final.")
            return "ROTA_CONFIRMACAO_FINAL"
        elif fase == "AGENDAMENTO_CONCLUIDO":
             logger.info(f"Fase {fase}. Agendamento concluído.")
             return END
        elif fase == "NENHUM_PROFISSIONAL_ENCONTRADO":
             logger.info(f"Fase {fase}. Nenhum profissional encontrado.")
             return "LIDAR_NENHUM_PROFISSIONAL"
        elif fase in ["ERRO_CONSULTA_API_PROFISSIONAIS", "ERRO_PROCESSAMENTO_NOMES_PROFISSIONAIS", "NENHUMA_DATA_DISPONIVEL", "ERRO_API_DATAS"]:
            logger.warning(f"Fase de erro ou sem opções: {fase}. Finalizando fluxo.")
            return END

        logger.debug(f"[route_entry_logic] Nenhuma fase de agendamento ativa ou correspondente ('{fase}'). Roteando para NOVA_CONSULTA para categorização.")
        return "NOVA_CONSULTA"

def route_after_processing_nome(state: State) -> str:
    if state.get("erro_coleta_agendamento"):
        print(f"DEBUG [route_after_processing_nome]: Erro na coleta '{state.get('erro_coleta_agendamento')}'. Esperando nova tentativa do usuário.")
        return END
    elif state.get("fase_agendamento") == "NOME_COLETADO":
        # Agora, ao invés de finalizar, vamos para a coleta de especialidade
        print("DEBUG [route_after_processing_nome]: Nome coletado. Iniciando coleta de especialidade.")
        return "COLETAR_ESPECIALIDADE"
    else:
        print(f"AVISO [route_after_processing_nome]: Estado inesperado após processar nome. Fase: {state.get('fase_agendamento')}")
        return END
    
def route_after_processing_preferencia_profissional(state: State) -> str:
    preferencia = state.get("preferencia_profissional")
    fase_agendamento_atual = state.get("fase_agendamento")

    logger.debug(f"[route_after_processing_preferencia_profissional] Preferência: '{preferencia}', Fase: '{fase_agendamento_atual}'")

    # Caso o nó anterior (processar_preferencia_profissional) tenha pedido um reprompt
    if fase_agendamento_atual == "ESPECIALIDADE_COLETADA_AGUARDANDO_PREFERENCIA_PROFISSIONAL" and not preferencia:
        logger.info("[route_after_processing_preferencia_profissional] Preferência não determinada após reprompt, aguardando nova entrada.")
        return END # Espera nova entrada do usuário

    # Roteamento baseado na preferência processada
    if preferencia == "pedir_recomendacao":
        logger.info("[route_after_processing_preferencia_profissional] Usuário pediu recomendação. Roteando para preparar consulta API.")
        return "ROTA_PARA_PREPARAR_CONSULTA_API" # Deverá ir para 'preparar_consulta_api_profissionais'
    
    elif preferencia == "escolher_especifico_sem_nome_ainda":
        logger.info("[route_after_processing_preferencia_profissional] Usuário quer escolher, mas sem nome ainda. Roteando para iniciar coleta de nome.")
        return "ROTA_PARA_INICIAR_COLETA_NOME_PROFISSIONAL" # Deverá ir para 'iniciar_coleta_nome_profissional'

    elif preferencia == "escolher_especifico_com_nome_fornecido":
        logger.info("[route_after_processing_preferencia_profissional] Usuário forneceu nome. Roteando para preparar consulta API (que validará o nome).")
        return "ROTA_PARA_PREPARAR_CONSULTA_API" # Deverá ir para 'preparar_consulta_api_profissionais'
        
    else:
        # Se a preferência não for nenhuma das esperadas ou se a fase não for PREFERENCIA_PROFISSIONAL_PROCESSADA
        # (o que seria estranho aqui, mas é uma salvaguarda)
        # Isso pode indicar que processar_preferencia_profissional não conseguiu definir uma preferência clara
        # e já deve ter pedido um reprompt ao usuário.
        logger.warning(f"[route_after_processing_preferencia_profissional] Preferência não clara ('{preferencia}') ou fluxo inesperado. Fase: '{fase_agendamento_atual}'. Aguardando nova entrada.")
        # O nó processar_preferencia_profissional já deve ter lidado com o reprompt se necessário.
        # Aqui, apenas esperamos a próxima entrada do usuário.
        return END
    
def route_after_preparar_consulta_api(state: State) -> str:
    """
    Decide o próximo nó após a tentativa de preparar a consulta à API de profissionais (obter ID da especialidade).
    """
    logger.debug(f"Avaliando rota após preparação da consulta API. Fase: {state.get('fase_agendamento')}") # Adicionei logger para consistência

    if state.get("fase_agendamento") == "PRONTO_PARA_CONSULTAR_API_PROFISSIONAIS":
        logger.info("Estado PRONTO_PARA_CONSULTAR_API_PROFISSIONAIS. Roteando para 'consultar_api_profissionais'.")
        return "consultar_api_profissionais" # <--- AJUSTE PRINCIPAL AQUI
    else:
        # Se não estiver "PRONTO_PARA_CONSULTAR_API_PROFISSIONAIS", significa que
        # preparar_consulta_api_profissionais encontrou um problema, já deve ter
        # enviado uma mensagem ao usuário e definido a fase para aguardar nova entrada.
        # Retornar END aqui significa que o grafo encerra este ciclo de processamento,
        # e o sistema aguardará a próxima interação do usuário.
        logger.info(f"Estado não pronto para consulta ({state.get('fase_agendamento')}). Roteando para END.")
        return END
    
def route_after_consultar_api_profissionais(state: State) -> str:
    """
    Decide o próximo nó após a tentativa de consultar a API de profissionais.
    """
    fase_atual = state.get("fase_agendamento")
    logger.debug(f"Avaliando rota após consulta à API de profissionais. Fase: {fase_atual}")

    if fase_atual == "PROFISSIONAIS_APRESENTADOS_AGUARDANDO_ESCOLHA":
        # O agente apresentou os profissionais e agora precisa processar a escolha do usuário.
        # Vamos precisar de um novo nó para isso, por exemplo, "processar_escolha_profissional"
        logger.info("Profissionais apresentados. Roteando para 'processar_escolha_profissional'.")
        return "processar_escolha_profissional" # Novo nó que precisaremos definir
    elif fase_atual == "NENHUM_PROFISSIONAL_ENCONTRADO":
        # Nenhum profissional foi encontrado. O agente informou isso.
        # Poderíamos ter um nó para perguntar se o usuário quer tentar outra especialidade,
        # ou simplesmente terminar o fluxo de agendamento específico aqui.
        # Por enquanto, vamos direcionar para um nó "lidar_com_nenhum_profissional_encontrado".
        logger.info("Nenhum profissional encontrado. Roteando para 'lidar_com_nenhum_profissional_encontrado'.")
        return "lidar_com_nenhum_profissional_encontrado" # Novo nó que precisaremos definir
    elif fase_atual in [
        "ERRO_PROCESSAMENTO_NOMES_PROFISSIONAIS",
        "ERRO_API_PROFISSIONAIS",
        "ERRO_CONEXAO_API_PROFISSIONAIS",
        "ERRO_INESPERADO_BUSCA_PROFISSIONAIS",
        "ERRO_INTERNO_BUSCA_PROFISSIONAIS", # Adicionando erro da função consultar_api (se ID não encontrado)
        "ERRO_CONFIGURACAO_API_HEADERS" # Adicionando erro da função consultar_api (se headers não encontrados)
    ]:
        # Em todos esses casos de erro, a função consultar_api_profissionais (ou a preparação)
        # já deve ter adicionado uma mensagem informativa ao usuário.
        # Podemos simplesmente terminar o fluxo (END) para que o usuário veja a mensagem
        # e decida o que fazer (tentar novamente, mudar de assunto, etc.).
        logger.warning(f"Erro durante ou após consulta de profissionais: {fase_atual}. Roteando para END.")
        return END
    else:
        # Caso alguma fase inesperada apareça.
        logger.error(f"Fase de agendamento desconhecida após consulta de profissionais: {fase_atual}. Roteando para END.")
        return END

# %%
# Primeiro, vamos limpar todas as arestas existentes (opcional, mas pode ajudar em caso de conflito)
workflow = StateGraph(State)

# Adicionar nós
workflow.add_node("checar_fase_ou_categorizar", lambda s: s)
workflow.add_node("categorize", categorize)
workflow.add_node("iniciar_coleta_nome_agendamento", iniciar_coleta_nome_agendamento)
workflow.add_node("processar_input_nome_agendamento", processar_input_nome_agendamento)
workflow.add_node("iniciar_coleta_especialidade", iniciar_coleta_especialidade)
workflow.add_node("processar_input_especialidade", processar_input_especialidade)
workflow.add_node("processar_preferencia_profissional", processar_preferencia_profissional)
workflow.add_node("iniciar_coleta_nome_profissional", iniciar_coleta_nome_profissional)
workflow.add_node("processar_input_nome_profissional_indicado", processar_input_nome_profissional_indicado) 
workflow.add_node("gerar_resposta_consulta_unidade", gerar_resposta_informacao_unidade)
workflow.add_node("gerar_resposta_saudacao_despedida", gerar_resposta_saudacao_despedida)
workflow.add_node("gerar_resposta_fora_escopo", gerar_resposta_fora_escopo)
workflow.add_node("preparar_consulta_api_profissionais", preparar_consulta_api_profissionais)
workflow.add_node("processar_confirmacao_nome_profissional", processar_confirmacao_nome_profissional)
workflow.add_node("consultar_api_profissionais", consultar_api_profissionais)
workflow.add_node("processar_escolha_profissional", processar_escolha_profissional)
workflow.add_node("lidar_com_nenhum_profissional_encontrado", lidar_com_nenhum_profissional_encontrado_placeholder)
workflow.add_node("executar_consulta_profissionais_e_apresentar", executar_consulta_profissionais_e_apresentar)
workflow.add_node("processar_input_turno", processar_input_turno)
workflow.add_node("consultar_e_apresentar_datas_disponiveis", consultar_e_apresentar_datas_disponiveis)
workflow.add_node("processar_escolha_data_e_buscar_horarios", processar_escolha_data_e_buscar_horarios)
workflow.add_node("processar_escolha_horario", processar_escolha_horario)
workflow.add_node("processar_confirmacao_final", processar_confirmacao_final_agendamento)
# Adicionar Arestas
# Ponto de entrada vai para o nó de decisão inicial
workflow.add_edge(START, "checar_fase_ou_categorizar")

# Roteamento a partir da checagem de fase
workflow.add_conditional_edges(
        "checar_fase_ou_categorizar",
        route_entry_logic, # Use a função corrigida acima
        {
            "CONTINUAR_AGENDAMENTO": "processar_input_nome_agendamento",
            "PROCESSAR_ESPECIALIDADE": "processar_input_especialidade",
            "PROCESSAR_PREFERENCIA_PROFISSIONAL": "processar_preferencia_profissional",
            "PROCESSAR_NOME_PROFISSIONAL_INDICADO": "processar_input_nome_profissional_indicado",
            "PROCESSAR_CONFIRMACAO_NOME_PROFISSIONAL": "processar_confirmacao_nome_profissional",
            # Removido: "FINALIZAR_FLUXO_PROFISSIONAL_ESPECIFICO": END, (não é mais usado)
            # Removido: "PROCESSAR_ESCOLHA_PROFISSIONAL_RECOMENDADO": "placeholder_processar_escolha_profissional_recomendado", (não é mais usado)
            "PROCESSAR_ESCOLHA_PROFISSIONAL": "processar_escolha_profissional",
            "PROCESSAR_TURNO": "processar_input_turno",
            "CONSULTAR_DATAS_DISPONIVEIS": "consultar_e_apresentar_datas_disponiveis",
            "PROCESSAR_ESCOLHA_DATA": "processar_escolha_data_e_buscar_horarios",
            "PROCESSAR_ESCOLHA_HORARIO": "processar_escolha_horario",
            "ROTA_CONFIRMACAO_FINAL": "processar_confirmacao_final", # <<< NOVA ARESTA ADICIONADA
            "LIDAR_NENHUM_PROFISSIONAL": "lidar_com_nenhum_profissional_encontrado", # <<< CHAVE CORRIGIDA/ADICIONADA
            "NOVA_CONSULTA": "categorize",
             END: END # Adicionar mapeamento explícito para END retornado por route_entry_logic
            # Removido: "FINALIZAR_FLUXO_SEM_RECOMENDACOES": END, (tratado por END)
            # Removido: "FINALIZAR_FLUXO_ERRO_API": END, (tratado por END)
        }
    )

# Roteamento após a categorização
workflow.add_conditional_edges(
    "categorize",
    route_after_categorization,
    {
        "ROTA_CONSULTA_UNIDADE": "gerar_resposta_consulta_unidade",
        "ROTA_INICIAR_COLETA_NOME_AGENDAMENTO": "iniciar_coleta_nome_agendamento",
        "ROTA_PROCESSAR_INPUT_AGENDAMENTO": "processar_input_nome_agendamento",
        "ROTA_SAUDACAO_DESPEDIDA": "gerar_resposta_saudacao_despedida",
        "ROTA_FORA_ESCOPO": "gerar_resposta_fora_escopo"
    }
)

# Após iniciar a coleta de nome, o fluxo termina para esperar a resposta do usuário
workflow.add_edge("iniciar_coleta_nome_agendamento", END)

# Após processar o input do nome
workflow.add_conditional_edges(
    "processar_input_nome_agendamento",
    route_after_processing_nome,
    {
        "COLETAR_ESPECIALIDADE": "iniciar_coleta_especialidade",
        END: END
    }
)

workflow.add_conditional_edges(
    "processar_preferencia_profissional", # Nó de origem
    route_after_processing_preferencia_profissional, # Função de roteamento que acabamos de ajustar
    {
        # Mapeamento das strings retornadas pela função de roteamento para os nós de destino REAIS:
        "ROTA_PARA_PREPARAR_CONSULTA_API": "preparar_consulta_api_profissionais",
        "ROTA_PARA_INICIAR_COLETA_NOME_PROFISSIONAL": "iniciar_coleta_nome_profissional",
        END: END 
    }
)

workflow.add_conditional_edges(
    "preparar_consulta_api_profissionais",
    route_after_preparar_consulta_api,
    {
        "consultar_api_profissionais": "consultar_api_profissionais",
        END: END
    }
)

workflow.add_conditional_edges(
    "consultar_api_profissionais",
    route_after_consultar_api_profissionais,
    {
        "processar_escolha_profissional": "processar_escolha_profissional", # Mapeia para o futuro nó
        "lidar_com_nenhum_profissional_encontrado": "lidar_com_nenhum_profissional_encontrado", # Mapeia para o futuro nó
        END: END
    }
)

# Após processar a especialidade
workflow.add_edge("processar_input_especialidade", END)


# Adicionando os nós placeholder
workflow.add_edge("iniciar_coleta_nome_profissional", END)
workflow.add_edge("processar_input_nome_profissional_indicado", END)

workflow.add_edge("processar_confirmacao_nome_profissional", END)
# Nós que geram respostas finais
workflow.add_edge("gerar_resposta_consulta_unidade", END)
workflow.add_edge("iniciar_coleta_especialidade", END)
workflow.add_edge("gerar_resposta_saudacao_despedida", END)
workflow.add_edge("gerar_resposta_fora_escopo", END)
workflow.add_edge("processar_escolha_profissional", END)
workflow.add_edge("executar_consulta_profissionais_e_apresentar", END) # Por enquanto
workflow.add_edge("placeholder_processar_escolha_profissional_recomendado", END)
workflow.add_edge("processar_input_turno", END) # NOVA ARESTA
workflow.add_edge("consultar_e_apresentar_datas_disponiveis", END)
workflow.add_edge("processar_escolha_data_e_buscar_horarios", END)
workflow.add_edge("processar_escolha_horario", END)
workflow.add_edge("processar_confirmacao_final", END)

workflow.add_node("placeholder_processar_escolha_profissional_recomendado", placeholder_processar_escolha_profissional_recomendado)

# %%
# compilar o grafo
app = workflow.compile(checkpointer=memory)

# %%
# Visualização do Grafo (opcional, mas útil)
try:
    graph_representation = app.get_graph(xray=True)
    mermaid_syntax = graph_representation.draw_mermaid()
    print("--- Sintaxe Mermaid Gerada ---")
    print(mermaid_syntax)
    print("-----------------------------")
    display(Image(graph_representation.draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER))) # Comente se causar problemas
except Exception as e:
    print(f"Erro ao gerar visualização do grafo: {e}")

# %%
def chat_gradio(user_input: str, history_list_de_listas: list) -> str:
    """
    Função de chat para o Gradio.
    """
    thread_id = "healthai_thread_1" # Usar um ID consistente para manter o histórico da conversa
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Verifica o estado atual na memória ANTES da nova invocação
        current_state_before_stream = memory.get(config)
        if current_state_before_stream:
            # Acessa o estado 'values' dentro do checkpoint, se aplicável (depende da versão/implementação do checkpointer)
            state_values = current_state_before_stream.get('values', current_state_before_stream) if isinstance(current_state_before_stream, dict) else {}
            phase_before = state_values.get("fase_agendamento")
            logger.debug(f"DEBUG [chat_gradio] State BEFORE app.stream: fase_agendamento='{phase_before}'")
            # Você pode adicionar logs para outras chaves se necessário:
            # logger.debug(f"DEBUG [chat_gradio] State BEFORE app.stream: datas_disponiveis_apresentadas='{state_values.get('datas_disponiveis_apresentadas')}'")
        else:
            logger.debug("DEBUG [chat_gradio] State BEFORE app.stream: No previous state found in memory.")
    except Exception as e:
        logger.error(f"DEBUG [chat_gradio] Error getting state before stream: {e}", exc_info=True) # Log com traceback
    # --- FIM DA ADIÇÃO ---


    inputs = {"messages": [HumanMessage(content=user_input)]}

    try:
        # Invoca o grafo LangGraph
        # O stream pode ser mais informativo para debug
        final_state_after_invocation = None
        for event_chunk in app.stream(inputs, config=config, stream_mode="values"):
            final_state_after_invocation = event_chunk # O último chunk em 'values' mode é o estado final
            # Você pode inspecionar event_chunk aqui para ver os estados intermediários se precisar

        # Log do estado final após a invocação
        if final_state_after_invocation:
             logger.debug(f"DEBUG [chat_gradio] State AFTER app.stream: {final_state_after_invocation}")
        else:
             logger.debug("DEBUG [chat_gradio] State AFTER app.stream: final_state_after_invocation is None")

        if final_state_after_invocation and "messages" in final_state_after_invocation and final_state_after_invocation["messages"]:
            latest_message_obj = final_state_after_invocation["messages"][-1]

            content_to_return = ""
            # Extrai conteúdo da última mensagem, que deve ser do AI
            if hasattr(latest_message_obj, 'content') and isinstance(latest_message_obj, AIMessage): # Verifica tipo AIMessage
                content_to_return = latest_message_obj.content
            # Adicione outras verificações se você usar formatos diferentes como ('ai', 'content')
            elif isinstance(latest_message_obj, tuple) and len(latest_message_obj) == 2 and latest_message_obj[0] == 'ai':
                 content_to_return = latest_message_obj[1]
            elif isinstance(latest_message_obj, dict) and latest_message_obj.get("role") == "ai":
                 content_to_return = latest_message_obj.get("content", "Erro: AI não retornou conteúdo.")
            else:
                 # Se a última mensagem não for do AI (pode acontecer se o grafo terminar inesperadamente)
                logger.warning(f"AVISO [chat_gradio]: Última mensagem no estado não é do AI ou está em formato inesperado: {latest_message_obj}")
                # Tenta pegar a penúltima mensagem (assumindo que a última foi o HumanMessage adicionado)
                if len(final_state_after_invocation["messages"]) > 1:
                    penultimate_message = final_state_after_invocation["messages"][-2]
                    if hasattr(penultimate_message, 'content') and isinstance(penultimate_message, AIMessage):
                         content_to_return = penultimate_message.content
                    elif isinstance(penultimate_message, tuple) and len(penultimate_message) == 2 and penultimate_message[0] == 'ai':
                         content_to_return = penultimate_message[1]
                    # ... outras verificações ...

                if not content_to_return:
                    content_to_return = "Não consegui processar uma resposta no momento. Tente novamente."


            return content_to_return
        else:
             logger.error("ERRO [chat_gradio]: Estado final inválido ou sem mensagens após invocação.") # Log de erro
             return "Desculpe, ocorreu um erro e não recebi uma resposta completa do sistema."
            
    except Exception as e:
        logger.error(f"Erro CRÍTICO ao invocar o app LangGraph: {e}", exc_info=True) # Log com traceback
        # import traceback # Import já feito no logger
        # traceback.print_exc() # Logger já faz isso com exc_info=True
        return f"Ocorreu um erro crítico no sistema ao processar sua solicitação."

interface = gr.ChatInterface(
    fn=chat_gradio,
    title="HealthAI Assistente (PoC com LangGraph)",
    description="Clínica HealthAI - Assistente Virtual para informações e agendamentos (em desenvolvimento).",
    chatbot=gr.Chatbot(
        height=600,
        show_label=False, # Opcional: para não mostrar "Chatbot" acima da caixa de chat
        # type="messages" # Removido daqui, pois ChatInterface lida com o formato da mensagem
    ),
    textbox=gr.Textbox(placeholder="Digite sua mensagem aqui...", container=False, scale=7),
    # Os botões retry_btn, undo_btn, clear_btn foram removidos, pois são gerenciados de forma diferente
    # ou habilitados por padrão em versões mais recentes.
)

interface.launch()


