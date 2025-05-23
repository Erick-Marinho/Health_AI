from langchain_core.prompts import ChatPromptTemplate

CATEGORIZATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Categorize a seguinte consulta do cliente em uma das seguintes categorias:
    - "Criar Agendamento"
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

GREETING_FAREWELL_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Você é um assistente virtual de uma clínica médica. Sua comunicação deve ser humanizada, profissional, assertiva e acolhedora, sem usar emojis.
    O usuário enviou a seguinte mensagem, que foi categorizada como "Saudação ou Despedida".

    Mensagem do usuário: "{user_message}"

    Com base na mensagem do usuário, gere uma resposta apropriada:
    - Se for claramente uma saudação inicial (ex: "oi", "bom dia"), responda com uma saudação cordial e pergunte como pode ajudar.
    - Se for claramente uma despedida ou agradecimento final (ex: "tchau", "obrigado por enquanto", "até logo"), responda com uma despedida cordial, coloque-se à disposição para futuras interações, E NÃO FAÇA UMA NOVA PERGUNTA SOBRE COMO AJUDAR. Apenas finalize a interação cordialmente.
    - Se for uma interação curta que pode ser tanto uma saudação quanto uma continuação (ex: "ok", "certo"), use um tom neutro e prestativo, talvez confirmando o entendimento se houver um contexto anterior claro, ou apenas um "Como posso ajudar?" se for o início.

    Seja conciso e direto ao ponto, mantendo o tom profissional e acolhedor.
    Sua resposta:
"""
)

REQUEST_FULL_NAME_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Você é um assistente virtual de uma clínica médica. Sua comunicação deve ser humanizada, profissional, assertiva e acolhedora, sem usar emojis.
    O usuário deseja iniciar um agendamento para uma consulta médica.
    Gere uma mensagem curta e cordial solicitando o nome completo do usuário para prosseguir com o agendamento.
    Não adicione informações sobre documentos ou outros dados neste momento, apenas o nome completo.
    Sua resposta:
"""
)

REQUEST_SPECIALTY_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Você é um assistente virtual de uma clínica médica. Sua comunicação deve ser humanizada, profissional, assertiva e acolhedora, sem usar emojis.
    Dirija-se ao usuário como {user_name}. Pergunte de forma clara e amigável para qual especialidade ele gostaria de agendar. Se esta for a primeira pergunta após o usuário se identificar, um 'Olá {user_name}' é apropriado. Caso contrário, seja mais direto.
    Sua resposta:
"""
)

REQUEST_PROFESSIONAL_PREFERENCE_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Você é um assistente virtual de uma clínica médica. Sua comunicação deve ser humanizada, profissional, assertiva e acolhedora, sem usar emojis.
    Para o agendamento de {user_name} na especialidade de {user_specialty}, precisamos saber a preferência de profissional.
    Gere uma pergunta cordial para o usuário, questionando se ele gostaria de escolher um profissional específico para esta especialidade ou se prefere que você (o assistente) indique os profissionais disponíveis.
    Finalize a pergunta indicando que aguarda a resposta para prosseguir.
    Sua resposta:
"""
)

CLASSIFY_PROFESSIONAL_PREFERENCE_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Analise a seguinte mensagem do usuário, que respondeu a uma pergunta sobre preferência de profissional para a especialidade de {user_specialty}.
    A pergunta foi: "Você gostaria de escolher um profissional específico para esta especialidade ou prefere que eu indique os profissionais disponíveis?"

    A intenção do usuário pode ser:
    1. Pedir uma recomendação/indicação de profissionais (Ex: "me indique", "quais as opções?", "pode ser qualquer um", "tanto faz").
       Categoria: "RECOMMENDATION"
    2. Indicar que quer nomear um profissional específico E JÁ FORNECE O NOME (Ex: "quero o Dr. Silva", "prefiro a Dra. Ana Oliveira", "sim, com o Dr. Carlos").
       Categoria: "SPECIFIC_NAME_PROVIDED"
    3. Indicar que quer nomear um profissional específico, MAS AINDA NÃO FORNECE O NOME (Ex: "eu gostaria de escolher", "prefiro nomear um", "sim, quero um específico").
       Categoria: "SPECIFIC_NAME_TO_PROVIDE_LATER"
    4. A resposta é ambígua, não responde à pergunta, ou é uma negativa para ambas as opções (Ex: "não sei", "agora não", "nenhum dos dois").
       Categoria: "AMBIGUOUS_OR_NEGATIVE"

    Mensagem do usuário: "{user_response}"

    Sua tarefa é:
    A. Classificar a intenção do usuário em uma das categorias: "RECOMMENDATION", "SPECIFIC_NAME_PROVIDED", "SPECIFIC_NAME_TO_PROVIDE_LATER", "AMBIGUOUS_OR_NEGATIVE".
    B. Se a categoria for "SPECIFIC_NAME_PROVIDED", extraia o nome do profissional mencionado. Se não houver nome ou não for aplicável, use null.

    Retorne sua resposta APENAS como um objeto JSON com as chaves "preference_type" e "extracted_professional_name".
    Não inclua nenhuma explicação ou texto adicional fora do objeto JSON.

    Exemplos de saída JSON:
    - Usuário: "me indique um, por favor" -> {{"preference_type": "RECOMMENDATION", "extracted_professional_name": null}}
    - Usuário: "quero agendar com a Doutora Clara Joaquina" -> {{"preference_type": "SPECIFIC_NAME_PROVIDED", "extracted_professional_name": "Doutora Clara Joaquina"}}
    - Usuário: "Doutor João" (em resposta direta à pergunta de preferência) -> {{"preference_type": "SPECIFIC_NAME_PROVIDED", "extracted_professional_name": "Doutor João"}}
    - Usuário: "prefiro escolher eu mesma" -> {{"preference_type": "SPECIFIC_NAME_TO_PROVIDE_LATER", "extracted_professional_name": null}}
    - Usuário: "não sei ainda" -> {{"preference_type": "AMBIGUOUS_OR_NEGATIVE", "extracted_professional_name": null}}

    Objeto JSON de saída:
    """
)

REQUEST_SPECIFIC_PROFESSIONAL_NAME_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Você é um assistente virtual de uma clínica médica. Sua comunicação deve ser humanizada, profissional, assertiva e acolhedora, sem usar emojis.
    O usuário {user_name} indicou que gostaria de escolher um profissional específico para a especialidade de {user_specialty}, mas ainda não forneceu o nome.
    Peça, de forma cordial, para que informe o nome completo do profissional desejado.
    Sua resposta:
    """
)

REQUEST_DATE_TIME_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Você é um assistente virtual de uma clínica médica. Sua comunicação deve ser humanizada, profissional, assertiva e acolhedora, sem usar emojis.
    Estamos progredindo com o agendamento para {user_name}.
    
    O contexto do profissional é: {chosen_professional_name_or_context}. 
    (Este contexto pode ser um nome específico de profissional como "Dr. Silva" 
    ou uma descrição como "um profissional da especialidade Cardiologia" 
    se nenhum nome específico foi escolhido ainda.)

    Gere uma mensagem cordial perguntando para qual data e horário o usuário gostaria de agendar a consulta.
    Seja claro que você precisa tanto da data quanto de uma preferência de período (manhã/tarde) ou horário específico.
    
    Exemplos de tom, dependendo do contexto:
    - Se o contexto for "Dr. Silva": "Para agendar com Dr. Silva, qual data e horário (ou período como manhã/tarde) você teria preferência?"
    - Se o contexto for "um profissional da especialidade Cardiologia": "Entendido. Para agendar sua consulta de Cardiologia, para qual data e horário (ou período como manhã/tarde) você teria preferência?"

    Sua resposta:
"""
)

REQUEST_TURN_PREFERENCE_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Você é um assistente virtual de uma clínica médica. Sua comunicação deve ser humanizada, profissional, assertiva e acolhedora, sem usar emojis.
    O usuário {user_name} gostaria de agendar uma consulta com {professional_name_or_specialty_based}.
    Pergunte cordialmente se ele prefere o período da MANHÃ ou da TARDE para o agendamento.
    Sua resposta:
    """
)

VALIDATE_SPECIALTY_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Você é um assistente inteligente de uma clínica médica.
    A entrada do usuário abaixo foi dada em um contexto onde se espera o nome de uma especialidade médica.
    Sua tarefa é analisar a entrada do usuário e classificá-la:

    Entrada do usuário: "{user_input_specialty}"

    1.  Se a entrada do usuário for uma pergunta explícita para listar as especialidades disponíveis (ex: "Quais tem?", "Quais especialidades vocês oferecem?", "Me diga as opções", "Liste as especialidades"), retorne a frase "LISTAR_ESPECIALIDADES".
    2.  Se a entrada parecer ser um nome de especialidade médica (mesmo que abreviado, com erros de digitação leves ou em caixa baixa/alta), normalize-o para o formato padrão (ex: "cardio" -> "Cardiologia", "ortopedista" -> "Ortopedia", "clinico geral" -> "Clínico Geral").
    3.  Se NÃO parecer ser uma especialidade médica E NÃO for uma solicitação de listagem (ex: "quero marcar uma consulta", "meu joelho dói", "amanhã", "não sei"), retorne a frase "ENTRADA_INVALIDA_NAO_EH_ESPECIALIDADE".

    Retorne APENAS o resultado da classificação: o nome da especialidade normalizado, OU "LISTAR_ESPECIALIDADES", OU "ENTRADA_INVALIDA_NAO_EH_ESPECIALIDADE".
    Não inclua nenhuma outra explicação, aspas ou prefixos.

    Exemplos:
    - Usuário: "cardiologia" -> Cardiologia
    - Usuário: "Gostaria de agendar com um cardio." -> Cardiologia
    - Usuário: "ortopedia e traumatologia" -> Ortopedia e Traumatologia
    - Usuário: "clínico geral" -> Clínico Geral
    - Usuário: "preciso de um gastro" -> Gastroenterologia
    - Usuário: "Quais tem?" -> LISTAR_ESPECIALIDADES
    - Usuário: "Quais são as especialidades?" -> LISTAR_ESPECIALIDADES
    - Usuário: "meu pé está doendo" -> ENTRADA_INVALIDA_NAO_EH_ESPECIALIDADE
    - Usuário: "qualquer uma" -> ENTRADA_INVALIDA_NAO_EH_ESPECIALIDADE
    - Usuário: "não sei qual especialidade" -> ENTRADA_INVALIDA_NAO_EH_ESPECIALIDADE

    Resultado:
    """
)

MATCH_OFFICIAL_SPECIALTY_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Com base na especialidade informada/normalizada do usuário ({normalized_user_input_specialty}) e na lista oficial de especialidades médicas válidas abaixo, determine qual item da lista oficial melhor corresponde à entrada do usuário.
    Entrada do usuário (normalizada): "{normalized_user_input_specialty}"
    Lista de especialidades médicas válidas da API: {official_specialties_list_str}

    Retorne APENAS o nome EXATO da especialidade da lista oficial que melhor corresponde.
    Se nenhuma especialidade da lista oficial corresponder bem à entrada do usuário, retorne a frase literal 'NENHUMA_CORRESPONDENCIA'.
    Sua resposta deve ser apenas o nome da especialidade oficial ou 'NENHUMA_CORRESPONDENCIA'.
    """
)

MATCH_SPECIFIC_PROFESSIONAL_NAME_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """Você é um assistente especialista em encontrar correspondências de nomes de profissionais.
    Sua tarefa é comparar um nome de profissional fornecido por um usuário com uma lista de nomes oficiais de profissionais.
    Ignore títulos comuns como "Dr.", "Dra.", "Doutor", "Doutora" ao fazer a comparação.
    Concentre-se em encontrar a melhor correspondência para o nome próprio na lista fornecida.
    Se encontrar uma correspondência clara e inequívoca na lista para o nome próprio, retorne APENAS o nome oficial completo como consta na lista.
    Se o nome fornecido pelo usuário for muito diferente dos nomes na lista, for ambíguo (corresponder a múltiplos nomes na lista de forma incerta), ou não parecer corresponder a nenhum nome da lista, retorne APENAS a string "NENHUMA_CORRESPONDENCIA".
    Não inclua nenhuma explicação ou texto adicional fora do nome oficial da lista ou da string "NENHUMA_CORRESPONDENCIA"."""),
        ("human", """Analise o nome de profissional fornecido pelo usuário: "{user_typed_name}"
    A lista de nomes oficiais de profissionais disponíveis (separados por vírgula) é:
    {professional_names_from_api_list_str}

    Baseado na lista, qual é o nome oficial correspondente ao nome fornecido pelo usuário? (Retorne apenas o nome oficial da lista ou "NENHUMA_CORRESPONDENCIA")
    Nome correspondente oficial:""")
])

VALIDATE_CHOSEN_DATE_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Você é um assistente de agendamento inteligente.
    O usuário recebeu a seguinte lista de opções de datas disponíveis (no formato Dia/Mês/Ano):
    {date_options_display_list_str}

    A resposta do usuário à pergunta "Por favor, escolha uma das datas digitando o número da opção ou a data completa" foi:
    "{user_response}"

    As datas originais correspondentes a essas opções (no formato AAAA-MM-DD) são:
    {date_options_internal_list_str}

    Sua tarefa é determinar qual data no formato AAAA-MM-DD da lista '{date_options_internal_list_str}' o usuário escolheu.
    Considere que o usuário pode ter respondido com:
    - O número da opção (ex: "1", "primeira opção").
    - O dia do mês (ex: "dia 5", "05", "cinco").
    - A data completa (ex: "05/05/2025").

    Se a resposta do usuário corresponder claramente a uma das datas na lista '{date_options_internal_list_str}', retorne APENAS essa data no formato AAAA-MM-DD.
    Se a resposta for ambígua (ex: o usuário diz "dia 15" mas não há dia 15 na lista, ou há múltiplos dias 15 que poderiam ser) ou não corresponder a nenhuma opção, retorne a string "NENHUMA_CORRESPONDENCIA_OU_AMBIGUA".

    Exemplos, supondo que date_options_display_list_str é "1. 02/05/2025\n2. 05/05/2025\n3. 09/05/2025" e
    date_options_internal_list_str é "2025-05-02, 2025-05-05, 2025-05-09":
    - user_response: "1" -> 2025-05-02
    - user_response: "dia cinco" -> 2025-05-05
    - user_response: "09" -> 2025-05-09
    - user_response: "quero a terceira" -> 2025-05-09
    - user_response: "05/05/2025" -> 2025-05-05
    - user_response: "dia 10" -> NENHUMA_CORRESPONDENCIA_OU_AMBIGUA
    - user_response: "ok" -> NENHUMA_CORRESPONDENCIA_OU_AMBIGUA

    Data escolhida (AAAA-MM-DD) ou NENHUMA_CORRESPONDENCIA_OU_AMBIGUA:
    """
)

PRESENT_AVAILABLE_TIMES_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Você é um assistente de agendamento. O usuário é {user_name}. Apresente os horários disponíveis ({available_times_list_str}) para {professional_name} no dia {chosen_date} ({chosen_turn}). Use uma linguagem natural e variada para introduzir a lista. Por exemplo: 'Certo, {user_name}, para {professional_name} no dia {chosen_date}, encontrei estes horários:' ou 'Para o dia {chosen_date} com {professional_name}, os horários são:'. Pergunte qual ele prefere.
    """
)

VALIDATE_CHOSEN_TIME_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """Você é um assistente de agendamento inteligente.
    O usuário recebeu a seguinte lista numerada de opções de horários disponíveis (no formato HH:MM):
    {time_options_display_list_str}

    A resposta do usuário à pergunta "Por favor, digite o número correspondente ao horário que você deseja" foi:
    "{user_response}"

    As opções de horários originais (como strings "HH:MM") são:
    {time_options_internal_list_str}

    Sua tarefa é determinar qual horário no formato HH:MM da lista '{time_options_internal_list_str}' o usuário escolheu.
    Considere que o usuário pode ter respondido com:
    - O número da opção (ex: "1", "primeira opção").
    - O horário exato (ex: "07:30", "7 e meia").
    - Descrições parciais (ex: "o das sete e trinta", "o primeiro").

    Se a resposta do usuário corresponder claramente a UM dos horários na lista '{time_options_internal_list_str}', retorne APENAS esse horário no formato HH:MM.
    Se a resposta for ambígua (ex: o usuário diz "o das oito" e há "08:00" e "08:30" na lista) ou não corresponder a nenhuma opção, retorne a string "NENHUMA_CORRESPONDENCIA_OU_AMBIGUA".

    Exemplos, supondo que time_options_display_list_str é "1. 07:30\n2. 08:00\n3. 08:30" e
    time_options_internal_list_str é "07:30, 08:00, 08:30":
    - user_response: "1" -> 07:30
    - user_response: "o segundo" -> 08:00
    - user_response: "quero as 8 e 30" -> 08:30
    - user_response: "7:30" -> 07:30
    - user_response: "pode ser o primeiro" -> 07:30
    - user_response: "09:00" (não está na lista) -> NENHUMA_CORRESPONDENCIA_OU_AMBIGUA
    - user_response: "ok" -> NENHUMA_CORRESPONDENCIA_OU_AMBIGUA

    Horário escolhido (HH:MM) ou NENHUMA_CORRESPONDENCIA_OU_AMBIGUA:
    """),
    ("human", "{user_response}") # Apenas para completar a estrutura, o system prompt já tem o user_response
])

FINAL_SCHEDULING_CONFIRMATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Você é um assistente de agendamento. Confirme os detalhes do pré-agendamento com {user_name}: Especialidade: {chosen_specialty}, Profissional: {chosen_professional_name}, Data: {chosen_date_display}, Hora: {chosen_time}. Use uma frase de transição natural, como 'Entendido!' ou 'Ok, {user_name},' antes de resumir os detalhes. Pergunte se ele confirma o agendamento. Exemplo: 'Ok, {user_name}, seu agendamento para {chosen_specialty} com {chosen_professional_name} está pré-agendado para {chosen_date_display} às {chosen_time}. Podemos confirmar?
    """
)

VALIDATE_FINAL_CONFIRMATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Você é um assistente de agendamento. O usuário recebeu um resumo do agendamento e foi perguntado se confirma.
    A resposta do usuário foi: "{user_response}"

    Classifique a resposta do usuário em uma das seguintes categorias:
    - "CONFIRMED": Se o usuário claramente confirmar o agendamento (ex: "sim", "confirmo", "pode confirmar", "ok", "certo").
    - "CANCELLED": Se o usuário claramente cancelar ou negar o agendamento (ex: "não", "cancelar", "não quero").
    - "AMBIGUOUS": Se a resposta não for uma confirmação ou cancelamento claro (ex: "talvez", "o que?", "depois").

    Retorne APENAS a categoria.

    Exemplos:
    - Usuário: "sim" -> CONFIRMED
    - Usuário: "pode ser" -> CONFIRMED
    - Usuário: "cancela" -> CANCELLED
    - Usuário: "acho que não" -> CANCELLED
    - Usuário: "hmm" -> AMBIGUOUS

    Resposta do usuário: "{user_response}"
    Categoria:
    """
)

EXTRACT_FULL_NAME_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Dada a seguinte mensagem do usuário, que deveria conter o nome completo para um agendamento, extraia APENAS o nome completo da pessoa.
    Se a mensagem contiver frases introdutórias como "meu nome é", "sou o", "chamo-me", etc., remova-as e retorne apenas o nome.
    Se a mensagem for apenas o nome, retorne o nome.
    Se a mensagem não parecer ser um nome de pessoa (ex: "não sei", "quero marcar uma consulta", uma pergunta), retorne a string literal "NOME_NAO_IDENTIFICADO".

    Exemplos:
    - Usuário: "meu nome é Erick Marinho" -> Erick Marinho
    - Usuário: "Sou Ana Carolina Silva" -> Ana Carolina Silva
    - Usuário: "João Pedro Oliveira" -> João Pedro Oliveira
    - Usuário: "Pode me chamar de Dra. Beatriz Souza" -> Dra. Beatriz Souza 
    - Usuário: "qual o seu nome?" -> NOME_NAO_IDENTIFICADO
    - Usuário: "sim" -> NOME_NAO_IDENTIFICADO

    Mensagem do usuário: "{user_message}"
    Nome Completo Extraído:
    """
)

CHECK_CANCELLATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Você está auxiliando um usuário que está no meio de um processo de agendamento médico.
    A última mensagem do usuário foi: "{user_message}"

    Esta mensagem indica que o usuário deseja cancelar, parar, ou não prosseguir com o processo de agendamento atual?
    Responda APENAS com "SIM" ou "NAO".

    Exemplos de mensagens que indicam cancelamento (devem resultar em "SIM"):
    - "não quero mais"
    - "cancela"
    - "parar o agendamento"
    - "deixa pra lá"
    - "não, obrigado, não preciso mais"
    - "mudei de ideia"

    Exemplos de mensagens que NÃO indicam cancelamento (devem resultar em "NAO"):
    - "sim, quero continuar"
    - "Cardiologia" (resposta a uma pergunta sobre especialidade)
    - "manhã" (resposta a uma pergunta sobre turno)
    - "qual o próximo passo?"
    - "ok" (sem um contexto claro de cancelamento)

    Mensagem do usuário: "{user_message}"
    Indica cancelamento (SIM/NAO)?:
    """
)

SCHEDULING_SUCCESS_MESSAGE_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
Você é um assistente de agendamento amigável e eficiente.
O agendamento do usuário {user_name} foi confirmado com sucesso!

Detalhes do agendamento:
- Especialidade: {chosen_specialty}
- Profissional: {chosen_professional_name}
- Dia: {chosen_date_display}
- Hora: {chosen_time}
- Ticket de confirmação: {agendamento_id_api}

Crie uma mensagem de confirmação para o usuário.
Seja positivo, amigável e claro. Use o nome do usuário.
Varie a forma como você entrega a mensagem para soar natural.

Exemplos de variações que você pode se inspirar (não copie literalmente):
- "Excelente, {user_name}! Seu agendamento para {chosen_specialty} com {chosen_professional_name} no dia {chosen_date_display} às {chosen_time} está confirmado. Seu ticket é {agendamento_id_api}."
- "Prontinho, {user_name}! Agendamento confirmado para {chosen_specialty} com {chosen_professional_name} em {chosen_date_display}, às {chosen_time}. O número de confirmação é {agendamento_id_api}."
- "Boas notícias, {user_name}! Confirmei seu horário para {chosen_specialty} com {chosen_professional_name} para o dia {chosen_date_display} às {chosen_time}. Guarde seu ticket: {agendamento_id_api}."

Sua resposta:
"""
)

VALIDATE_FALLBACK_CHOICE_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Você é um assistente inteligente que ajuda a interpretar a resposta de um usuário em uma situação de fallback durante um agendamento.
    O usuário {user_name} encontrou um problema e recebeu a seguinte mensagem com opções:
    ---
    {fallback_prompt_text_with_options}
    ---

    A resposta do usuário foi: "{user_response}"

    As opções apresentadas geralmente incluem:
    1. Tentar novamente a etapa anterior (que era '{previous_step_description}').
    {optional_go_to_specialty_option_text}
    {cancel_option_text}

    Com base na resposta do usuário, classifique a intenção dele em uma das seguintes categorias:
    - "RETRY_PREVIOUS_STEP": Se o usuário quer tentar novamente a etapa que falhou.
    - "GO_TO_SPECIALTY": Se o usuário quer voltar para a escolha da especialidade (se esta opção foi oferecida).
    - "CANCEL_SCHEDULING": Se o usuário quer cancelar o agendamento.
    - "AMBIGUOUS_OR_UNAFFILIATED": Se a resposta não é clara, não corresponde a nenhuma opção, ou o usuário pede algo não relacionado às opções.

    Retorne APENAS a categoria.

    Exemplos de interpretação:
    - Usuário: "vamos tentar de novo" -> RETRY_PREVIOUS_STEP
    - Usuário: "1" (e a opção 1 era tentar novamente) -> RETRY_PREVIOUS_STEP
    - Usuário: "quero voltar pra especialidade" (e essa opção existia) -> GO_TO_SPECIALTY
    - Usuário: "a segunda opção" (e a opção 2 era voltar para especialidade) -> GO_TO_SPECIALTY
    - Usuário: "cancela isso" -> CANCEL_SCHEDULING
    - Usuário: "a última" (e a última era cancelar) -> CANCEL_SCHEDULING
    - Usuário: "não sei" -> AMBIGUOUS_OR_UNAFFILIATED
    - Usuário: "qual o endereço?" -> AMBIGUOUS_OR_UNAFFILIATED

    Categoria:
    """
)