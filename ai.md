health_ai/
├── app/                     # Código fonte da aplicação
│   ├── __init__.py
│   ├── main.py              # Ponto de entrada da aplicação FastAPI. Inicializa a instância `app = FastAPI()`, inclui routers das camadas de apresentação (ex: `health_router.py`, `webhook_router.py`) e pode configurar middlewares globais. Executado por Uvicorn conforme definido no `Dockerfile` e `docker-compose.yml`.
│   │
│   ├── core/                # Configurações centrais, logging, utilitários transversais
│   │   ├── __init__.py
│   │   ├── config.py        # Gestão de configurações utilizando Pydantic Settings para carregar e validar variáveis de ambiente definidas, por exemplo, no `.env` e injetadas via `docker-compose.yml` (como `DATABASE_URL`, chaves de API para serviços externos).
│   │   └── logging_config.py# Configuração de logging
│   │
│   ├── domain/              # Camada de Domínio (coração da lógica de negócios)
│   │   ├── __init__.py
│   │   ├── entities/        # Entidades de domínio (ex: DocumentoRAG, ChunkRAG, Agendamento)
│   │   │   ├── __init__.py
│   │   │   ├── base_entity.py # (Opcional) Classe base para entidades
│   │   │   ├── rag_entities.py
│   │   │   └── appointment_entities.py
│   │   ├── value_objects/   # Value Objects (ex: Embedding, SlotDeTempo, MetadadosDocumento)
│   │   │   ├── __init__.py
│   │   │   └── ...
│   │   ├── services/        # Serviços de Domínio (lógica de negócio que não se encaixa em uma entidade)
│   │   │   ├── __init__.py
│   │   │   └── ... # Ex: ValidacaoDisponibilidadeServico
│   │   └── repositories/    # Interfaces dos Repositórios (contratos para persistência)
│   │       ├── __init__.py
│   │       ├── i_rag_repository.py         # Interface para repositório RAG
│   │       └── i_appointment_repository.py # Interface para repositório de agendamentos
│   │
│   ├── application/         # Camada de Aplicação (casos de uso, orquestração)
│   │   ├── __init__.py
│   │   ├── services/        # Serviços de Aplicação que orquestram os casos de uso
│   │   │   ├── __init__.py
│   │   │   ├── conversational_agent_service.py # Orquestra o agente LangGraph e interações com usuário
│   │   │   ├── rag_app_service.py          # Lógica de aplicação para RAG (ingestão, consulta)
│   │   │   └── appointment_app_service.py  # Lógica de aplicação para agendamentos
│   │   ├── dtos/            # Data Transfer Objects (se necessário para comunicação entre camadas ou com o exterior)
│   │   │   ├── __init__.py
│   │   │   └── ...
│   │   └── event_handlers/  # (Opcional) Handlers para eventos de domínio, se utilizar Domain Events
│   │       ├── __init__.py
│   │       └── ...
│   │
│   ├── infrastructure/      # Camada de Infraestrutura (implementações concretas de interfaces, acesso a BD, APIs externas)
│   │   ├── __init__.py
│   │   ├── database/        # Configuração e acesso ao banco de dados
│   │   │   ├── __init__.py
│   │   │   ├── connection.py    # Configuração da conexão (ex: SQLAlchemy engine para PostgreSQL com pgvector)
│   │   │   ├── db_models.py     # Modelos ORM (ex: SQLAlchemy models, se diferentes das entidades de domínio)
│   │   │   └── repositories/    # Implementações concretas dos Repositórios
│   │   │       ├── __init__.py
│   │   │       ├── postgres_rag_repository.py        # Implementação para pgvector
│   │   │       └── postgres_appointment_repository.py # Implementação para agendamentos
│   │   ├── external_apis/   # Clientes para APIs externas
│   │   │   ├── __init__.py
│   │   │   ├── clinic_api_client.py # Cliente para a API da clínica (agendamentos)
│   │   │   └── whatsapp_client.py   # Cliente para a API do provedor WhatsApp
│   │   ├── embeddings/      # Serviços de geração de embeddings
│   │   │   ├── __init__.py
│   │   │   └── embedding_generator_service.py # Ex: usando sentence-transformers
│   │   └── messaging/       # (Opcional) Se usar sistemas de filas de mensagens (ex: RabbitMQ, Kafka)
│   │       ├── __init__.py
│   │       └── ...
│   │
│   ├── presentation/        # Camada de Apresentação (API FastAPI, Webhooks) - também chamada de 'interfaces' ou 'api'
│   │   ├── __init__.py
│   │   ├── routers/         # Endpoints FastAPI organizados por recurso
│   │   │   ├── __init__.py
│   │   │   ├── dependencies.py # Dependências reutilizáveis do FastAPI (ex: obter usuário atual)
│   │   │   ├── webhook_router.py   # Endpoint para webhooks do WhatsApp
│   │   │   ├── rag_router.py       # (Opcional) Endpoints para gerenciar ou consultar o RAG diretamente
│   │   │   ├── appointment_router.py # (Opcional) Endpoints para gerenciar agendamentos diretamente
│   │   │   └── health_router.py    # Endpoint de "health check" (ex: implementado em app/main.py ou movido para cá)
│   │   └── schemas/         # Schemas Pydantic para validação de entrada/saída da API (Request/Response models)
│   │       ├── __init__.py
│   │       ├── webhook_schemas.py
│   │       ├── rag_schemas.py
│   │       └── appointment_schemas.py
│   │
│   ├── agents/              # Lógica e configuração dos agentes conversacionais (LangGraph, PydanticAI)
│   │   ├── __init__.py
│   │   ├── langgraph_config/ # Configuração específica do agente LangGraph
│   │   │   ├── __init__.py
│   │   │   ├── graph_state.py  # Modelo Pydantic para o estado do grafo conversacional
│   │   │   ├── nodes.py        # Funções que representam os nós do grafo (etapas da conversa)
│   │   │   ├── edges.py        # Lógica condicional para transição entre nós (arestas)
│   │   │   └── agent_builder.py# Constrói e compila o agente LangGraph
│   │   └── pydantic_ai_tools/ # Ferramentas ou sub-agentes construídos com PydanticAI
│   │       ├── __init__.py
│   │       └── information_extractor_agent.py # Ex: Agente PydanticAI para extrair dados estruturados de mensagens
│   │
│   └── utils/               # Funções utilitárias específicas da aplicação
│       ├── __init__.py
│       └── text_processing.py # Ex: para chunking semântico, normalização de texto
│
├── scripts/                 # Scripts de apoio (ex: ingestão de dados, setup inicial, migrations manuais)
│   ├── __init__.py
│   ├── ingest_rag_data.py   # Script para popular a base de dados do RAG com documentos
│   └── setup_pgvector.py    # Script para configurar a extensão pgvector no PostgreSQL (se não for via ORM/migrations)
│
├── tests/                   # Testes automatizados
│   ├── __init__.py
│   ├── fixtures/            # Fixtures reutilizáveis para os testes
│   ├── unit/                # Testes unitários (isolados por camada/módulo)
│   │   ├── __init__.py
│   │   ├── domain/
│   │   ├── application/
│   │   └── ...
│   └── integration/         # Testes de integração (interação entre componentes/camadas)
│       ├── __init__.py
│       └── ...
│
├── .env                     # Arquivo local (ignorado pelo Git via .gitignore) para definir variáveis de ambiente durante o desenvolvimento. Lido pelo docker-compose.yml para injetar configurações nos contêineres.
├── .env.example             # Arquivo de exemplo para variáveis de ambiente, para ser versionado.
├── .gitignore               # Arquivos e diretórios a serem ignorados pelo Git (ex: .env, __pycache__, .venv/).
├── .dockerignore            # Arquivos e diretórios a serem ignorados pelo Docker durante o build da imagem (ex: .git, .venv/, __pycache__/).
├── Dockerfile               # Define a imagem Docker para a aplicação Python/FastAPI. Usa Poetry para gerenciamento de dependências, copia o código da aplicação, expõe a porta 8000 e define o comando para iniciar Uvicorn.
├── docker-compose.yml       # Orquestra os contêineres da aplicação (`app`), banco de dados (`db` com imagem `ankane/pgvector`), e `pgadmin`. Gerencia redes, volumes (para persistência de dados do PostgreSQL e live-reloading do código da app) e variáveis de ambiente.
├── pyproject.toml           # Define metadados do projeto (nome, versão) e dependências Python, gerenciadas pelo Poetry.
├── poetry.lock              # Garante builds determinísticos travando as versões exatas das dependências.
└── README.md                # Documentação principal do projeto, instruções de setup, visão geral da arquitetura, etc.

- app/: Contém todo o código da sua aplicação FastAPI.

- core/: Para configurações transversais como config.py (que pode usar Pydantic para validar variáveis de ambiente) e logging_config.py.

- domain/: O coração da sua aplicação, conforme os princípios DDD. Aqui residem:

- entities/: Seus modelos de domínio ricos (DocumentoRAG, ChunkRAG com seus metadados, Agendamento, Profissional, Especialidade, etc.).

- value_objects/: Conceitos imutáveis como Embedding, IntervaloDeData, StatusAgendamento.

- services/: Lógica de domínio que não pertence a uma entidade específica.

- repositories/: Interfaces (contratos) para persistência, como i_rag_repository.py (para buscar chunks e embeddings) e i_appointment_repository.py.

- application/: Orquestra os casos de uso.

- services/: conversational_agent_service.py irá interagir com o LangGraph (definido em app/agents/). rag_app_service.py cuidará da lógica de ingestão (chamando processamento de texto, geração de embedding e o repositório RAG) e recuperação. appointment_app_service.py irá interagir com o clinic_api_client.py.

- infrastructure/: Implementações concretas.

- database/repositories/: Implementações dos repositórios (ex: postgres_rag_repository.py usando pgvector).

- external_apis/: Clientes para a API da clínica e WhatsApp.

- embeddings/: Onde você implementará a lógica para gerar embeddings usando um modelo específico.

- presentation/: Interface da sua aplicação com o mundo externo (FastAPI).

- routers/: Seus endpoints, incluindo o webhook_router.py para o WhatsApp e o health_router.py.

- schemas/: Modelos Pydantic para validação de dados de entrada e formatação de saída das suas rotas. Estes são distintos dos modelos de domínio, servindo como DTOs para a API.

- agents/: Este diretório é crucial para a sua lógica de IA.

- langgraph_config/: Aqui você define o estado do grafo (graph_state.py com Pydantic), os nós (nodes.py - cada função é uma etapa do fluxograma de ai.md, como saudação, coleta de dados para agendamento, processamento de pergunta RAG) e as arestas (edges.py).

- pydantic_ai_tools/: Aqui você pode aproveitar PydanticAI. Por exemplo, criar um InformationExtractorAgent 1 para analisar a mensagem do usuário e extrair informações estruturadas (como nome, preferência de especialidade, data desejada). O output_type deste agente seria um modelo Pydantic (definido, por exemplo, em app/presentation/schemas/ ou app/application/dtos/). Este agente PydanticAI pode ser invocado por um dos nós do seu LangGraph.

- utils/: Para funções utilitárias, como text_processing.py para o chunking semântico mencionado em ai.md.

- scripts/: Para tarefas pontuais como ingest_rag_data.py (carregar documentos, chunking, gerar embeddings e salvar no pgvector via rag_app_service.py) e setup_pgvector.py.

- tests/: Essencial para garantir a qualidade, com separação entre testes unitários e de integração.

- Arquivos Raiz: Dockerfile, docker-compose.yml (para rodar a aplicação e o PostgreSQL com pgvector), .env.example, etc., conforme boas práticas.