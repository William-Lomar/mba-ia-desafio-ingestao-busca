# Desafio MBA Engenharia de Software com IA - Full Cycle

Ingestão de um PDF em um banco vetorial (PostgreSQL + pgvector) e busca via chat de linha de comando (RAG), usando LangChain e embeddings/LLM da OpenAI.

## Pré-requisitos

- Python 3.13
- Docker e Docker Compose
- Uma chave de API da OpenAI

## 1. Subir o banco de dados

```bash
docker compose up -d
```

Isso sobe um Postgres com a extensão `pgvector` já habilitada, na porta `5432` (usuário `postgres`, senha `postgres`, banco `rag`).

## 2. Configurar variáveis de ambiente

Copie o `.env.example` para `.env`:

```bash
cp .env.example .env
```

E preencha:

```env
OPENAI_API_KEY=sua-chave-da-openai
OPENAI_EMBEDDING_MODEL='text-embedding-3-small'
DATABASE_URL='postgresql+psycopg://postgres:postgres@localhost:5432/rag'
PG_VECTOR_COLLECTION_NAME='gpt5_collection'
PDF_PATH='D:\caminho\completo\para\mba-ia-desafio-ingestao-busca\document.pdf'
```

- `DATABASE_URL` deve apontar para o Postgres do `docker-compose.yml`.
- `PDF_PATH` é o caminho **completo (absoluto)** do PDF a ser ingerido.

## 3. Instalar as dependências

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

## 4. Ingerir o PDF

```bash
python src/ingest.py
```

O script carrega o PDF definido em `PDF_PATH`, divide o conteúdo em chunks, gera os embeddings e grava no Postgres. Rodar novamente sem alterar o `PDF_PATH` não duplica os dados — o script detecta que o arquivo já foi ingerido e pula o processamento.

## 5. Conversar com o conteúdo ingerido

```bash
python src/chat.py
```

Um chat interativo abre no terminal. Digite sua pergunta e pressione Enter; a resposta é gerada com base apenas nos trechos do PDF recuperados do banco vetorial. Digite `sair` (ou `exit`/`quit`) para encerrar.
