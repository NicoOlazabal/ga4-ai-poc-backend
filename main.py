"""
GA4 AI Analytics POC - Backend
FastAPI + BigQuery + LLM (Claude/OpenAI)
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
import anthropic

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ─── Config ───────────────────────────────────────────────────────────────────

GCP_PROJECT_ID    = os.getenv("GCP_PROJECT_ID", "your-gcp-project")
BQ_DATASET        = os.getenv("BQ_DATASET", "analytics_XXXXXXXXX")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_CLIENT_ID  = os.getenv("GOOGLE_CLIENT_ID", "")
ALLOWED_EMAILS    = {e.strip() for e in os.getenv("ALLOWED_EMAILS", "").split(",") if e.strip()}

# ─── Auth ─────────────────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)

async def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> str:
    if not GOOGLE_CLIENT_ID:
        return "dev@local"
    if not creds:
        raise HTTPException(status_code=401, detail="Token ausente")
    try:
        info = google_id_token.verify_oauth2_token(
            creds.credentials, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        email = info.get("email", "")
        if ALLOWED_EMAILS and email not in ALLOWED_EMAILS:
            raise HTTPException(status_code=403, detail=f"Acesso negado: {email}")
        return email
    except ValueError:
        raise HTTPException(status_code=401, detail="Token inválido ou expirado")

# ─── Clients ──────────────────────────────────────────────────────────────────

_creds_json_str = os.getenv("GOOGLE_CREDENTIALS_JSON")
if _creds_json_str:
    _creds_info = json.loads(_creds_json_str)
    _credentials = service_account.Credentials.from_service_account_info(
        _creds_info,
        scopes=["https://www.googleapis.com/auth/bigquery"]
    )
    bq_client = bigquery.Client(project=GCP_PROJECT_ID, credentials=_credentials)
else:
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── Schema helper ────────────────────────────────────────────────────────────

GA4_SCHEMA_CONTEXT = f"""
Você tem acesso ao dataset GA4 do Google BigQuery.
Projeto: {GCP_PROJECT_ID}
Dataset: {BQ_DATASET}

TABELAS DISPONÍVEIS:
- `{GCP_PROJECT_ID}.{BQ_DATASET}.events_*`   → Eventos diários do GA4 (use _TABLE_SUFFIX para filtrar datas)
- `{GCP_PROJECT_ID}.{BQ_DATASET}.events_intraday_*` → Eventos do dia atual (intraday)

COLUNAS PRINCIPAIS da tabela events_*:
- event_date         STRING    (formato YYYYMMDD)
- event_timestamp    INT64     (microssegundos UTC)
- event_name         STRING    (page_view, session_start, purchase, etc.)
- event_params       ARRAY<STRUCT<key STRING, value STRUCT<...>>>
- user_pseudo_id     STRING    (identificador anônimo do usuário)
- user_id            STRING    (user ID logado, pode ser NULL)
- device.category    STRING    (desktop, mobile, tablet)
- device.browser     STRING
- geo.country        STRING
- geo.city           STRING
- traffic_source.source   STRING
- traffic_source.medium   STRING
- traffic_source.name     STRING
- ecommerce.purchase_revenue FLOAT64
- ecommerce.transaction_id  STRING

FUNÇÕES ÚTEIS PARA event_params:
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location') AS page_url
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_title') AS page_title
  (SELECT value.int_value    FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS session_id
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'source') AS source

REGRAS IMPORTANTES:
1. SEMPRE use _TABLE_SUFFIX para filtrar datas: WHERE _TABLE_SUFFIX BETWEEN '20240101' AND '20241231'
2. Limite resultados com LIMIT para evitar custos altos (padrão: LIMIT 1000)
3. Use FORMAT_DATE e PARSE_DATE quando necessário
4. Para agregar sessões únicas, agrupe por user_pseudo_id + ga_session_id
5. Nunca faça full table scan sem filtro de data
"""

# ─── Pydantic models ──────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question:   str
    date_start: Optional[str] = None   # YYYYMMDD
    date_end:   Optional[str] = None   # YYYYMMDD
    max_rows:   Optional[int] = 500
    language:   Optional[str] = "Português"

class QueryResponse(BaseModel):
    question:   str
    sql_query:  str
    data:       list
    insight:    str
    row_count:  int
    bytes_processed: Optional[int] = None
    executed_at: str

# ─── Core functions ───────────────────────────────────────────────────────────

def generate_sql_from_question(question: str, date_start: str, date_end: str) -> str:
    """Usa Claude para transformar pergunta em SQL BigQuery."""

    prompt = f"""
{GA4_SCHEMA_CONTEXT}

Período padrão para esta query:
- date_start: {date_start}
- date_end:   {date_end}

PERGUNTA DO USUÁRIO:
"{question}"

Gere APENAS o SQL BigQuery válido para responder essa pergunta.
Regras:
- Retorne SOMENTE o SQL, sem explicações, sem markdown, sem backticks
- Use o período date_start e date_end fornecido
- Adicione LIMIT 1000 ao final, salvo se outra quantidade for especificada
- A query deve ser eficiente e segura (sem full scans)
- Use aliases legíveis nas colunas (pt-BR é ok)
"""

    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    sql = response.content[0].text.strip()
    # Remove possíveis backticks que o modelo possa ter incluído
    sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^```\s*",    "", sql)
    sql = re.sub(r"\s*```$",    "", sql)
    return sql.strip()


def execute_bigquery(sql: str, max_rows: int = 500) -> tuple[list, int]:
    """Executa SQL no BigQuery e retorna rows + bytes processados."""
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=10 * 1024 ** 3  # 10 GB safety cap
    )
    query_job = bq_client.query(sql, job_config=job_config)
    results   = query_job.result()

    rows = []
    for row in results:
        rows.append(dict(row))
        if len(rows) >= max_rows:
            break

    bytes_processed = query_job.total_bytes_processed
    return rows, bytes_processed


def generate_insight(question: str, sql: str, data: list, language: str = "Português") -> str:
    """Usa Claude para interpretar os dados e gerar insight no idioma especificado."""

    data_sample = json.dumps(data[:50], ensure_ascii=False, default=str)

    prompt = f"""
Você é um especialista em Analytics e Growth.
O usuário fez a seguinte pergunta sobre seus dados do GA4:

PERGUNTA: "{question}"

SQL executado:
{sql}

DADOS RETORNADOS ({len(data)} linhas, mostrando até 50):
{data_sample}

Com base nesses dados reais:
1. Responda a pergunta de forma clara e objetiva
2. Destaque os números mais relevantes
3. Identifique padrões ou anomalias importantes
4. Sugira 1-2 ações práticas baseadas nos dados

Responda em {language}, de forma concisa mas completa (máx. 300 palavras).
Use emojis com moderação para destacar pontos-chave.
"""

    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

# ─── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 GA4 AI Analytics POC started")
    yield
    print("🛑 Shutting down")

app = FastAPI(
    title="GA4 AI Analytics POC",
    description="Analise dados do GA4/BigQuery usando linguagem natural com IA",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção: restrinja aos seus domínios
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/config")
def get_config():
    return {"google_client_id": GOOGLE_CLIENT_ID, "auth_enabled": bool(GOOGLE_CLIENT_ID)}

@app.get("/")
def serve_frontend():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/ask", response_model=QueryResponse)
async def ask(req: QuestionRequest, user: str = Depends(get_current_user)):
    """
    Endpoint principal: recebe pergunta em linguagem natural,
    gera SQL, executa no BigQuery e retorna insight da IA.
    """
    # Defaults de data: últimos 30 dias
    today = datetime.utcnow()
    date_end   = req.date_end   or today.strftime("%Y%m%d")
    date_start = req.date_start or (today - timedelta(days=30)).strftime("%Y%m%d")

    try:
        # 1. Gera SQL com IA
        sql = generate_sql_from_question(req.question, date_start, date_end)
    except Exception as e:
        raise HTTPException(502, f"Erro ao gerar SQL: {e}")

    try:
        # 2. Executa no BigQuery
        data, bytes_processed = execute_bigquery(sql, req.max_rows or 500)
    except Exception as e:
        raise HTTPException(502, f"Erro no BigQuery: {e}\n\nSQL gerado:\n{sql}")

    try:
        # 3. Gera insight com IA
        insight = generate_insight(req.question, sql, data, req.language or "Português")
    except Exception as e:
        insight = f"(Erro ao gerar insight: {e})"

    return QueryResponse(
        question=req.question,
        sql_query=sql,
        data=data,
        insight=insight,
        row_count=len(data),
        bytes_processed=bytes_processed,
        executed_at=datetime.utcnow().isoformat()
    )


@app.get("/sample-questions")
def sample_questions():
    """Retorna perguntas de exemplo para o usuário testar."""
    return {
        "questions": [
            "Quantos usuários únicos tivemos nos últimos 30 dias?",
            "Quais são as 10 páginas mais visitadas?",
            "De quais países vieram mais visitantes?",
            "Qual é a taxa de conversão de purchase nos últimos 7 dias?",
            "Qual canal de marketing trouxe mais sessões esse mês?",
            "Quais dispositivos os usuários mais usam (mobile, desktop, tablet)?",
            "Qual é o tempo médio de engajamento por sessão?",
            "Quais eventos aconteceram mais vezes ontem?",
        ]
    }
