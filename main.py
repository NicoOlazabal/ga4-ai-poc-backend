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
    user_level: Optional[str] = "intermediate"  # beginner | intermediate | advanced
    industry:   Optional[str] = "other"         # GA4 industry category key

class QueryResponse(BaseModel):
    question:   str
    sql_query:  str
    data:       list
    insight:    str
    row_count:  int
    bytes_processed: Optional[int] = None
    executed_at: str

# ─── Core functions ───────────────────────────────────────────────────────────

# ─── Industry context ─────────────────────────────────────────────────────────

INDUSTRY_SQL_HINTS: dict[str, str] = {
    "ecommerce": """
CONTEXTO DE INDÚSTRIA — E-COMMERCE:
Eventos GA4 prioritários: purchase, add_to_cart, begin_checkout, view_item, view_item_list,
  remove_from_cart, add_to_wishlist, refund.
Métricas-chave: ecommerce.purchase_revenue, ecommerce.transaction_id,
  (SELECT value.int_value FROM UNNEST(event_params) WHERE key='value') AS item_value.
Foco em: funil de conversão (view_item → add_to_cart → begin_checkout → purchase),
  receita por canal, ticket médio, taxa de abandono de carrinho.
Quando calcular conversão use COUNT(DISTINCT CASE WHEN event_name='purchase' THEN user_pseudo_id END)
  / COUNT(DISTINCT user_pseudo_id) AS taxa_conversao.
""",
    "saas": """
CONTEXTO DE INDÚSTRIA — SAAS / SOFTWARE:
Eventos GA4 prioritários: sign_up, login, trial_start, subscription_start, feature_used,
  page_view em páginas /dashboard, /pricing, /signup, generate_lead.
Foco em: ativação de usuários (sign_up → first feature use), retenção (DAU/WAU/MAU),
  páginas de pricing e conversão de trial.
Para cohort/retenção agrupe por DATE_TRUNC(PARSE_DATE('%Y%m%d', event_date), WEEK).
Métricas: usuários ativos = COUNT(DISTINCT user_pseudo_id) com pelo menos 1 evento por período.
""",
    "media": """
CONTEXTO DE INDÚSTRIA — MEDIA & PUBLISHING:
Eventos GA4 prioritários: page_view, scroll, video_start, video_complete, video_progress,
  search, share, user_engagement.
Métricas-chave: (SELECT value.int_value FROM UNNEST(event_params) WHERE key='percent_scrolled') AS scroll_pct,
  (SELECT value.int_value FROM UNNEST(event_params) WHERE key='engaged_session_event') AS engaged,
  (SELECT value.int_value FROM UNNEST(event_params) WHERE key='engagement_time_msec') / 1000 AS engagement_sec.
Foco em: páginas mais lidas, profundidade de scroll, tempo de engajamento, tráfego por categoria de conteúdo.
""",
    "finance": """
CONTEXTO DE INDÚSTRIA — FINANÇAS / FINTECH:
Eventos GA4 prioritários: generate_lead, form_submit, calculator_use, page_view em /produtos /simulador /abrir-conta,
  login, view_item (produtos financeiros).
Foco em: geração de leads qualificados, páginas de produto mais visitadas, funil de abertura de conta,
  uso de simuladores/calculadoras, origem do tráfego de maior intenção.
Métricas: leads = COUNT(DISTINCT CASE WHEN event_name='generate_lead' THEN user_pseudo_id END).
""",
    "education": """
CONTEXTO DE INDÚSTRIA — EDUCAÇÃO / EDTECH:
Eventos GA4 prioritários: page_view em /curso /aula, video_start, video_complete, video_progress,
  sign_up, enroll (custom), certificate_earned (custom), search.
Foco em: consumo de conteúdo (taxa de conclusão de vídeos), matrículas, buscas internas,
  páginas de curso mais visitadas, progressão do aluno.
Taxa conclusão vídeo: COUNT(CASE WHEN event_name='video_complete' THEN 1 END) /
  NULLIF(COUNT(CASE WHEN event_name='video_start' THEN 1 END), 0).
""",
    "travel": """
CONTEXTO DE INDÚSTRIA — VIAGENS & TURISMO:
Eventos GA4 prioritários: search, view_item (destinos/pacotes), add_to_cart, begin_checkout,
  purchase, view_promotion.
Params importantes: (SELECT value.string_value FROM UNNEST(event_params) WHERE key='destination') AS destino,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key='origin') AS origem.
Foco em: destinos mais pesquisados, funil de reserva, sazonalidade, origem do tráfego de maior conversão,
  valor médio de reserva (ecommerce.purchase_revenue).
""",
    "health": """
CONTEXTO DE INDÚSTRIA — SAÚDE & BEM-ESTAR:
Eventos GA4 prioritários: page_view em /sintomas /tratamentos /médicos, generate_lead,
  schedule_appointment (custom), search, form_submit.
Foco em: conteúdo de saúde mais consumido, buscas internas (sintomas/especialidades),
  agendamentos/leads gerados, tráfego orgânico (health queries), dispositivos mais usados.
Note: dados de saúde são sensíveis; não revelar dados individuais, sempre agregar.
""",
    "food": """
CONTEXTO DE INDÚSTRIA — ALIMENTAÇÃO & RESTAURANTES:
Eventos GA4 prioritários: view_menu (custom), find_location, make_reservation, order_online,
  page_view em /cardapio /delivery, purchase (pedidos online).
Foco em: páginas de cardápio mais visitadas, horários de pico de acesso, conversão para pedido/reserva,
  origem do tráfego (Google Maps, busca local), dispositivo (mobile prioritário).
""",
    "realestate": """
CONTEXTO DE INDÚSTRIA — IMÓVEIS:
Eventos GA4 prioritários: view_item (imóvel específico), search, generate_lead, schedule_visit (custom),
  page_view em /venda /aluguel /lançamentos.
Params: (SELECT value.string_value FROM UNNEST(event_params) WHERE key='property_type') AS tipo_imovel.
Foco em: imóveis mais visualizados, buscas por tipo/localidade, leads gerados por página,
  funil de interesse → contato, origem de tráfego mais convertida.
""",
    "games": """
CONTEXTO DE INDÚSTRIA — GAMES:
Eventos GA4 prioritários: level_start, level_end, level_up, unlock_achievement, ad_impression,
  earn_virtual_currency, spend_virtual_currency, purchase (in-app), tutorial_begin, tutorial_complete.
Foco em: retenção D1/D7/D30, completion rate de levels/tutorial, monetização (ARPU/ARPPU),
  eventos de progressão, funil de compra in-app.
Para D1 retention: compare users who had events on day 0 vs day 1 usando event_date.
""",
    "automotive": """
CONTEXTO DE INDÚSTRIA — AUTOMOTIVO:
Eventos GA4 prioritários: view_item (modelo de veículo), generate_lead, schedule_test_drive (custom),
  configurator_use (custom), page_view em /modelos /concessionarias /financiamento.
Foco em: modelos mais visualizados, agendamentos de test drive, uso do configurador,
  funil pesquisa → lead → agendamento, origem do tráfego qualificado.
""",
    "beauty": """
CONTEXTO DE INDÚSTRIA — BELEZA & FITNESS:
Eventos GA4 prioritários: view_item, add_to_cart, purchase, search, page_view em /produtos /rotina /blog,
  video_start (tutoriais), generate_lead (agendamentos).
Foco em: produtos mais visualizados vs comprados, buscas de produtos, tutorial engagement,
  agendamentos (salão/academia), sazonalidade de compras.
""",
    "sports": """
CONTEXTO DE INDÚSTRIA — ESPORTES:
Eventos GA4 prioritários: view_item (produtos/ingressos), purchase, page_view em /jogos /atletas /noticias,
  video_start (melhores momentos), search, add_to_cart.
Foco em: conteúdo esportivo mais consumido, vendas de ingressos/produtos, picos de tráfego
  associados a eventos ao vivo, engajamento com vídeos, tráfego mobile vs desktop.
""",
    "b2b": """
CONTEXTO DE INDÚSTRIA — B2B / MERCADOS INDUSTRIAIS:
Eventos GA4 prioritários: generate_lead, form_submit, page_view em /produtos /solucoes /cases /contato,
  file_download (whitepapers/brochures), search, video_start (demos).
Foco em: páginas de solução mais visitadas, downloads de materiais, leads por formulário,
  tráfego de empresas (usar geo.city para proxy), jornada longa de conversão,
  canais de maior intenção (busca orgânica branded vs non-branded).
""",
    "entertainment": """
CONTEXTO DE INDÚSTRIA — ARTE & ENTRETENIMENTO:
Eventos GA4 prioritários: page_view, video_start, video_complete, purchase (ingressos),
  view_item (eventos/shows), search, share.
Foco em: conteúdo e eventos mais populares, vendas de ingressos, engajamento com vídeos/trailers,
  tráfego por tipo de evento, sazonalidade, compartilhamentos.
""",
    "other": """
CONTEXTO DE INDÚSTRIA — GERAL:
Use os eventos e métricas padrão do GA4 relevantes para a pergunta.
Eventos comuns: page_view, session_start, user_engagement, scroll, click, search.
""",
}

INDUSTRY_INSIGHT_CONTEXT: dict[str, str] = {
    "ecommerce":    "Foque em conversão, receita, ticket médio e performance de produtos. Use benchmarks de e-commerce (ex: taxa de conversão típica 1-4%).",
    "saas":         "Foque em ativação, retenção e engajamento com features. Métricas-chave: DAU/MAU, trial-to-paid conversion, churn signals.",
    "media":        "Foque em engajamento de conteúdo, tempo de leitura, scroll depth e crescimento de audiência.",
    "finance":      "Foque em geração de leads qualificados, intenção de compra e funil de abertura de conta/produto. Seja conservador em projeções.",
    "education":    "Foque em engajamento com conteúdo, conclusão de cursos/vídeos e conversão de matrículas.",
    "travel":       "Foque em intenção de viagem, funil de reserva, sazonalidade e destinos de maior demanda.",
    "health":       "Foque em conteúdo mais buscado, agendamentos/leads e acesso mobile. Nunca extrapole dados individuais.",
    "food":         "Foque em horários de pico, páginas de cardápio/delivery e conversão para pedido ou reserva.",
    "realestate":   "Foque em imóveis de maior interesse, geração de leads e jornada do comprador (longa, multicanal).",
    "games":        "Foque em retenção, progressão e monetização. Compare D1/D7/D30 retention e ARPU.",
    "automotive":   "Foque em modelos de maior interesse, test drives agendados e jornada pesquisa-→-concessionária.",
    "beauty":       "Foque em produtos mais desejados, sazonalidade (datas comemorativas) e engajamento com tutoriais.",
    "sports":       "Foque em picos de tráfego relacionados a eventos, vendas de ingressos/produtos e engajamento mobile.",
    "b2b":          "Foque em qualidade dos leads, conteúdo de maior intenção e jornada longa de decisão. CAC e pipeline são prioritários.",
    "entertainment":"Foque em popularidade de conteúdo, vendas de ingressos e engajamento com trailers/vídeos.",
    "other":        "Analise os dados com foco em padrões de comportamento relevantes para o negócio.",
}


def generate_sql_from_question(question: str, date_start: str, date_end: str, industry: str = "other") -> str:
    """Usa Claude para transformar pergunta em SQL BigQuery."""

    industry_hint = INDUSTRY_SQL_HINTS.get(industry, INDUSTRY_SQL_HINTS["other"])

    prompt = f"""
{GA4_SCHEMA_CONTEXT}

Período padrão para esta query:
- date_start: {date_start}
- date_end:   {date_end}

{industry_hint}

PERGUNTA DO USUÁRIO:
"{question}"

Gere APENAS o SQL BigQuery válido para responder essa pergunta.
Regras:
- Retorne SOMENTE o SQL, sem explicações, sem markdown, sem backticks
- Use o período date_start e date_end fornecido
- Priorize os eventos e métricas da indústria indicada acima quando relevante
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


def generate_insight(question: str, sql: str, data: list, language: str = "Português", user_level: str = "intermediate", industry: str = "other") -> str:
    """Usa Claude para interpretar os dados e gerar insight no idioma especificado."""

    data_sample = json.dumps(data[:50], ensure_ascii=False, default=str)

    level_instructions = {
        "beginner": """
O usuário é INICIANTE em Analytics e GA4. Adapte sua resposta:
- Use linguagem simples, sem jargões técnicos
- Quando mencionar métricas (sessões, bounce rate, CTR, etc.), explique brevemente o que significa em parênteses
- Inclua uma seção "📚 Conceitos desta análise" ao final com 2-3 definições curtas dos termos usados
- Foque em "o que isso significa para o negócio" mais do que nos números técnicos
- Compare com referências simples (ex: "isso é bom/médio/ruim para este tipo de site")
""",
        "intermediate": """
O usuário tem conhecimento INTERMEDIÁRIO de Analytics e GA4. Adapte sua resposta:
- Use linguagem profissional mas acessível
- Pode mencionar métricas padrão sem explicar cada uma
- Foque em padrões, tendências e ações concretas
- Inclua contexto de benchmark quando relevante
""",
        "advanced": """
O usuário é AVANÇADO em Analytics e GA4 (analista/desenvolvedor). Adapte sua resposta:
- Use terminologia técnica sem simplificações
- Inclua observações sobre a qualidade dos dados ou limitações da query se relevante
- Pode sugerir segmentações ou análises complementares avançadas
- Seja direto e técnico, sem explicações básicas
"""
    }

    level_context = level_instructions.get(user_level, level_instructions["intermediate"])
    industry_context = INDUSTRY_INSIGHT_CONTEXT.get(industry, INDUSTRY_INSIGHT_CONTEXT["other"])

    prompt = f"""
Você é um especialista em Analytics e Growth com foco em {industry.upper()}.
O usuário fez a seguinte pergunta sobre seus dados do GA4:

PERGUNTA: "{question}"

SQL executado:
{sql}

DADOS RETORNADOS ({len(data)} linhas, mostrando até 50):
{data_sample}

PERFIL DO USUÁRIO:
{level_context}

CONTEXTO DA INDÚSTRIA:
{industry_context}

Com base nesses dados reais:
1. Responda a pergunta de forma clara e objetiva
2. Destaque os números mais relevantes
3. Identifique padrões ou anomalias importantes
4. Sugira 1-2 ações práticas baseadas nos dados

Responda em {language}, de forma concisa mas completa (máx. 400 palavras).
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
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "version": "2.0.0"}


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
        sql = generate_sql_from_question(req.question, date_start, date_end, req.industry or "other")
    except Exception as e:
        raise HTTPException(502, f"Erro ao gerar SQL: {e}")

    try:
        # 2. Executa no BigQuery
        data, bytes_processed = execute_bigquery(sql, req.max_rows or 500)
    except Exception as e:
        raise HTTPException(502, f"Erro no BigQuery: {e}\n\nSQL gerado:\n{sql}")

    try:
        # 3. Gera insight com IA
        insight = generate_insight(req.question, sql, data, req.language or "Português", req.user_level or "intermediate", req.industry or "other")
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

