# 🤖 Vyapar Sathi – AI Copilot Microservice

> **Enterprise Autonomous Agentic Copilot & Predictive Demand Intelligence for Retail Operations**  
> Powered by **FastAPI, LangGraph, Google Gemini, Mem0, CatBoost ML, Pinecone, and MongoDB**.

---

## 📖 Overview

**vyaparSathi-ai-service** is the intelligence microservice of the **Vyapar Sathi** retail platform. Built around a sophisticated **LangGraph StateGraph**, the AI Copilot functions as an autonomous retail operations assistant. It answers complex operational queries, forecasts product demand using Machine Learning, prioritizes restocks, recalls store-specific historical context via **Mem0**, and supports full-duplex **real-time voice interactions**.

---

## 🧠 Autonomous Agent Architecture

The AI Agent is implemented as a looped, stateful multi-node graph using **LangGraph**:

```
                       [ User Request ]
                              │
                              ▼
                      ┌───────────────┐
                      │  Grader Node  │ ──► (Out of Scope / Safety Rejection)
                      └───────┬───────┘
                              │
                              ▼
                      ┌───────────────┐
                      │  Intent Node  │ ──► (live_data, memory, recap, general)
                      └───────┬───────┘
                              │
                              ▼
                      ┌───────────────┐
                      │ Planner Node  │ ──► (Decomposes complex requests)
                      └───────┬───────┘
                              │
                              ▼
                      ┌───────────────┐
                      │  Think Node   │ ◄─────────────────────────┐
                      └───────┬───────┘                           │
                              │ (Tool Calls)                      │
                              ▼                                   │
                      ┌───────────────┐                           │
                      │  Tool Node    │ ──► (Runs Domain Tools) ──┘
                      └───────┬───────┘
                              │ (Final Synthesis)
                              ▼
                      ┌───────────────┐
                      │ Final Output  │
                      └───────────────┘
```

### Core Architecture Nodes
1. **Grader Node (`grader_node.py`)**: Security & domain scope validation. Filters non-retail, prompt-injection, or out-of-scope queries before reaching the reasoning model.
2. **Intent Classifier (`intent_node.py`)**: Detects request complexity and intent (`live_data`, `memory`, `recap`, `mixed`, `general`).
3. **Planner Node (`planner_node.py`)**: Formulates multi-step execution plans for complex analytical tasks (e.g., comparing cross-store profit margins against vendor purchase prices).
4. **Think Node (`think_node.py`)**: Central LLM reasoning node powered by **Google Gemini** with automatic failover to **Groq (Llama-3.3-70b)** or **NVIDIA NIM**.
5. **Memory Retrieval (`memory_query.py`)**: Fetches long-term contextual memory from **Mem0** (e.g., store owner preferences, typical discount thresholds, vendor habits).
6. **Tool Execution Node (`tool_node.py`)**: Executes sandboxed tools across 20+ specialized domain modules.
7. **MongoDB Checkpointer (`checkpointer.py`)**: Uses `langgraph-checkpoint-mongodb` to persist complete conversation history and agent state across `thread_id` sessions.

---

## ⚡ Core Capabilities

- **📈 Machine Learning Sales Forecasting**:
  - Embedded **CatBoost ML** regression engine (`catboost==1.2.8`).
  - Generates multi-day demand predictions, stockout probability, and automated safety stock calculations.
- **🎙️ Real-Time Voice Bot (`/ws/voice`)**:
  - Low-latency bi-directional WebSocket interface for counter-side voice commands and hands-free retail operations.
- **🧠 Long-Term Memory (Mem0)**:
  - Retains personalized store knowledge and user preferences across multiple chat sessions.
- **🔍 Vector Catalog Search (Pinecone)**:
  - Semantic vector search for fast product identification and context matching.
- **🌐 Real-Time External Market Intelligence**:
  - Integrated with **Tavily** and **Firecrawl** to search competitor pricing, market trends, and supplier catalogs.
- **📄 Automated Document & Report Generation**:
  - Generates comprehensive Excel spreadsheets (`openpyxl`) and formatted Word reports (`python-docx`).
  - Uploads generated files securely to **Cloudflare R2 Object Storage** with presigned download links.
- **📊 Observability & Metrics**:
  - Full Prometheus integration (`/metrics`) and structured JSON logs with **Structlog**.

---

## 🛠️ Specialized Agent Tools (20+ Domains)

| Domain | Capabilities |
|---|---|
| **Inventory** | Stock levels, low-stock warnings, expiry tracking, SKU lookups |
| **Sales** | Daily/weekly revenue, invoice lookups, sales velocity, payment breakdowns |
| **Forecasting** | CatBoost ML demand predictions, restock priority scoring |
| **Profit & Loss** | Net profit, cost of goods sold (COGS), margin calculations |
| **Purchases** | Supplier order history, incoming shipment status, PO reconciliation |
| **Suppliers & Sellers** | Vendor directory, credit balances, purchase terms |
| **Buyers & Customers** | Customer Khata balances, repeat purchase analysis |
| **Expenses** | Operational overhead, miscellaneous store expenditure |
| **Communication** | Trigger email alerts to suppliers, send receipt summaries via Express backend |
| **Web & Intelligence** | Real-time web pricing lookup, vendor search |
| **Code Execution** | Sandboxed calculations for custom business analytics |

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|---|---|---|
| **API Framework** | FastAPI 0.111 & Uvicorn | High-performance asynchronous Python REST & WebSocket framework |
| **Agentic Framework** | LangGraph & LangChain | StateGraph workflow, cyclic agent reasoning, and tool execution |
| **Primary LLM** | Google Gemini (`langchain-google-genai`) | Gemini 2.0 Flash / Gemini 1.5 Pro |
| **Fallback LLMs** | Groq & NVIDIA NIM | Ultra-fast Llama-3.3 fallbacks for maximum uptime |
| **Conversation State** | `langgraph-checkpoint-mongodb` | Persistent state checkpointer in MongoDB |
| **Long-Term Memory** | Mem0 (`mem0ai`) | Long-term memory store for personalized retail context |
| **Demand Forecasting** | CatBoost 1.2.8 | Gradient boosting ML model for demand prediction |
| **Vector Database** | Pinecone | Semantic indexing and product catalog embedding retrieval |
| **Web Search & Scraping** | Tavily & Firecrawl | Real-time market intelligence and supplier lookups |
| **Cloud Storage** | Cloudflare R2 (boto3) | S3-compatible cloud storage for generated reports |
| **Caching & Workers** | Redis & BullMQ | Redis cache and distributed worker queue tasks |
| **Observability** | Prometheus & Structlog | Metrics collection and structured JSON logging |

---

## 📂 Project Structure

```text
vyaparSathi-ai-service/
├── app/
│   ├── agent/                          # LangGraph Agent Core
│   │   ├── checkpointer.py             # MongoDB conversation checkpointer
│   │   ├── graph.py                    # StateGraph definition & conditional edges
│   │   ├── state.py                    # VyaparAgentState schema & reducers
│   │   ├── nodes/                      # Agent workflow nodes
│   │   │   ├── grader_node.py          # Safety & scope guardrail
│   │   │   ├── intent_node.py          # Intent classification & plan triggers
│   │   │   ├── planner_node.py         # Multi-step query planning
│   │   │   ├── think_node.py           # LLM reasoning & tool dispatcher
│   │   │   ├── memory_query.py         # Mem0 memory recall
│   │   │   └── tool_node.py            # Sandboxed tool runner
│   │   ├── tools/                      # 20+ Domain tool modules
│   │   │   ├── analytics/              # Store analytics & baselines
│   │   │   ├── forecast/               # Demand forecasting tools
│   │   │   ├── inventory/              # Inventory query & stockout tools
│   │   │   ├── sales/                  # Revenue & order tools
│   │   │   ├── profit_loss/            # Financial & COGS tools
│   │   │   ├── purchases/              # Supplier purchase tools
│   │   │   ├── registry.py             # Global tool registry
│   │   │   └── web/                    # Tavily / Firecrawl web search
│   │   ├── memory/                     # Mem0 client configuration
│   │   ├── prompts/                    # System prompts & few-shot examples
│   │   └── subgraphs/                  # Specialized analytical subgraphs
│   ├── config/                         # App configuration & settings
│   │   ├── database.py                 # Async Motor MongoDB connection
│   │   ├── redis.py                    # Redis cache setup
│   │   ├── logging.py                  # Structlog JSON logger config
│   │   └── settings.py                 # Pydantic BaseSettings (.env loading)
│   ├── controllers/                    # Route handlers & logic
│   ├── routes/                         # FastAPI APIRouters
│   │   ├── chat_routes.py              # Chat completions & SSE streaming
│   │   ├── forecast_routes.py          # ML demand prediction endpoints
│   │   ├── voice_routes.py             # WebSocket voice endpoint (/ws/voice)
│   │   ├── files_routes.py             # Report generation & R2 downloads
│   │   ├── analytics_routes.py         # Store analytical metrics
│   │   ├── insight_routes.py           # Restock & operational insights
│   │   └── health_routes.py            # Healthcheck & Prometheus metrics
│   ├── services/                       # Business & ML services
│   │   └── forecast_service.py         # CatBoost model inference engine
│   ├── workers/                        # Background BullMQ & Redis workers
│   └── main.py                         # FastAPI app lifecycle & middleware
├── artifacts/                          # Trained CatBoost model files (.cbm)
├── scripts/                            # Model training & data preparation scripts
├── tests/                              # Pytest test suite
├── requirements.txt                    # Python dependency manifest
└── runtime.txt                         # Python version specification (3.11)
```

---

## 🚦 API Endpoints Reference

### 1. Health & Observability
- `GET /health` - System health (MongoDB, Redis, ML model status).
- `GET /metrics` - Prometheus metrics for request latency, token consumption, and errors.

### 2. Conversational Agent & Chat
- `POST /api/chat` - Query the LangGraph Agent (returns response, tool outputs, and execution plan).
- `POST /api/chat/stream` - Server-Sent Events (SSE) streaming response tokens in real-time.
- `GET /api/chat/history/:threadId` - Fetch persisted conversation state from MongoDB checkpointer.

### 3. Machine Learning Forecasting (`/api/forecast`)
- `GET /api/forecast/:storeId` - Store-level demand forecast for the next 7/14/30 days.
- `GET /api/forecast/:storeId/restock` - Prioritized restock recommendations with reorder points.

### 4. Real-Time Voice WebSocket (`/ws/voice`)
- `WS /ws/voice` - Full-duplex WebSocket for streaming audio/text interaction with low-latency response.

### 5. Document Generation & Storage (`/api/files`)
- `POST /api/files/generate-report` - Generate Excel or Word report and upload to Cloudflare R2.
- `GET /api/files/download/:fileKey` - Get presigned R2 download URL.

---

## ⚙️ Environment Variables

Create `.env` in `vyaparSathi-ai-service/`:

```env
# Primary LLM API Keys
GEMINI_API_KEY=your_gemini_api_key

# Secondary / Fallback LLM API Keys (Groq / NVIDIA / OpenRouter)
GROQ_API_KEY=your_groq_api_key
NVIDIA_API_KEY=your_nvidia_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

# Databases
MONGO_URL=mongodb+srv://<user>:<password>@<cluster>.mongodb.net
MONGO_DB_NAME=vyapar-sathi
REDIS_URL=redis://default:<password>@<host>:<port>

# Long-Term Memory (Mem0)
MEM0_API_KEY=your_mem0_api_key
MEM0_ENABLED=true

# Vector Database (Pinecone)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=vyapar-sathi

# External Intelligence
TAVILY_API_KEY=your_tavily_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key

# Cloudflare R2 Object Storage
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_R2_ENDPOINT=https://<account_id>.r2.cloudflarestorage.com
CLOUDFLARE_R2_ACCESS_KEY_ID=your_access_key
CLOUDFLARE_R2_SECRET_ACCESS_KEY=your_secret_key
CLOUDFLARE_R2_BUCKET_NAME=vyapar-sathi-reports

# Express Backend Service
EXPRESS_BACKEND_URL=http://localhost:5000

# LangSmith Tracing (Optional)
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=vyaparSathi-ai
LANGCHAIN_TRACING_V2=true
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python**: 3.11+
- **MongoDB**: Active MongoDB instance with replica set (for checkpointer transaction support)
- **Redis**: Running Redis instance

### 2. Environment Setup
```bash
cd vyaparSathi-ai-service

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The AI service will be accessible at `http://localhost:8000`.  
Interactive Swagger docs: `http://localhost:8000/docs`.

### 4. Running Tests
```bash
# Run full pytest test suite
pytest

# Run tests with output and verbose logging
pytest -v -s

# Run store isolation tests specifically
pytest tests/test_store_comparison_isolation.py
```

---

## 🛡️ Guardrails & Resilience

- **Failover Chain**: If primary LLM rate limits or returns 5xx, the execution router automatically switches to backup providers (Groq/NVIDIA) without dropping user context.
- **Strict Data Isolation**: Every tool call validates `store_id` against the requester's context to eliminate cross-tenant data leaks.
- **Context Compression**: The summarizer compresses large tool outputs before passing them into the main context window, keeping inference costs low and response speeds high.
