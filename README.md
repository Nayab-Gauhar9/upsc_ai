Polity Mentor – AI-Powered UPSC Prep Platform & RAG Engine
Live Website: politymentor.com

Polity Mentor is a full-stack, distributed educational technology platform designed for Indian Civil Services (UPSC) aspirants. It autonomously aggregates, processes, and classifies daily government press releases (PIB) and current affairs using an asynchronous ETL pipeline and serves them through a cross-platform mobile and web client.

The core of the platform is powered by a Retrieval-Augmented Generation (RAG) engine utilizing pgvector for semantic search and a highly fault-tolerant LLM orchestration layer for automated content normalization.

🏗 System Architecture
The system is designed with a decoupled, microservices-oriented architecture to ensure the heavy data ingestion and LLM processing tasks do not block the user-facing API.

Frontend: Flutter & FlutterFlow (iOS, Android, Web) communicating via REST APIs and Firebase Auth.

API Gateway / Backend: FastAPI running on Uvicorn/Gunicorn.

Task Queue: Celery & Redis for asynchronous background job execution.

Scheduler: Celery Beat for daily, automated cron-like scraping and ingestion.

Database Layer: PostgreSQL (relational data & pgvector embeddings) managed via Alembic migrations.

Object Storage: MinIO (S3-compatible) for raw JSON preservation, source images, and document chunking traceability.

LLM Layer: Provider rotation and API key pooling across Hugging Face and local Ollama instances to bypass rate limits.

✨ Core Engineering Features
1. Distributed Background ETL Pipeline
Instead of blocking the main application threads, daily ingestion of 30-40 government articles is handled entirely in the background. pib_rss.py and pib_article.py extract raw HTML/XML, which is pushed to a Redis message broker. Celery workers pick up these tasks, normalize the text, and load the final structured data into PostgreSQL.

2. Semantic Search & RAG Integration
Traditional keyword search falls short for complex constitutional and policy queries. The platform chunks incoming articles, generates vector embeddings using src/rag/embeddings.py, and stores them in PostgreSQL using the pgvector extension. This enables sub-second semantic retrieval across thousands of study notes.

3. Fault-Tolerant LLM Orchestration
To process, summarize, and classify articles against the standardized UPSC syllabus taxonomy without incurring massive API costs or hitting rate limits, the system utilizes an api_key_pool.py and llm_rotator.py. If Hugging Face rate-limits a request, the system automatically gracefully degrades to a local Ollama fallback, ensuring 100% ingestion uptime.

4. Strict Data Validation
Raw web data is inherently messy. The normalize_validate.py processor uses strict Pydantic schemas to validate LLM outputs, ensuring no hallucinated taxonomy tags or malformed JSON payloads ever reach the production database.

💻 Tech Stack
Backend Framework: Python, FastAPI

Asynchronous Workers: Celery, Celery Beat, Redis

Database & Search: PostgreSQL, SQLAlchemy, Alembic, pgvector

Object Storage: MinIO (S3-compatible)

AI / Machine Learning: Hugging Face, Ollama, Retrieval-Augmented Generation (RAG)

Frontend Client: FlutterFlow, Dart, Firebase Authentication

DevOps & Deployment: Docker, Docker Compose, AWS EC2

🚀 Local Development & Setup
The entire backend infrastructure is containerized. You can spin up the API, Postgres, Redis, MinIO, and Celery workers using Docker Compose.

Prerequisites
Docker & Docker Compose

Python 3.10+

Git

Installation
Clone the repository

Bash
git clone https://github.com/Nayab-Gauhar9/upsc_ai.git
cd upsc_ai
Environment Configuration
Create a .env file in the root directory and configure your variables (database credentials, API keys, MinIO secrets).

Bash
cp .env.example .env
Build and Run the Containers

Bash
docker-compose up --build -d
Run Database Migrations
Initialize the PostgreSQL database and pgvector extension.

Bash
docker-compose exec api alembic upgrade head
Access the Services

FastAPI Interactive Docs: http://localhost:8000/docs

MinIO Console: http://localhost:9001

Web App: politymentor.com

📂 Repository Structure
Plaintext
├── alembic.ini             # Alembic configuration
├── docker-compose.yml      # Multi-container orchestration
├── app.py                  # FastAPI application entry point
├── migrations/             # Alembic database migration scripts
│   └── versions/           # Schema history (incl. pgvector setup)
├── src/
│   ├── analyser/           # LLM evaluation and schema extraction
│   ├── celery/             # Task queues, scheduler, and API key rotator
│   ├── classifiers/        # Taxonomy mapping and HF/Ollama integration
│   ├── collectors/         # Automated web scraping (PIB, RSS)
│   ├── database/           # SQLAlchemy models and session management
│   ├── pipelines/          # End-to-end ETL and RAG generation workflows
│   ├── processors/         # Data cleaning and Pydantic validation
│   ├── rag/                # Vector embeddings and semantic retriever
│   └── storage/            # MinIO object storage client
└── tests/                  # Pytest suite for classification, MinIO, and RAG
📈 Future Scope / Roadmap
GraphRAG Integration: Transitioning from pure vector search to GraphRAG to map entity relationships between government ministries and schemes.

CI/CD Automation: Implementing GitHub Actions for automated pytest execution and zero-downtime deployment to AWS ECS.

Advanced Observability: Integrating OpenTelemetry for deeper tracing of Celery task execution times and LLM latency
