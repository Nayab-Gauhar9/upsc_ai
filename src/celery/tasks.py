from src.celery.app import app

# Direct imports from your pipeline modules
from src.pipelines.pib_ingestion import run_pib_ingestion
from src.pipelines.pib_classification import PIBClassificationPipeline
from src.pipelines.analyser import StudySynthesisPipeline
from src.pipelines.vector_ingestion import VectorIngestionPipeline


@app.task(name="tasks.batch_ingest", bind=True, max_retries=3, default_retry_delay=60)
def batch_ingest(self):
    """
    Step 1: Runs every 2 hours at :00.
    Fetches RSS feed, normalizes, validates, stores raw objects in MinIO, 
    and logs run stats to PostgreSQL IngestionRun table.
    """
    print("\n" + "=" * 60)
    print("[CELERY TASK: INGESTION] Starting scheduled PIB RSS run...")
    print("=" * 60)
    try:
        report = run_pib_ingestion()
        print(f"[CELERY TASK: INGESTION] Succeeded. Summary: {report}\n")
        return report
    except Exception as exc:
        print(f"[CELERY TASK: INGESTION ERROR] Pipeline crashed: {exc}\n")
        raise self.retry(exc=exc)


@app.task(name="tasks.batch_classify", bind=True, max_retries=3, default_retry_delay=60)
def batch_classify(self):
    """
    Step 2: Runs at :05 every 2 hours (5 mins after ingestion).
    Scans pib/raw/, skips already-classified articles, classifies new ones 
    with Groq LLM, and materializes relevant articles to pib/classified/.
    """
    print("\n" + "=" * 60)
    print("[CELERY TASK: CLASSIFICATION] Starting scheduled PIB classification...")
    print("=" * 60)
    try:
        pipeline = PIBClassificationPipeline()
        report = pipeline.process_all(
            limit=None,
            chunk_size=15,
            delay_between_chunks_sec=60,
            delay_between_calls_sec=30,
        )
        print(f"[CELERY TASK: CLASSIFICATION] Succeeded. Summary: {report}\n")
        return report
    except Exception as exc:
        print(f"[CELERY TASK: CLASSIFICATION ERROR] Pipeline crashed: {exc}\n")
        raise self.retry(exc=exc)


@app.task(name="tasks.batch_analyse", bind=True, max_retries=3, default_retry_delay=60)
def batch_analyse(self):
    """
    Step 3: Runs at :20 every 2 hours (15 mins after classification).
    Scans pib/classified/, skips existing smart notes, invokes LLM to generate 
    dossiers (Prelims MCQs, Mains practice, linkages), and writes to pib/smart_notes/.
    """
    print("\n" + "=" * 60)
    print("[CELERY TASK: ANALYSER] Starting scheduled smart note synthesis...")
    print("=" * 60)
    try:
        pipeline = StudySynthesisPipeline()
        report = pipeline.process_all_relevant(limit=None, delay_between_calls=20.0)
        print(f"[CELERY TASK: ANALYSER] Succeeded. Summary: {report}\n")
        return report
    except Exception as exc:
        print(f"[CELERY TASK: ANALYSER ERROR] Pipeline crashed: {exc}\n")
        raise self.retry(exc=exc)


@app.task(name="tasks.batch_vector_ingest", bind=True, max_retries=2, default_retry_delay=60)
def batch_vector_ingest(self):
    """
    Step 4: Runs at :35 every 2 hours (15 mins after analyser).
    Reads new smart notes from MinIO, chunks them, generates embeddings via 
    local model, and stores embeddings into PostgreSQL pgvector DocumentChunkModel table.
    """
    print("\n" + "=" * 60)
    print("[CELERY TASK: VECTOR INGESTION] Starting scheduled pgvector pipeline...")
    print("=" * 60)
    try:
        pipeline = VectorIngestionPipeline()
        pipeline.process_all_smart_notes(limit=None)
        print("[CELERY TASK: VECTOR INGESTION] Succeeded.\n")
        return "Vector Ingestion Complete"
    except Exception as exc:
        print(f"[CELERY TASK: VECTOR INGESTION ERROR] Pipeline crashed: {exc}\n")
        raise self.retry(exc=exc)
