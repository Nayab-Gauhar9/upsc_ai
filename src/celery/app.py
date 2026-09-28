import os
from celery import Celery
from celery.schedules import crontab

# Connect to your existing Redis 7 instance on WSL port 6380
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6380/0")

app = Celery("polity_tasks", broker=REDIS_URL, backend=REDIS_URL, include=['src.celery.tasks'],)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    # Route tasks to distinct queues
    task_routes={
        "tasks.batch_ingest": {"queue": "ingestion_queue"},
        "tasks.batch_classify": {"queue": "classification_queue"},
        "tasks.batch_analyse": {"queue": "analysis_queue"},
        "tasks.batch_vector_ingest": {"queue": "vector_queue"},
    },
    beat_schedule={
        "scheduled-pib-ingest": {
            "task": "tasks.batch_ingest",
            "schedule": crontab(minute=0, hour="*/2"),
        },
        "scheduled-pib-classification": {
            "task": "tasks.batch_classify",
            "schedule": crontab(minute=5, hour="*/2"),
        },
        "scheduled-pib-analyser": {
            "task": "tasks.batch_analyse",
            "schedule": crontab(minute=20, hour="*/2"),
        },
        "scheduled-pib-vector-ingest": {
            "task": "tasks.batch_vector_ingest",
            "schedule": crontab(minute=35, hour="*/2"),
        },
    },
)
