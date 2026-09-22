
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, UUID
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from src.database.base import Base


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    total_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    collected: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    valid: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    uploaded: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    already_exists: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    failed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    collection_failures: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    validation_failures: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    storage_failures: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    unexpected_failures: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )


class DocumentChunkModel(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    prid: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    chapter_name: Mapped[str] = mapped_column(
        String(250),
        nullable=False,
    )

    topic: Mapped[str] = mapped_column(
        String(250),
        nullable=False,
    )

    chunk_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    chunk_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    embedding = mapped_column(
        Vector(384),  # Dimensions for sentence-transformers all-MiniLM-L6-v2
        nullable=True,
    )

    embedding_model: Mapped[str] = mapped_column(
        String(100),
        default="all-MiniLM-L6-v2",
    )
