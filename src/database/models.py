import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, UUID
from sqlalchemy.orm import Mapped, mapped_column

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
