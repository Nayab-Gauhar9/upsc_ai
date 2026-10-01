
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, UUID, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
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

    embedding_1024 = mapped_column(
            Vector(1024),  
            nullable=True,
        )

    embedding_model: Mapped[str] = mapped_column(
        String(100),
        default="qwen3-embedding",
    )

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    firebase_uid: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        index=True,
        nullable=False,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    display_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    queries: Mapped[list["UserQueryHistory"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )

    notes: Mapped[list["UserNote"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class UserQueryHistory(Base):
    __tablename__ = "user_query_history"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    model_answer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    user: Mapped["User"] = relationship(
        back_populates="queries",
    )

class UserChapterProgress(Base):
    __tablename__ = "user_chapter_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    chapter_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chapter_title: Mapped[str] = mapped_column(String(255), nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship to user
    user = relationship("User", backref="completed_chapters")

    # Prevent duplicate records for the same chapter per user
    table_args = (
        UniqueConstraint("user_id", "chapter_number", name="uq_user_chapter"),
    )

class UserNote(Base):
    __tablename__ = "user_notes"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        index=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        default="Untitled Note",
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        default="",
        nullable=False,
    )

    category: Mapped[str] = mapped_column(
        String(100),
        default="Polity GS-II",
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    user: Mapped["User"] = relationship(
        back_populates="notes",
    )
