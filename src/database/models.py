import uuid
from datetime import datetime, date
from sqlalchemy import Boolean, Index
from sqlalchemy import DateTime, Integer, String, Text, UUID, ForeignKey, UniqueConstraint, func, Date
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from src.database.base import Base
from typing import Optional

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

    user = relationship("User", backref="completed_chapters")

    __table_args__ = (
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


class IssueDossierModel(Base):
    """Represents a developing constitutional/governance issue anchor (Topic State Node)."""
    __tablename__ = "issue_dossiers"

    issue_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    issue_slug: Mapped[str] = mapped_column(
        String(120),
        unique=True,
        index=True,
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    chapter_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        default="active",
        nullable=False,
    )

    canonical_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Mains dialectics, structural dimensions, way forward
    mains_framework: Mapped[dict | None] = mapped_column(
        JSONB,
        default=dict,
        nullable=True,
    )

    # NEW: Prelims Master Factsheet
    # Structure:
    # {
    #     "constitutional_articles": ["Article 324(2)", "Article 326"],
    #     "statutory_acts_sections": ["Representation of the People Act 1951, Sec 62(5)"],
    #     "nodal_body": "Election Commission of India / Ministry of Law and Justice",
    #     "legal_status": "Constitutional Body (Art 324)",
    #     "tenure_or_eligibility": "6 years or 65 years of age",
    #     "appointment_removal": "Appointed by President; Removed on grounds of proved misbehaviour like SC judge",
    #     "key_committees": ["Dinesh Goswami Committee (1990)", "Law Commission 255th Report"]
    # }
    prelims_factsheet: Mapped[dict | None] = mapped_column(
        JSONB,
        default=dict,
        nullable=True,
    )

    article_count: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    first_detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    last_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # 1-to-many relationship with graph edges
    edges: Mapped[list["IssueGraphEdgeModel"]] = relationship(
        back_populates="dossier",
        cascade="all, delete-orphan",
        order_by="IssueGraphEdgeModel.event_date.asc()",
    )


class IssueGraphEdgeModel(Base):
    """Directed edge linking an ingested article, viewpoint stance, and legal anchors to an issue."""
    __tablename__ = "issue_graph_edges"

    edge_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    issue_slug: Mapped[str] = mapped_column(
        String(120),
        ForeignKey("issue_dossiers.issue_slug", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    prid: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    publication: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    event_date: Mapped[date] = mapped_column(
        Date,
        default=date.today,
        nullable=False,
        index=True,
    )

    perspective_stance: Mapped[str] = mapped_column(
        String(64),
        nullable=False,  # e.g., 'JUDICIAL_PRIMACY', 'EXECUTIVE_ACCOUNTABILITY'
    )

    core_claim: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    legal_anchors: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        default=list,
        nullable=False,
    )

    # NEW: Granular Prelims Evolutionary Facts for this specific event/article
    # Structure:
    # {
    #     "provisions_invoked": ["Section 123(3) RPA", "Article 142"],
    #     "bench_or_authority": "5-Judge Constitution Bench headed by CJI",
    #     "numerical_thresholds": "Majority of 4:1",
    #     "prelims_statements": [
    #         {"statement": "Voting is a statutory right, not a fundamental right.", "is_correct": true, "reason": "Held in PUCL (2003)"},
    #         {"statement": "Section 62(5) disenfranchises undertrials entirely.", "is_correct": true, "reason": "Explicit bar in 1951 Act"}
    #     ]
    # }
    prelims_event_facts: Mapped[dict | None] = mapped_column(
        JSONB,
        default=dict,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    dossier: Mapped["IssueDossierModel"] = relationship(
        back_populates="edges",
    )

class ConstitutionalArticleModel(Base):
    """Represents a master constitutional provision with multi-epoch evolution and Prelims/Mains anchors."""
    __tablename__ = "constitutional_articles"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    article_number: Mapped[str] = mapped_column(
        String(32),
        unique=True,
        index=True,
        nullable=False,  # e.g., 'Article 356'
    )

    article_slug: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,  # e.g., 'article-356'
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,  # e.g., 'Provisions in case of failure of constitutional machinery in States'
    )

    part_number: Mapped[str] = mapped_column(
        String(32),
        index=True,
        nullable=False,  # e.g., 'Part XVIII'
    )

    doctrine_cluster_id: Mapped[str | None] = mapped_column(
        String(64),
        index=True,
        nullable=True,  # e.g., 'emergency-powers', 'pardoning-powers'
    )

    verbatim_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    bare_act_analysis: Mapped[list[dict]] = mapped_column(
        JSONB,
        default=list,
        server_default="[]",
        nullable=False,
    )
    # Aspirant summary, core mandate, threshold conditions, prelims traps
    # Structure:
    # {
    #     "core_mandate": "...",
    #     "threshold_conditions": "...",
    #     "prelims_traps": ["...", "..."]
    # }
    aspirant_summary: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    # 4-Epoch Evolutionary Trajectory
    # Structure:
    # {
    #     "colonial_genesis": {"source_statute": "...", "historical_context": "..."},
    #     "cad_crucible": {"draft_article": "...", "key_speakers": [...], "pivotal_quote": "..."},
    #     "judicial_and_amendments": [{"milestone": "...", "impact": "..."}],
    #     "contemporary_frictions": ["..."]
    # }
    evolution_trajectory: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    # Optional side-by-side comparison matrix (e.g., Art 356 vs Art 352 or Art 72 vs Art 161)
    # Structure:
    # {
    #     "paired_article": "Article 352",
    #     "rows": [{"parameter": "...", "primary": "...", "secondary": "..."}]
    # }
    comparative_matrix: Mapped[dict | None] = mapped_column(
        JSONB,
        default=None,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

class PrelimsPYQModel(Base):
    """Represents historical UPSC GS Prelims questions mapped to Laxmikanth taxonomy and Constitutional Articles."""
    __tablename__ = "prelims_pyqs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    year: Mapped[int] = mapped_column(
        Integer,
        index=True,
        nullable=False,
    )

    question_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    question_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Options structure: {"A": "...", "B": "...", "C": "...", "D": "..."}
    options: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )

    correct_option: Mapped[str] = mapped_column(
        String(2),
        nullable=False,  # e.g., 'A', 'B', 'C', 'D'
    )

    # Taxonomy Mapping (Laxmikanth 8th Edition: 1 to 92)
    chapter_number: Mapped[int] = mapped_column(
        Integer,
        index=True,
        nullable=False,
    )

    chapter_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    # Constitutional Article Mapping (array of integer article numbers 1-395)
    # Allows fast retrieval for: WHERE 13 = ANY(mapped_articles)
    mapped_articles: Mapped[list[int]] = mapped_column(
        ARRAY(Integer),
        default=list,
        server_default="{}",
        nullable=False,
    )

    explanation: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    difficulty_level: Mapped[str | None] = mapped_column(
        String(20),
        default="Moderate",
        nullable=True,  # "Easy", "Moderate", "Tough"
    )

    is_conceptual: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("year", "question_number", name="uq_pyq_year_qnum"),
        Index("idx_pyq_mapped_articles", "mapped_articles", postgresql_using="gin"),
    )

class DailyChallengeModel(Base):
    """Represents a daily static Prelims drill set consisting of 15 curated MCQs."""
    __tablename__ = "daily_challenges"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    challenge_date: Mapped[date] = mapped_column(
        Date,
        unique=True,
        index=True,
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        default="Daily Static Polity Prelims Drill",
        nullable=False,
    )

    total_questions: Mapped[int] = mapped_column(
        Integer,
        default=15,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # 1-to-many relationship with questions ordered sequentially
    questions: Mapped[list["DailyChallengeQuestionModel"]] = relationship(
        back_populates="challenge",
        cascade="all, delete-orphan",
        order_by="DailyChallengeQuestionModel.question_order.asc()",
    )


class DailyChallengeQuestionModel(Base):
    """Represents an individual MCQ within a Daily Challenge set."""
    __tablename__ = "daily_challenge_questions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    challenge_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("daily_challenges.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    question_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,  # 1 to 15
    )

    archetype: Mapped[str] = mapped_column(
        String(64),
        nullable=False,  # 'Pairing', 'Conceptual & Theory', 'Institutional Mechanics', etc.
    )

    theme: Mapped[str] = mapped_column(
        String(128),
        nullable=False,  # e.g., 'Emergency Provisions', 'Money Bills'
    )

    primary_article_ref: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,  # e.g., 'Article 356'
    )

    question_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Statements list for pairing or multi-statement format: ["1. ...", "2. ..."]
    statements: Mapped[list[str] | None] = mapped_column(
        JSONB,
        default=list,
        nullable=True,
    )

    # Options structure: {"A": "...", "B": "...", "C": "...", "D": "..."}
    options: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )

    correct_option: Mapped[str] = mapped_column(
        String(2),
        nullable=False,  # 'A', 'B', 'C', 'D'
    )

    explanation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    fingerprint: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)

    traps_analyzed: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    challenge: Mapped["DailyChallengeModel"] = relationship(
        back_populates="questions",
    )

    __table_args__ = (
        UniqueConstraint("challenge_id", "question_order", name="uq_challenge_question_order"),
    )

class QuestionReport(Base):
    __tablename__ = "question_reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    drill_date: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    question_index: Mapped[int] = mapped_column(Integer, nullable=False)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    issue_type: Mapped[str] = mapped_column(String(100), nullable=False)
    user_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="PENDING")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow, 
        nullable=False
    )