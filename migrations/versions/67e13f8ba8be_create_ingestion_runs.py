"""create ingestion runs

Revision ID: 67e13f8ba8be
Revises: 
Create Date: 2026-09-19 03:30:07.909536

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '67e13f8ba8be'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "ingestion_runs",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_items", sa.Integer(), nullable=False),
        sa.Column("collected", sa.Integer(), nullable=False),
        sa.Column("valid", sa.Integer(), nullable=False),
        sa.Column("uploaded", sa.Integer(), nullable=False),
        sa.Column("already_exists", sa.Integer(), nullable=False),
        sa.Column("failed", sa.Integer(), nullable=False),
        sa.Column("collection_failures", sa.Integer(), nullable=False),
        sa.Column("validation_failures", sa.Integer(), nullable=False),
        sa.Column("storage_failures", sa.Integer(), nullable=False),
        sa.Column("unexpected_failures", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("run_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("ingestion_runs")
