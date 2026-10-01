"""add_embedding_1024_to_document_chunks.

Revision ID: f8b643596668
Revises: e602a0d7314f
Create Date: 2026-10-02 03:16:15.318119

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = 'f8b643596668'
down_revision: Union[str, Sequence[str], None] = 'e602a0d7314f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Safely adds the 1024-dimension column without touching existing rows
    op.add_column(
        'document_chunks',
        sa.Column('embedding_1024', Vector(dim=1024), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('document_chunks', 'embedding_1024')
