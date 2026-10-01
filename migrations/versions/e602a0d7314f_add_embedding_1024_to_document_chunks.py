"""add_embedding_1024_to_document_chunks

Revision ID: e602a0d7314f
Revises: de91f64293f5
Create Date: 2026-10-02 03:14:14.648132

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e602a0d7314f'
down_revision: Union[str, Sequence[str], None] = 'de91f64293f5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
