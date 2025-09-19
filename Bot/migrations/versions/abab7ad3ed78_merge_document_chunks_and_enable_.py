"""Merge document_chunks and enable_pgvector

Revision ID: abab7ad3ed78
Revises: enable_pgvector, 701381d0a641
Create Date: 2025-09-03 17:21:54.425370

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'abab7ad3ed78'
down_revision: Union[str, Sequence[str], None] = ('enable_pgvector', '701381d0a641')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
