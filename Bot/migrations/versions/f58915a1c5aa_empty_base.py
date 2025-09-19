"""Empty base

Revision ID: f58915a1c5aa
Revises: 7a4932c9c9e7
Create Date: 2025-09-03 15:47:52.657221

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f58915a1c5aa'
down_revision: Union[str, Sequence[str], None] = '7a4932c9c9e7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
