"""add document chunks table"""

from alembic import op
import sqlalchemy as sa

# Revision identifiers, used by Alembic
revision = "123456789abc"        # <-- give a unique id (can be any random hex string)
down_revision = None             # <-- if this is your first migration, keep None
branch_labels = None
depends_on = None
