"""create chat_sessions, chat_history, query_audit_log

Revision ID: 0001_initial_postgres
Revises: —
Create Date: 2026-06-10
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial_postgres"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(128), unique=True, nullable=False, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("NOW()")),
        sa.Column("last_active", sa.DateTime(timezone=True),
                  server_default=sa.text("NOW()")),
        sa.Column("status", sa.String(32), server_default="active"),
        sa.Column("file_names", sa.Text(), server_default=""),
        sa.Column("chunk_size", sa.Integer(), server_default="400"),
        sa.Column("chunk_overlap", sa.Integer(), server_default="50"),
        sa.Column("k", sa.Integer(), server_default="3"),
    )

    op.create_table(
        "chat_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(128), nullable=False, index=True),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("NOW()")),
    )

    op.create_table(
        "query_audit_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(128), nullable=False, index=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text()),
        sa.Column("latency_ms", sa.Float()),
        sa.Column("prompt_tokens", sa.Integer()),
        sa.Column("completion_tokens", sa.Integer()),
        sa.Column("k_used", sa.Integer()),
        sa.Column("pii_redacted", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("NOW()")),
    )


def downgrade() -> None:
    op.drop_table("query_audit_log")
    op.drop_table("chat_history")
    op.drop_table("chat_sessions")
