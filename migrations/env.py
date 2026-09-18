"""Alembic environment for the separate Phase 3 application database."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import MetaData

from consumer_finance_rag.config import load_settings
from consumer_finance_rag.lineage.database import (
    create_database_engine,
    require_database_url,
)

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Business-table metadata is intentionally absent from this foundation migration setup.
target_metadata = MetaData()


def run_migrations_offline() -> None:
    """Run migrations without opening a database connection."""

    settings = load_settings()
    context.configure(
        url=require_database_url(settings.database),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against the explicitly configured application database."""

    engine = create_database_engine(load_settings().database)
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
