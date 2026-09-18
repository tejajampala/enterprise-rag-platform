"""Lazy SQLAlchemy construction for the optional Phase 3 application database."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from consumer_finance_rag.config import DatabaseSettings


class DatabaseConfigurationError(RuntimeError):
    """Raised when Phase 3 database functionality is used without configuration."""


def require_database_url(settings: DatabaseSettings) -> str:
    """Return the configured URL or fail only when database functionality is requested."""

    if settings.url is None or not settings.url.strip():
        raise DatabaseConfigurationError(
            "The Phase 3 application database is not configured. "
            "Set RAG__DATABASE__URL to a PostgreSQL SQLAlchemy URL."
        )
    return settings.url


def create_database_engine(settings: DatabaseSettings) -> Engine:
    """Create a lazy SQLAlchemy engine without opening a database connection."""

    return create_engine(require_database_url(settings), pool_pre_ping=True)


def create_session_factory(settings: DatabaseSettings) -> sessionmaker[Session]:
    """Create a typed session factory bound to the configured lazy engine."""

    engine = create_database_engine(settings)
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
