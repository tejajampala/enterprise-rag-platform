from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
import sqlalchemy

from consumer_finance_rag.config import DatabaseSettings

MODULE_NAME = "consumer_finance_rag.lineage.database"


def test_importing_database_module_does_not_create_an_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_engine_creation(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("database module created an engine during import")

    sys.modules.pop(MODULE_NAME, None)
    with monkeypatch.context() as context:
        context.setattr(sqlalchemy, "create_engine", unexpected_engine_creation)
        module = importlib.import_module(MODULE_NAME)
        assert hasattr(module, "create_database_engine")

    # Restore a normally imported module for subsequent tests regardless of execution order.
    sys.modules.pop(MODULE_NAME, None)
    importlib.import_module(MODULE_NAME)


def test_database_operations_fail_clearly_without_configured_url() -> None:
    from consumer_finance_rag.lineage.database import (
        DatabaseConfigurationError,
        create_database_engine,
        create_session_factory,
    )

    settings = DatabaseSettings()
    with pytest.raises(DatabaseConfigurationError, match="RAG__DATABASE__URL"):
        create_database_engine(settings)
    with pytest.raises(DatabaseConfigurationError, match="RAG__DATABASE__URL"):
        create_session_factory(settings)


def test_engine_and_session_factory_are_lazy_without_a_postgres_server() -> None:
    from consumer_finance_rag.lineage.database import (
        create_database_engine,
        create_session_factory,
    )

    settings = DatabaseSettings(
        url="postgresql+psycopg://lineage:secret@127.0.0.1:1/unreachable"
    )
    engine = create_database_engine(settings)
    session_factory = create_session_factory(settings)

    assert engine.url.drivername == "postgresql+psycopg"
    assert session_factory.kw["bind"].url.drivername == "postgresql+psycopg"
    engine.dispose()
    session_factory.kw["bind"].dispose()
