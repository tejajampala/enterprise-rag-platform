"""Strict, Arrow-safe dataframe boundary for Streamlit presentation tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from enum import StrEnum
from numbers import Integral, Real
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]


class DisplayColumnKind(StrEnum):
    STRING = "string"
    NULLABLE_INTEGER = "nullable_integer"
    NULLABLE_FLOAT = "nullable_float"
    NULLABLE_BOOLEAN = "nullable_boolean"


class DisplayDataFrameError(ValueError):
    """Raised before rendering when a presentation column violates its type contract."""


def arrow_safe_dataframe(
    rows: Sequence[Mapping[str, object]],
    *,
    schema: Mapping[str, DisplayColumnKind] | None = None,
) -> Any:
    """Build a typed dataframe and verify the same PyArrow boundary Streamlit uses.

    Columns not listed in ``schema`` retain pandas inference, but mixed text/numeric values
    are rejected explicitly instead of relying on Streamlit's lossy fallback conversion.
    """

    frame = pd.DataFrame(list(rows))
    declared = schema or {}
    missing = set(declared) - set(frame.columns)
    if missing and rows:
        raise DisplayDataFrameError(f"Declared display columns are missing: {sorted(missing)}")
    for column, kind in declared.items():
        if column not in frame.columns:
            frame[column] = _empty_series(kind)
            continue
        values = frame[column].tolist()
        _validate_values(column, values, kind)
        frame[column] = _typed_series(values, kind)
    for column in frame.columns:
        if column not in declared:
            _reject_mixed_text_and_numeric(column, frame[column].tolist())
    try:
        pa.Table.from_pandas(frame, preserve_index=False)
    except (pa.ArrowException, TypeError, ValueError) as error:
        raise DisplayDataFrameError(
            f"Dataframe column contract is not Arrow serializable: {error}"
        ) from error
    return frame


def render_arrow_dataframe(
    streamlit: Any,
    rows: Sequence[Mapping[str, object]],
    *,
    schema: Mapping[str, DisplayColumnKind] | None = None,
    column_config: Mapping[str, object] | None = None,
) -> None:
    """Validate and render one dataframe through the shared Streamlit boundary."""

    streamlit.dataframe(
        arrow_safe_dataframe(rows, schema=schema),
        hide_index=True,
        width="stretch",
        column_config=column_config,
    )


def format_optional_count(value: int | None) -> str:
    """Render a display-only count without treating missing as zero."""

    return "Unavailable" if value is None else f"{value:,}"


def format_optional_float(value: float | None, *, digits: int = 3) -> str:
    return "Unavailable" if value is None else f"{value:,.{digits}f}"


def _validate_values(
    column: str,
    values: Sequence[object],
    kind: DisplayColumnKind,
) -> None:
    for value in values:
        if _is_missing(value):
            continue
        valid = (
            isinstance(value, str)
            if kind is DisplayColumnKind.STRING
            else isinstance(value, Integral) and not isinstance(value, bool)
            if kind is DisplayColumnKind.NULLABLE_INTEGER
            else isinstance(value, Real | Decimal) and not isinstance(value, bool)
            if kind is DisplayColumnKind.NULLABLE_FLOAT
            else isinstance(value, bool)
        )
        if not valid:
            raise DisplayDataFrameError(
                f"Column {column!r} expected {kind.value}, got {type(value).__name__}"
            )


def _typed_series(values: Sequence[object], kind: DisplayColumnKind) -> Any:
    if kind is DisplayColumnKind.STRING:
        return pd.Series(pd.array(values, dtype="string"))
    if kind is DisplayColumnKind.NULLABLE_INTEGER:
        return pd.Series(pd.array(values, dtype="Int64"))
    if kind is DisplayColumnKind.NULLABLE_FLOAT:
        normalized = [
            None if _is_missing(value) else float(cast(Real | Decimal, value))
            for value in values
        ]
        return pd.Series(pd.array(normalized, dtype="Float64"))
    return pd.Series(pd.array(values, dtype="boolean"))


def _empty_series(kind: DisplayColumnKind) -> Any:
    return _typed_series([], kind)


def _reject_mixed_text_and_numeric(column: object, values: Sequence[object]) -> None:
    non_null = [value for value in values if not _is_missing(value)]
    has_text = any(isinstance(value, str) for value in non_null)
    has_numeric = any(
        isinstance(value, Real | Decimal) and not isinstance(value, bool)
        for value in non_null
    )
    if has_text and has_numeric:
        raise DisplayDataFrameError(
            f"Column {column!r} mixes presentation text and numeric values"
        )


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    result = pd.isna(value)
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False
