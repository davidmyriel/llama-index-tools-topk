"""Turn the small filter dicts agents write into LlamaIndex ``MetadataFilters``."""

from __future__ import annotations

from typing import Any

from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters

_OPERATORS = {
    "$eq": FilterOperator.EQ,
    "$ne": FilterOperator.NE,
    "$gt": FilterOperator.GT,
    "$gte": FilterOperator.GTE,
    "$lt": FilterOperator.LT,
    "$lte": FilterOperator.LTE,
    "$in": FilterOperator.IN,
    "$nin": FilterOperator.NIN,
    "$contains": FilterOperator.CONTAINS,
}


def dict_to_filters(filters: dict[str, Any] | None) -> MetadataFilters | None:
    """
    Translate ``{"source": "runbook", "year": {"$gte": 2024}}`` into ``MetadataFilters``.

    Keys are metadata fields (dots for nested ones) and are ANDed together. A plain
    value means equality; a dict maps operators to values. Operators: ``$eq``,
    ``$ne``, ``$gt``, ``$gte``, ``$lt``, ``$lte``, ``$in``, ``$nin``, ``$contains``.
    """
    if not filters:
        return None
    if not isinstance(filters, dict):
        msg = f"filters must be an object like {{'source': 'runbook'}}, got {type(filters).__name__}"
        raise ValueError(msg)

    out = []
    for key, value in filters.items():
        if isinstance(value, dict):
            for op, operand in value.items():
                if op not in _OPERATORS:
                    msg = f"Unknown filter operator {op!r} on {key!r}; use one of {', '.join(_OPERATORS)}"
                    raise ValueError(msg)
                out.append(MetadataFilter(key=key, value=operand, operator=_OPERATORS[op]))
        else:
            out.append(MetadataFilter(key=key, value=value))
    return MetadataFilters(filters=out)
