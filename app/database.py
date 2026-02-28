"""
Database layer stub.

Current implementation: in-memory log via models.py.
To add SQLite persistence, implement get_db() here and
swap models.py for SQLAlchemy ORM models.

This file exists to satisfy the project structure spec
and provide a clean upgrade path.
"""


def get_db():
    """Placeholder. Replace with SQLAlchemy session factory if persistence needed."""
    raise NotImplementedError("Database layer not configured. Using in-memory log.")
