"""
Check database connectivity for local setup validation.

This helper opens a SQLAlchemy connection and prints the database version so
developers can verify environment variables and network access.
"""

from sqlalchemy import text

from src.database.db import engine

with engine.connect() as conn:

    # A lightweight query confirms that the configured engine can connect.
    result = conn.execute(
        text("SELECT version();")
    )

    print(result.fetchone())
