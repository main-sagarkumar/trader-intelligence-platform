"""
Create the database engine for analytics persistence.

This module reads connection settings from environment variables and exposes a
SQLAlchemy engine used by data loading and query helpers.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

# Build the PostgreSQL URL from environment variables kept outside source code.
DATABASE_URL = (
    f"postgresql://"
    f"{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

engine = create_engine(DATABASE_URL)
