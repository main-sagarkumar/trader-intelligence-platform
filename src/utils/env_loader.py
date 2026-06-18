"""
Load environment variables for local project execution.

This module centralizes dotenv loading so scripts can import it before reading
configuration from the process environment.
"""

from dotenv import load_dotenv
load_dotenv()
