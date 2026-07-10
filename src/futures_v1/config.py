"""Local configuration helpers for the V1 migration project."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILES = (
    PROJECT_ROOT / ".env",
    PROJECT_ROOT.parent / ".env",
)

try:
    from dotenv import load_dotenv

    for env_file in ENV_FILES:
        if env_file.exists():
            load_dotenv(env_file)
            break
except ImportError:
    pass


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "dbname": os.getenv("DB_NAME", "quant_db"),
    "user": os.getenv("DB_USER", "quant"),
    "password": os.getenv("DB_PASSWORD", "quant_password"),
}


def get_db_url() -> str:
    return (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    )
