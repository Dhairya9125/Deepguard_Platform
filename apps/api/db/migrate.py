"""
DeepGuard Platform — Database Migration Helper
Module  : apps.api.db.migrate

Run this script ONCE after setting your Neon DATABASE_URL in .env to create
all tables (User, AnalysisJob, ForensicResult) on your Neon database.

Usage:
    python -m apps.api.db.migrate

This is a one-shot replacement for Alembic migrations during development.
In production, switch to Alembic for proper schema versioning.
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


async def main() -> None:
    from apps.api.db.database import init_db, engine
    from apps.api.core.config import settings

    print("=" * 60)
    print("DeepGuard Platform — Database Table Creation")
    print("=" * 60)
    print(f"Target:  {settings.DATABASE_URL[:50]}...")
    print(f"SSL:     {settings.DB_SSL_REQUIRED}")
    print()

    print("Creating tables...")
    try:
        await init_db()
        print("[SUCCESS] Tables created successfully!")
        print()
        print("Tables:")
        print("  - users")
        print("  - analysis_jobs")
        print("  - forensic_results")
        print()
        print("You can now start the API server:")
        print("  uvicorn apps.api.main:app --reload --port 8000")
    except Exception as exc:
        print(f"[ERROR] Error creating tables: {exc}")

        print()
        print("Troubleshooting:")
        print("  1. Make sure DATABASE_URL is set in your .env file")
        print("  2. Verify the Neon project is active (check console.neon.tech)")
        print("  3. Ensure DB_SSL_REQUIRED=True for Neon")
        raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
