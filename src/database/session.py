import os
from typing import Generator
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured")

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=300,  # Recycles connections every 5 mins to stay ahead of Neon idle drops
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that guarantees rollbacks on exceptions and closes sessions."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()  # Resets the connection state if a query or route fails
        raise
    finally:
        db.close()
