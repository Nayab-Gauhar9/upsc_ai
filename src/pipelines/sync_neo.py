import os
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

LOCAL_DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/upsc_ai")
NEON_DB_URL = os.getenv("NEON_DATABASE_URL")

if not NEON_DB_URL:
    raise ValueError("Missing NEON_DATABASE_URL in .env file!")

for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://"):
    if LOCAL_DB_URL.startswith(prefix):
        LOCAL_DB_URL = LOCAL_DB_URL.replace(prefix, "postgresql://")
    if NEON_DB_URL.startswith(prefix):
        NEON_DB_URL = NEON_DB_URL.replace(prefix, "postgresql://")

local_engine = create_engine(LOCAL_DB_URL)
neon_engine = create_engine(NEON_DB_URL)

TABLE_CONFIG = [
    # Core User & Tracking Models
    {"name": "users", "pk": "id", "seq": "users_id_seq", "upsert": False},
    {"name": "user_sessions", "pk": "id", "seq": "user_sessions_id_seq", "upsert": False},
    {"name": "ingestion_runs", "pk": "run_id", "seq": None, "upsert": False},
    {"name": "document_chunks", "pk": "id", "seq": "document_chunks_id_seq", "upsert": False},
    {"name": "user_chapter_progress", "pk": "id", "seq": "user_chapter_progress_id_seq", "upsert": False},
    {"name": "user_notes", "pk": "id", "seq": "user_notes_id_seq", "upsert": False},
    {"name": "user_query_history", "pk": "id", "seq": "user_query_history_id_seq", "upsert": False},
    
    # Knowledge Graph Models
    {"name": "issue_dossiers", "pk": "issue_id", "seq": None, "upsert": True},
    {"name": "issue_graph_edges", "pk": "edge_id", "seq": None, "upsert": False},
    {
        "name": "constitutional_articles",
        "pk": "id",
        "seq": "constitutional_articles_id_seq",
        "upsert": True,
    },
    {
        "name": "prelims_pyqs",
        "pk": "id",
        "seq": "prelims_pyqs_id_seq",
        "upsert": True,
    },

    # Daily Drill Sets & Questions (Parent table first, then child)
    {
        "name": "daily_challenges",
        "pk": "id",
        "seq": None,
        "upsert": True,
    },
    {
        "name": "daily_challenge_questions",
        "pk": "id",
        "seq": None,
        "upsert": True,
    },

    # Issue Reporting & Editorial Feedback
    {
        "name": "question_reports",
        "pk": "id",
        "seq": None,
        "upsert": True,
    },
]


def serialize_row_values(row_dict: dict) -> dict:
    """Convert dicts/JSON structures to strings for JSONB, but keep native lists for Postgres ARRAY columns."""
    cleaned = {}
    for key, value in row_dict.items():
        if key in ("mapped_articles", "legal_anchors"):
            cleaned[key] = value if value is not None else []
        elif isinstance(value, (dict, list)):
            cleaned[key] = json.dumps(value)
        else:
            cleaned[key] = value
    return cleaned


def sync_table(
    table_name: str,
    pk_column: str,
    seq_name: str = None,
    upsert: bool = False,
    chunk_size: int = 500,
):
    print(f"\n[{table_name}] Checking for updates/new rows...")

    with local_engine.connect() as local_conn, neon_engine.connect() as neon_conn:
        local_rows = local_conn.execute(text(f"SELECT * FROM {table_name}")).mappings().all()
        total_local = len(local_rows)
        if not local_rows:
            print(f"[{table_name}] Local table is empty. Skipped 0 rows.")
            return

        neon_ids_result = neon_conn.execute(text(f'SELECT "{pk_column}" FROM {table_name}'))
        neon_existing_ids = {row[0] for row in neon_ids_result.fetchall()}

        # Classify rows as new vs existing
        new_rows = [row for row in local_rows if row[pk_column] not in neon_existing_ids]
        existing_rows = [row for row in local_rows if row[pk_column] in neon_existing_ids]

        if upsert:
            target_rows = [serialize_row_values(dict(row)) for row in local_rows]

            to_insert = len(new_rows)
            to_update = len(existing_rows)
            skipped = 0
            print(f"[{table_name}] Total: {total_local} | Moving (New): {to_insert} | Updating: {to_update} | Skipped: {skipped}")
        else:
            target_rows = [serialize_row_values(dict(row)) for row in new_rows]
            to_insert = len(new_rows)
            to_update = 0
            skipped = len(existing_rows)
            print(f"[{table_name}] Total: {total_local} | Moving (New): {to_insert} | Updating: 0 | Skipped (Already Exist): {skipped}")

        if not target_rows:
            print(f"[{table_name}] All {skipped} rows are already present in Neon. None moved.")
            return

        keys = list(target_rows[0].keys())
        cols_str = ", ".join(f'"{k}"' for k in keys)
        vals_str = ", ".join(f":{k}" for k in keys)

        if upsert:
            update_cols = [k for k in keys if k != pk_column]
            updates_str = ", ".join(f'"{k}" = EXCLUDED."{k}"' for k in update_cols)
            sql_clause = f'ON CONFLICT ("{pk_column}") DO UPDATE SET {updates_str}'
        else:
            sql_clause = f'ON CONFLICT ("{pk_column}") DO NOTHING'

        insert_sql = text(f"INSERT INTO {table_name} ({cols_str}) VALUES ({vals_str}) {sql_clause}")

        for i in range(0, len(target_rows), chunk_size):
            batch = target_rows[i : i + chunk_size]
            neon_conn.execute(insert_sql, batch)
            neon_conn.commit()

        if upsert:
            print(f"[{table_name}] Completed: {to_insert} inserted, {to_update} updated.")
        else:
            print(f"[{table_name}] Completed: {to_insert} moved into Neon, {skipped} skipped.")

        if seq_name:
            neon_conn.execute(
                text(
                    f"SELECT setval('{seq_name}', COALESCE((SELECT MAX({pk_column}) FROM {table_name}), 1), true);"
                )
            )
            neon_conn.commit()
            print(f"[{table_name}] Sequence '{seq_name}' updated.")


def sync_all():
    print("=" * 60)
    print("STARTING LOCAL -> NEON FULL SYNC")
    print("=" * 60)

    for cfg in TABLE_CONFIG:
        sync_table(
            table_name=cfg["name"],
            pk_column=cfg["pk"],
            seq_name=cfg["seq"],
            upsert=cfg.get("upsert", False),
        )

    print("\n" + "=" * 60)
    print("ALL TABLES FULLY SYNCHRONIZED TO NEON")
    print("=" * 60)


if __name__ == "__main__":
    sync_all()
