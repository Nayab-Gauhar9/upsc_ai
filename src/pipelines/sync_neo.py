import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

LOCAL_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/upsc_ai"
)
NEON_DB_URL = os.getenv("NEON_DATABASE_URL")

if not NEON_DB_URL:
    raise ValueError("Missing NEON_DATABASE_URL in .env file!")

local_engine = create_engine(LOCAL_DB_URL)
neon_engine = create_engine(NEON_DB_URL)

# Defined in strict Foreign Key dependency order:
# Parents first, then dependent child tables.
TABLE_CONFIG = [
    {"name": "users", "pk": "id", "seq": "users_id_seq"},
    {"name": "ingestion_runs", "pk": "run_id", "seq": None},
    {"name": "document_chunks", "pk": "id", "seq": "document_chunks_id_seq"},
    {"name": "user_chapter_progress", "pk": "id", "seq": "user_chapter_progress_id_seq"},
    {"name": "user_notes", "pk": "id", "seq": "user_notes_id_seq"},
    {"name": "user_query_history", "pk": "id", "seq": "user_query_history_id_seq"},
]


def sync_table(table_name: str, pk_column: str, seq_name: str = None, chunk_size: int = 500):
    print(f"\n[{table_name}] Checking for new rows...")
    
    with local_engine.connect() as local_conn, neon_engine.connect() as neon_conn:
        # 1. Fetch all existing Primary Keys from Neon
        neon_ids_result = neon_conn.execute(text(f"SELECT {pk_column} FROM {table_name}"))
        neon_existing_ids = {row[0] for row in neon_ids_result.fetchall()}

        # 2. Fetch all local rows
        local_rows = local_conn.execute(text(f"SELECT * FROM {table_name}")).mappings().all()

        # 3. Filter for rows that don't exist in Neon
        new_rows = [dict(row) for row in local_rows if row[pk_column] not in neon_existing_ids]

        if not new_rows:
            print(f"[{table_name}] Already synchronized (0 new rows).")
            return

        print(f"[{table_name}] Found {len(new_rows)} new row(s) to push.")

        # 4. Insert in chunks using ON CONFLICT DO NOTHING
        keys = list(new_rows[0].keys())
        cols_str = ", ".join(f'"{k}"' for k in keys)
        vals_str = ", ".join(f":{k}" for k in keys)

        insert_sql = text(
            f'INSERT INTO {table_name} ({cols_str}) '
            f'VALUES ({vals_str}) '
            f'ON CONFLICT ("{pk_column}") DO NOTHING'
        )

        for i in range(0, len(new_rows), chunk_size):
            batch = new_rows[i : i + chunk_size]
            neon_conn.execute(insert_sql, batch)
            neon_conn.commit()

        print(f"[{table_name}] Pushed {len(new_rows)} rows to Neon.")

        # 5. Synchronize the PostgreSQL auto-increment sequence
        if seq_name:
            neon_conn.execute(text(
                f"SELECT setval('{seq_name}', COALESCE((SELECT MAX({pk_column}) FROM {table_name}), 1), true);"
            ))
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
            seq_name=cfg["seq"]
        )

    print("\n" + "=" * 60)
    print("ALL TABLES FULLY SYNCHRONIZED TO NEON")
    print("=" * 60)


if __name__ == "__main__":
    sync_all()
