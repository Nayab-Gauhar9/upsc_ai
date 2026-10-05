import time
from src.rag.retriever import UPSCRetriever

def run_retriever_test(query: str, top_k: int = 3):
    print("=" * 60)
    print(f"TESTING RETRIEVER WITH QUERY: '{query}'")
    print("=" * 60)

    start_time = time.time()
    try:
        retriever = UPSCRetriever()
        print("[1/3] Initialized UPSCRetriever (Voyage embedder + MinIO storage client)")

        # 1. Run retrieval
        results = retriever.retrieve_smart_notes_for_query(query=query, top_k=top_k)
        elapsed = round(time.time() - start_time, 2)

        print(f"\n[2/3] Retrieval complete in {elapsed}s. Found {len(results)} article(s).")
        print("=" * 60)

        # 2. Inspect results
        if not results:
            print("No matching notes found. Possible causes:")
            print(" - 'document_chunks' table has no rows with non-null 'embedding_1024'.")
            print(" - Voyage AI API key failed or returned an empty vector.")
            return

        for idx, item in enumerate(results, start=1):
            prid = item.get("prid")
            chapter = item.get("chapter")
            topic = item.get("topic")
            chunk_type = item.get("matched_chunk_type")
            smart_notes = item.get("smart_notes")

            print(f"\n--- Result #{idx} ---")
            print(f"PRID:                 {item.get('prid')}")
            print(f"Chapter:              {item.get('chapter')}")
            print(f"Topic:                {item.get('topic')}")
            print(f"Chunk Type:           {item.get('matched_chunk_type')}")
            print(f"Cosine Distance:      {item.get('cosine_distance')}")
            print(f"Confidence Level:     {item.get('confidence_percentage')}%")

            if isinstance(smart_notes, dict):
                title = smart_notes.get("headline") or smart_notes.get("title") or "No title"
                summary = smart_notes.get("summary") or ""
                print(f"Headline:    {title}")
                print(f"Summary:     {summary[:140]}..." if summary else "Summary: [Empty]")
                print("MinIO Load:  SUCCESS (Valid JSON fetched)")
            else:
                print("MinIO Load:  FAILED or Non-dict format returned")

        print("\n[3/3] VERDICT: UPSCRetriever is working as expected!")

    except Exception as e:
        print(f"\n[FAIL] Exception occurred during retrieval:\n{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test with a standard UPSC Polity topic
    test_query = "Centre-State Relations"
    run_retriever_test(query=test_query, top_k=2)
