import json
from src.storage.minio_client import MinIOStorage

def test_single_latest_note():
    print("Initializing MinIOStorage client...")
    storage = MinIOStorage()
    
    # Test retrieving only 1 latest note
    print("Fetching top 1 latest note from storage...")
    notes = storage.list_latest_smart_notes(limit=1)
    
    if not notes:
        print("❌ No notes found in bucket.")
        return

    latest = notes[0]
    
    print("\n✅ Successfully retrieved single latest note:")
    print("--------------------------------------------------")
    print(f"PRID:        {latest.get('prid')}")
    print(f"Chapter:     {latest.get('chapter')}")
    print(f"Topic:       {latest.get('topic')}")
    print(f"Headline:    {latest.get('headline')}")
    print(f"Object Key:  {latest.get('object_key')}")
    print("--------------------------------------------------")
    
    # Check smart_notes dictionary
    smart_notes = latest.get("smart_notes", {})
    if isinstance(smart_notes, dict):
        keys = list(smart_notes.keys())
        print(f"Payload Keys: {keys}")
    else:
        print("smart_notes payload is empty or not a dict.")

if __name__ == "__main__":
    test_single_latest_note()
