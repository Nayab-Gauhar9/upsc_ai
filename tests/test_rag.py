from src.rag.retriever import UPSCRetriever
from src.pipelines.rag_generator import UPSCRAGGenerator

def run_test():
    # Define a target UPSC aspirant query
    query = "Centre-State Relations"
    
    print(f"[*] Testing Query: '{query}'\n")
    
    # 1. Initialize retriever and fetch full smart notes corresponding to top vector matches
    print("[*] Retrieving top matching chunks and fetching full notes from MinIO...")
    retriever = UPSCRetriever()
    retrieved_articles = retriever.retrieve_smart_notes_for_query(query, top_k=2)
    
    print(f"[+] Successfully fetched {len(retrieved_articles)} unique smart note records from MinIO.")
    for article in retrieved_articles:
        print(f"    - PRID: {article['prid']} | Chapter: {article['chapter']}")

    if not retrieved_articles:
        print("[!] No matching articles found. Make sure vector ingestion has been run.")
        return

    # 2. Initialize the expert generator and produce the structured UPSC response
    print("\n[*] Generating expert UPSC response via LLM...")
    generator = UPSCRAGGenerator(model_name="qwen3:8b") # Ensure model matches your local Ollama instance
    response = generator.generate_expert_response(query, retrieved_articles)

    print("\n" + "="*50)
    print("EXPERT UPSC RAG GENERATION OUTPUT:")
    print("="*50)
    print(response)
    print("="*50)

if __name__ == "__main__":
    run_test()
