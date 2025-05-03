# src/retriever.py
"""
This module provides the interface for performing search queries against
the document knowledge base using the VectorStore for embedding lookup and similarity search.
"""
import logging
from typing import List, Dict, Any, Optional, Union # Import Union for type hints
# Import the Chunk type if the search results might include the full Chunk object
# from src.chunker import Chunk # Uncomment if VectorStore.search returns Chunk objects

# Import config
from src.config import DEFAULT_TOP_K

# Import the VectorStore class - ensures the singleton can be accessed
from src.vector_store import VectorStore


# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- Custom Exception (Optional - rely on VectorStore's exceptions for a thin wrapper) ---
# class RetrievalError(Exception):
#     """Custom exception for errors during the retrieval/search process."""
#     pass

# --- Core Search Function ---

# Update return type hint based on actual return structure from VectorStore.search
# Assuming VectorStore.search returns a list of dicts like:
# [{'chunk': '...', 'score': 0.123, 'rank': 1, 'doc_id': '...'}, ...]
# If it returns the full Chunk object or similar:
# List[Dict[str, Union[str, float, int, Chunk]]] or similar complex type
# Let's stick to List[Dict[str, Any]] as it's flexible, but the docstring describes the content.
SearchResult = Dict[str, Any] # Define a type alias for clarity

def search_query(
    query: str,
    doc_ids: Optional[Union[str, List[str]]] = None, # Accept single doc_id string or list
    top_k: int = DEFAULT_TOP_K
) -> List[SearchResult]:
    """
    Performs a similarity search against the vector store using the query.
    The search can be scoped to specific documents.

    Args:
        query: The search query string.
        doc_ids: Optional. A single document ID string or a list of document ID strings
                 to limit the search. If None, searches across all loaded knowledge.
        top_k: The maximum number of top results to return from the vector store.

    Returns:
        A list of dictionaries, where each dictionary represents a search result
        and typically contains keys like 'chunk' (the text), 'score' (similarity score),
        'rank' (position in results), and 'doc_id' (the source document ID).
        Returns an empty list if no results are found or if the vector store is empty.

    Raises:
        Exception: If an error occurs during the search process within the VectorStore.
                   Specific exceptions from VectorStore are re-raised.
    """
    if not query or not query.strip():
         logger.warning("Search query is empty or whitespace only.")
         return [] # Return empty list for empty queries

    # Ensure doc_ids is always a list or None for consistency in VectorStore.search
    if isinstance(doc_ids, str):
        doc_ids_list: Optional[List[str]] = [doc_ids]
    elif isinstance(doc_ids, list) or doc_ids is None:
        doc_ids_list = doc_ids
    else:
         logger.error(f"Invalid type for doc_ids: {type(doc_ids)}. Expected str, list of str, or None.")
         # Depending on strictness, could raise a TypeError here
         doc_ids_list = None # Default to searching all if input is invalid


    logger.info(f"🔍 Starting search for: '{query[:100]}...' (top_k={top_k}, doc_ids={doc_ids_list})")

    try:
        # Access the VectorStore singleton instance
        vector_store_instance = VectorStore()

        # Perform the search using the VectorStore's method
        # Assume VectorStore().search handles query embedding internally
        # and searches the appropriate index(es).
        results: List[SearchResult] = vector_store_instance.search(
            query=query,
            top_k=top_k,
            doc_ids=doc_ids_list # Pass the potentially converted list
        )

        # The VectorStore's search method is responsible for formatting the results
        # into the desired dictionary structure including 'chunk', 'score', etc.

        logger.info(f"✅ Search complete. Found {len(results)} results.")
        # Log snippets of results (careful with sensitive data)
        for i, result in enumerate(results[:3]): # Log first 3 results snippets
            snippet = result.get('chunk', '')[:100].replace('\n', ' ') + '...'
            logger.debug(f"Result {i+1}: Score={result.get('score', float('inf')):.4f}, Doc ID={result.get('doc_id', 'N/A')}, Snippet='{snippet}'")

        return results

    except Exception as e:
        # Catch any exception raised by the VectorStore.search method
        logger.error(f"❌ Error during search_query for '{query[:100]}...': {e}", exc_info=True)
        # Re-raise the caught exception to propagate the error upstream
        raise


# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging for standalone execution
    # Note: In a real app, setup_logging should be called once at the app entry point (e.g., app.py)
    from src.config import setup_logging
    # Note: This test requires the VectorStore singleton to be initialized
    # AND have knowledge (embeddings and index) loaded for a specific doc_id.
    # You would typically do this at your application's startup.
    # The VectorStore itself might need ensure_required_dirs called earlier.

    setup_logging()
    logger.info("--- Testing src/retriever.py search_query function ---")

    # --- Setup for Test ---
    # !!! IMPORTANT !!!
    # For this test to work, you MUST have the VectorStore initialized elsewhere
    # (e.g., by running a script that sets it up and loads knowledge).
    # The VectorStore needs to have loaded:
    # 1. The SentenceTransformer model.
    # 2. Embeddings and FAISS index for the test_doc_id from KNOWLEDGE_BASE_DIR.
    # This is usually done by calling VectorStore().load_document_knowledge(test_doc_id)
    # or a similar initialization/loading function in your main application flow.

    # Replace with a doc_id for which you have loaded knowledge in your VectorStore setup
    test_doc_id_single = "your_single_test_doc_id" # <<< REPLACE
    test_doc_ids_multi = ["doc_id_a", "doc_id_b"] # <<< REPLACE with actual loaded doc IDs for multi-search test

    # Create a dummy VectorStore instance to simulate the singleton being available,
    # but it won't have loaded knowledge unless your test setup script does that.
    # This call ensures the class exists, but doesn't guarantee loaded data.
    # You NEED external setup for actual search results.
    try:
        logger.info("Attempting to access VectorStore singleton...")
        vs_instance = VectorStore()
        logger.info("VectorStore singleton accessed.")
        # In a real test setup, you would load knowledge here:
        # vs_instance.load_document_knowledge(test_doc_id_single)
        # vs_instance.load_document_knowledge(test_doc_ids_multi[0])
        # vs_instance.load_document_knowledge(test_doc_ids_multi[1])
        logger.warning("Knowledge loading for test NOT automatically performed by this script.")
        logger.warning("Ensure VectorStore has loaded knowledge for the test doc_id(s) externally.")

    except Exception as e:
        logger.error(f"Failed to access or setup VectorStore for test: {e}", exc_info=True)
        logger.error("Cannot run retriever test without a working VectorStore setup.")
        test_doc_id_single = None # Disable test if setup fails
        test_doc_ids_multi = None

    # --- Run Search Tests ---
    if test_doc_id_single and test_doc_id_single != "your_single_test_doc_id":
        logger.info(f"\n--- Running search test for single doc_id: {test_doc_id_single} ---")
        test_query_single = "What is the company policy on vacation?" # Replace with a query relevant to your test doc

        try:
            logger.info(f"Searching single document '{test_doc_id_single}' for query: '{test_query_single}'")
            single_doc_results = search_query(test_query_single, doc_ids=test_doc_id_single, top_k=3)

            logger.info(f"Search returned {len(single_doc_results)} results for doc_id {test_doc_id_single}.")
            for i, r in enumerate(single_doc_results):
                snippet = r.get('chunk', '')[:150].replace('\n', ' ') + '...'
                logger.info(f"Result {i+1} (Rank {r.get('rank', '?')}): Score {r.get('score', float('inf')):.4f}, Doc ID {r.get('doc_id', 'N/A')} - '{snippet}'")

        except Exception as e:
            logger.error(f"Single document search test failed: {e}", exc_info=True)

    else:
         logger.warning("\nSkipping single document search test. Check test_doc_id_single value and VectorStore setup.")


    if test_doc_ids_multi and all(did != "doc_id_a" and did != "doc_id_b" for did in test_doc_ids_multi): # Basic check if they were replaced
        logger.info(f"\n--- Running search test for multiple doc_ids: {test_doc_ids_multi} ---")
        test_query_multi = "Tell me about benefits." # Replace with a query relevant to your docs

        try:
            logger.info(f"Searching multiple documents {test_doc_ids_multi} for query: '{test_query_multi}'")
            multi_doc_results = search_query(test_query_multi, doc_ids=test_doc_ids_multi, top_k=5) # Higher top_k for multi-doc

            logger.info(f"Search returned {len(multi_doc_results)} results for doc_ids {test_doc_ids_multi}.")
            for i, r in enumerate(multi_doc_results):
                 snippet = r.get('chunk', '')[:150].replace('\n', ' ') + '...'
                 logger.info(f"Result {i+1} (Rank {r.get('rank', '?')}): Score {r.get('score', float('inf')):.4f}, Doc ID {r.get('doc_id', 'N/A')} - '{snippet}'")

        except Exception as e:
            logger.error(f"Multi-document search test failed: {e}", exc_info=True)

    else:
         logger.warning("\nSkipping multi-document search test. Check test_doc_ids_multi value and VectorStore setup.")


    logger.info("\n--- End of retriever.py test ---")