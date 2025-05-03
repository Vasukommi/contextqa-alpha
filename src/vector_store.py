# src/vector_store.py
"""
This module implements a singleton VectorStore class for managing and searching
document embeddings and text chunks using SentenceTransformer and FAISS.
"""
import os
import json
import numpy as np
import faiss # FAISS is needed for index objects and read/write operations
import logging
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Tuple, Optional, Union # Import Union for type hints

# Import the Chunk type definition
from src.chunker import Chunk # Import the Chunk TypedDict

# Import paths and configuration from the config file
from src.config import (
    KNOWLEDGE_BASE_DIR,
    PROCESSED_DATA_DIR,
    SENTENCE_TRANSFORMER_MODEL,
    DEFAULT_TOP_K
)

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- Custom Exception ---
class VectorStoreError(Exception):
    """Custom exception for errors related to the VectorStore operations."""
    pass

# --- VectorStore Class (Singleton) ---

class VectorStore:
    """
    Manages loading, storing, and searching document embeddings and chunks.
    Loads the SentenceTransformer model once. Implements a singleton pattern.
    Stores loaded knowledge (embeddings, chunks, FAISS index) per document ID.
    """
    _instance: Optional['VectorStore'] = None # Class variable to hold the singleton instance
    _initialized: bool = False # Class variable to track initialization status

    def __new__(cls, *args, **kwargs):
        """Implements the singleton pattern."""
        if cls._instance is None:
            logger.debug("Creating the first instance of VectorStore.")
            cls._instance = super(VectorStore, cls).__new__(cls)
            # Don't set _initialized here, let __init__ handle it after potential super().__new__ issues
        else:
             logger.debug("Returning existing VectorStore instance.")
        return cls._instance


    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL):
        """
        Initializes the VectorStore singleton, loading the SentenceTransformer model.
        This initializer is called only once for the singleton instance.

        Args:
            model_name: The name of the SentenceTransformer model to load. Defaults
                        to SENTENCE_TRANSFORMER_MODEL from config.

        Raises:
            VectorStoreError: If the SentenceTransformer model fails to load.
        """
        # Use the class-level flag to prevent re-initialization
        if self._initialized:
            logger.debug("VectorStore already initialized, skipping __init__.")
            return

        logger.info("🧠 Initializing VectorStore (loading SentenceTransformer model)...")
        try:
            # Load the SentenceTransformer model once
            self.model: SentenceTransformer = SentenceTransformer(model_name)
            # Check if model loading was successful (e.g., ensure encode method exists)
            if not hasattr(self.model, 'encode') or not callable(self.model.encode):
                 raise AttributeError("Loaded object is not a valid SentenceTransformer model.")

            logger.info(f"✅ SentenceTransformer model '{model_name}' loaded successfully.")
        except Exception as e:
            # Use logger.critical as model loading failure is critical
            logger.critical(f"❌ Failed to load SentenceTransformer model '{model_name}': {e}", exc_info=True)
            # Wrap in custom exception for consistent error type
            raise VectorStoreError(f"Failed to load SentenceTransformer model '{model_name}': {e}") from e

        # Dictionary to hold loaded knowledge for different documents {doc_id: {embeddings, chunks, index}}
        # Store List[Chunk] directly
        self._knowledge: Dict[str, Dict[str, Union[np.ndarray, List[Chunk], faiss.Index]]] = {}

        # Mark as initialized *after* successful loading
        self._initialized = True
        logger.info("VectorStore initialization complete.")


    def load_document_knowledge(self, doc_id: str):
        """
        Loads the embeddings, chunks, and FAISS index for a specific document into memory.
        Knowledge for a doc_id is loaded only once per VectorStore instance.

        Args:
            doc_id: The ID of the document whose knowledge should be loaded.

        Raises:
            VectorStoreError: If necessary knowledge files are missing, corrupt,
                              or inconsistencies are found (e.g., dimension mismatch).
        """
        if doc_id in self._knowledge:
            logger.info(f"Knowledge for document '{doc_id}' is already loaded.")
            return

        logger.info(f"⚙️ Loading knowledge for document: {doc_id}...")

        emb_path = os.path.join(KNOWLEDGE_BASE_DIR, f"{doc_id}_embeddings.npy")
        chunk_path = os.path.join(PROCESSED_DATA_DIR, f"{doc_id}_chunks.json")
        index_path = os.path.join(KNOWLEDGE_BASE_DIR, f"{doc_id}_index.faiss")

        # --- Load Embeddings ---
        logger.debug(f"Attempting to load embeddings from: {emb_path}")
        if not os.path.exists(emb_path):
            logger.error(f"❌ Embeddings file not found: {emb_path}")
            # Wrap in custom exception
            raise VectorStoreError(f"Embeddings file not found for document {doc_id}: {emb_path}")
        try:
            embeddings: np.ndarray = np.load(emb_path)
            logger.info(f"✅ Loaded embeddings from: {emb_path} (Shape: {embeddings.shape})")
        except IOError as e:
            logger.error(f"❌ Error loading embeddings from {emb_path}: {e}", exc_info=True)
            # Wrap in custom exception
            raise VectorStoreError(f"Error loading embeddings for {doc_id}: {e}") from e
        except Exception as e:
             logger.error(f"❌ Unexpected error loading embeddings for {doc_id} from {emb_path}: {e}", exc_info=True)
             # Wrap in custom exception
             raise VectorStoreError(f"Unexpected error loading embeddings for {doc_id}: {e}") from e


        # --- Load Chunks ---
        # Now expecting List[Chunk] from the improved chunker.py
        logger.debug(f"Attempting to load chunks from: {chunk_path}")
        if not os.path.exists(chunk_path):
            # This is also a critical file, should exist if embeddings were created
            logger.error(f"❌ Chunks file not found: {chunk_path}")
            # Wrap in custom exception
            raise VectorStoreError(f"Chunks file not found for document {doc_id}: {chunk_path}")
        try:
            with open(chunk_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            # Expecting a dictionary with a "chunks" key containing the list of Chunk objects
            chunks: List[Chunk] = chunks_data.get("chunks", []) # Safely get the list

            if not isinstance(chunks, list):
                 logger.error(f"Data loaded from {chunk_path} under 'chunks' key is not a list.")
                 raise VectorStoreError(f"Invalid format in chunk file for {doc_id}: 'chunks' is not a list.")

            # Optional: Basic validation that items in chunks are dict-like and have 'text'
            if chunks and (not isinstance(chunks[0], dict) or 'text' not in chunks[0]):
                 logger.warning(f"Items in 'chunks' list in {chunk_path} do not seem like Chunk objects.")
                 # Decide if this should be a fatal error. For now, log warning.

            if not chunks:
                 logger.warning(f"Chunks file {chunk_path} is empty or 'chunks' list is empty.")
                 # If no chunks but embeddings exist, this is inconsistent.
                 # Decide if to raise error or proceed with empty chunks.
                 # Proceeding might lead to index search on empty index. Let's proceed but log warning.

            logger.info(f"✅ Loaded {len(chunks)} chunks from: {chunk_path}")
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"❌ Error loading chunks from {chunk_path}: {e}", exc_info=True)
            # Wrap in custom exception
            raise VectorStoreError(f"Error loading chunks for {doc_id}: {e}") from e
        except Exception as e:
             logger.error(f"❌ Unexpected error loading chunks for {doc_id} from {chunk_path}: {e}", exc_info=True)
             # Wrap in custom exception
             raise VectorStoreError(f"Unexpected error loading chunks for {doc_id}: {e}") from e


        # --- Load FAISS Index ---
        logger.debug(f"Attempting to load FAISS index from: {index_path}")
        if not os.path.exists(index_path):
            # This is also a critical file, should exist if embeddings were created
            logger.error(f"❌ FAISS index file not found: {index_path}")
            # Wrap in custom exception
            raise VectorStoreError(f"FAISS index file not found for document {doc_id}")
        try:
            index: faiss.Index = faiss.read_index(index_path)
            # Verify index total vectors matches number of chunks with text (robust check)
            # Need to count valid text chunks first
            valid_text_chunks_count = sum(1 for chunk in chunks if isinstance(chunk, dict) and chunk.get('text', '').strip())

            if index.ntotal != valid_text_chunks_count:
                 logger.error(f"❌ FAISS index total vectors ({index.ntotal}) does not match number of valid text chunks ({valid_text_chunks_count}) for {doc_id}.")
                 # This indicates a severe inconsistency in the processed data.
                 raise VectorStoreError(f"FAISS index/chunk count mismatch for {doc_id}. Index vectors: {index.ntotal}, Valid chunks: {valid_text_chunks_count}")

            # Verify index dimension matches embeddings dimension (critical check)
            if index.d != embeddings.shape[1]:
                 logger.error(f"❌ FAISS index dimension ({index.d}) does not match embeddings dimension ({embeddings.shape[1]}) for {doc_id}.")
                 # This is a fatal error, embeddings cannot be used with this index.
                 raise VectorStoreError(f"FAISS index dimension mismatch for {doc_id}. Index dim: {index.d}, Embeddings dim: {embeddings.shape[1]}")

            # Optional: Verify index total vectors matches embeddings count (redundant if dim matches)
            # if index.ntotal != embeddings.shape[0]:
            #      logger.warning(f"⚠️ FAISS index total vectors ({index.ntotal}) does not match embeddings count ({embeddings.shape[0]}) for {doc_id}.")
            #      # Log warning, might proceed if dimensions match

            logger.info(f"✅ Loaded FAISS index from: {index_path} (Dimension: {index.d}, Vectors: {index.ntotal})")
        except Exception as e: # FAISS read_index might raise various errors
            logger.error(f"❌ Error loading FAISS index from {index_path}: {e}", exc_info=True)
            # Wrap in custom exception
            raise VectorStoreError(f"Error loading FAISS index for {doc_id}: {e}") from e

        # Store the loaded knowledge in the instance dictionary
        self._knowledge[doc_id] = {
            # Storing embeddings in memory. For very large datasets, consider on-demand loading or IndexIDMap2.
            "embeddings": embeddings,
            "chunks": chunks, # Store the list of Chunk objects
            "index": index
        }
        logger.info(f"✅ Knowledge for document '{doc_id}' loaded successfully.")

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Performs a similarity search across one or more documents' loaded knowledge.

        Args:
            query: The search query string.
            top_k: The maximum number of top results to return across all documents. Defaults
                   to DEFAULT_TOP_K from config.
            doc_ids: Optional list of document IDs to search within. If None, searches
                     all documents currently loaded in the VectorStore instance.

        Returns:
            A list of dictionaries, where each dictionary represents a search result.
            Each result dictionary includes keys such as 'chunk' (the text), 'score'
            (similarity score - lower is better for L2), 'rank' (overall rank),
            and metadata from the original chunk ('doc_id', 'chunk_id', 'chunk_index').
            Returns an empty list if no documents are loaded, no doc_ids are specified
            that are loaded, or no search results are found.

        Raises:
            VectorStoreError: If an error occurs during the query embedding or FAISS search process.
        """
        logger.info(f"Starting search for query: '{query[:min(len(query), 100)]}...' (top_k={top_k})")
        if not self._knowledge:
            logger.warning("⚠️ No documents loaded in VectorStore. Cannot perform search.")
            return []

        if not query or not query.strip():
             logger.warning("Search query is empty or whitespace only.")
             return [] # Return empty list for empty queries


        if doc_ids is None:
            # Search all currently loaded documents
            search_doc_ids = list(self._knowledge.keys())
            logger.debug("No specific doc_ids provided for search. Searching all loaded documents.")
        else:
            # Filter requested doc_ids to only include those that are loaded
            search_doc_ids = [did for did in doc_ids if did in self._knowledge]
            if not search_doc_ids:
                 logger.warning(f"None of the specified doc_ids ({doc_ids}) are currently loaded in VectorStore. Cannot perform search.")
                 return [] # Return empty list if none of the requested docs are loaded
            logger.debug(f"Searching specified loaded documents: {search_doc_ids}")


        all_results: List[Dict[str, Any]] = []

        try:
            # Embed the query once
            logger.debug("Embedding search query...")
            # Ensure model is loaded before using it
            if not hasattr(self, 'model') or self.model is None:
                 raise VectorStoreError("SentenceTransformer model is not loaded.")
            query_vector = self.model.encode([query], convert_to_numpy=True)
            query_vector = np.array(query_vector).astype("float32")
            logger.debug("Query embedding complete.")
        except Exception as e:
             logger.error(f"❌ Error embedding search query: {e}", exc_info=True)
             # Wrap in custom exception
             raise VectorStoreError(f"Error embedding search query: {e}") from e


        # Perform search on each specified loaded document's index
        for doc_id in search_doc_ids:
            # We already checked if doc_id is in _knowledge
            knowledge = self._knowledge[doc_id]
            index = knowledge["index"]
            chunks = knowledge["chunks"] # This is List[Chunk]

            # Ensure index has vectors to search and chunks are available and aligned
            if index.ntotal == 0 or not chunks or index.ntotal != len(chunks):
                # This state should ideally be prevented by load_document_knowledge checks,
                # but adding a check here provides robustness during search.
                logger.warning(f"⚠️ Skipping search for doc '{doc_id}' due to missing index vectors ({index.ntotal}), chunks ({len(chunks) if chunks else 0}), or misalignment.")
                continue

            # We will collect up to top_k results *per document* temporarily,
            # then combine and re-rank globally.
            # A more advanced approach might use a single merged index or FAISS's IndexShards.
            doc_top_k = min(top_k, index.ntotal) # Don't ask for more results than there are vectors

            try:
                # Search the individual FAISS index
                distances, indices = index.search(query_vector, doc_top_k)

                # Process results from this document
                for i in range(doc_top_k):
                    idx_in_faiss = indices[0][i] # Index within the FAISS index
                    score = float(distances[0][i])

                    # Ensure the index is valid for the chunks list
                    if idx_in_faiss >= 0 and idx_in_faiss < len(chunks):
                        original_chunk: Chunk = chunks[idx_in_faiss] # Get the original Chunk object

                        # Append result including metadata from the Chunk object
                        result_dict: Dict[str, Any] = {
                            "chunk": original_chunk.get("text", ""), # Get text content
                            "score": score,
                            "rank": 0,  # Placeholder, will be updated after global sorting
                            # Include other metadata from the Chunk object
                            "doc_id": original_chunk.get("doc_id", doc_id), # Prefer doc_id from chunk if present, else use loop var
                            "chunk_id": original_chunk.get("chunk_id", f"{doc_id}-{idx_in_faiss}"), # Get chunk_id from chunk if present
                            "chunk_index": original_chunk.get("chunk_index", idx_in_faiss), # Get index from chunk if present
                            # Add any other metadata you stored in the Chunk object
                            # "original_source": original_chunk.get("original_source"),
                            # "page_number": original_chunk.get("page_number"),
                        }
                        all_results.append(result_dict)
                    else:
                        logger.warning(f"⚠️ Invalid index {idx_in_faiss} returned by FAISS search for doc '{doc_id}'. Skipping result.")

            except Exception as e:
                # Catch errors during FAISS search for a single document
                logger.error(f"❌ Error during FAISS search for document '{doc_id}': {e}", exc_info=True)
                # Decide if to continue with other documents or stop.
                # Let's log and continue for robustness in multi-doc search.

        # Sort combined results across all documents by score (L2 distance, lower is better)
        if all_results:
            all_results.sort(key=lambda x: x.get("score", float('inf'))) # Use .get for safety

            # Apply global rank and trim to overall top_k
            final_results = all_results[:top_k]
            for i, result in enumerate(final_results):
                result["rank"] = i + 1 # Assign global rank

            logger.info(f"✅ Search complete. Returning top {len(final_results)} results.")
            # Log snippets of final results (careful with sensitive data)
            for i, result in enumerate(final_results):
                snippet = result.get('chunk', '')[:100].replace('\n', ' ') + '...'
                logger.debug(f"Final Result {i+1} (Rank {result.get('rank', '?')}): Score {result.get('score', float('inf')):.4f}, Doc ID {result.get('doc_id', 'N/A')}, Snippet='{snippet}'")

            return final_results
        else:
            logger.info("✅ Search complete. No results found.")
            return [] # Return empty list if no results were collected


# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging for standalone execution
    from src.config import setup_logging, ensure_required_dirs
    # Note: This test requires actual knowledge files to exist in the knowledge/ and data/processed directories
    # for the specified test_doc_id. You would typically run the full processing/embedding workflow first.

    setup_logging() # Configure logging
    # ensure_required_dirs() # Uncomment if needed for creating directories if running this script stand-alone first time

    logger.info("--- Testing src/vector_store.py functionality ---")

    # --- Setup for Test ---
    # !!! IMPORTANT !!!
    # For this test to work, you MUST have run the full processing/embedding pipeline
    # for at least one document, resulting in:
    # 1. data/processed/<test_doc_id>_chunks.json (containing List[Chunk])
    # 2. knowledge/<test_doc_id>_embeddings.npy
    # 3. knowledge/<test_doc_id>_index.faiss
    # This script does NOT create these files. It assumes they exist.

    # Replace with a doc_id for which you have these files
    test_doc_id_single = "your_test_doc_id_here" # <<< REPLACE with an actual doc_id

    # Optional: Set up another doc_id and its files for multi-search test
    test_doc_id_multi_1 = "doc_id_multi_a" # <<< REPLACE if testing multi-doc search
    test_doc_id_multi_2 = "doc_id_multi_b" # <<< REPLACE if testing multi-doc search


    if test_doc_id_single == "your_test_doc_id_here":
         logger.warning("Replace 'your_test_doc_id_here' with an actual doc_id that has corresponding files in data/processed and knowledge/.")
         logger.warning("Skipping VectorStore test.")
         # Disable tests if main doc_id is not set up
         test_doc_id_single = None
         test_doc_id_multi_1 = None
         test_doc_id_multi_2 = None


    if test_doc_id_single:
        try:
            logger.info("--- Running VectorStore tests ---")
            # Get the singleton instance (initializes model on first call)
            logger.info("Attempting to get VectorStore instance (loads model)...")
            vs = VectorStore()
            logger.info("VectorStore instance obtained.")

            # Load knowledge for the test document(s)
            logger.info(f"\nAttempting to load knowledge for '{test_doc_id_single}'...")
            vs.load_document_knowledge(test_doc_id_single)
            logger.info(f"Knowledge for '{test_doc_id_single}' loaded.")

            # Load knowledge for multi-doc test if configured
            loaded_multi_docs = []
            if test_doc_id_multi_1 and test_doc_id_multi_1 != "doc_id_multi_a":
                 try:
                     logger.info(f"\nAttempting to load knowledge for '{test_doc_id_multi_1}'...")
                     vs.load_document_knowledge(test_doc_id_multi_1)
                     logger.info(f"Knowledge for '{test_doc_id_multi_1}' loaded.")
                     loaded_multi_docs.append(test_doc_id_multi_1)
                 except Exception as e:
                     logger.error(f"Failed to load knowledge for multi-doc test {test_doc_id_multi_1}: {e}", exc_info=True)

            if test_doc_id_multi_2 and test_doc_id_multi_2 != "doc_id_multi_b":
                 try:
                     logger.info(f"\nAttempting to load knowledge for '{test_doc_id_multi_2}'...")
                     vs.load_document_knowledge(test_doc_id_multi_2)
                     logger.info(f"Knowledge for '{test_doc_id_multi_2}' loaded.")
                     loaded_multi_docs.append(test_doc_id_multi_2)
                 except Exception as e:
                     logger.error(f"Failed to load knowledge for multi-doc test {test_doc_id_multi_2}: {e}", exc_info=True)


            # Perform searches
            search_query_text_single = "What is the purpose?" # Replace with a relevant query for your main test doc
            logger.info(f"\nPerforming search within single document '{test_doc_id_single}': '{search_query_text_single}'")
            single_doc_results = vs.search(query=search_query_text_single, doc_ids=[test_doc_id_single], top_k=3)

            logger.info(f"Search returned {len(single_doc_results)} results for doc_id {test_doc_id_single}.")
            for i, r in enumerate(single_doc_results):
                snippet = r.get('chunk', '')[:150].replace('\n', ' ') + '...'
                logger.info(f"Result {i+1} (Rank {r.get('rank', '?')}): Score {r.get('score', float('inf')):.4f}, Doc ID {r.get('doc_id', 'N/A')}, Chunk ID {r.get('chunk_id', 'N/A')}, Chunk Index {r.get('chunk_index', 'N/A')} - '{snippet}'")


            # Multi-document search test
            if loaded_multi_docs:
                 search_query_text_multi = "Tell me about benefits or holidays." # Replace with a query relevant to your multi test docs
                 logger.info(f"\nPerforming search across multiple documents {loaded_multi_docs}: '{search_query_text_multi}'")
                 multi_doc_results = vs.search(query=search_query_text_multi, doc_ids=loaded_multi_docs, top_k=5)

                 logger.info(f"Search returned {len(multi_doc_results)} results across docs {loaded_multi_docs}.")
                 for i, r in enumerate(multi_doc_results):
                     snippet = r.get('chunk', '')[:150].replace('\n', ' ') + '...'
                     logger.info(f"Result {i+1} (Rank {r.get('rank', '?')}): Score {r.get('score', float('inf')):.4f}, Doc ID {r.get('doc_id', 'N/A')}, Chunk ID {r.get('chunk_id', 'N/A')}, Chunk Index {r.get('chunk_index', 'N/A')} - '{snippet}'")
            else:
                 logger.warning("\nSkipping multi-document search test as no multi-doc test files were set up or loaded.")


            # Example of attempting to load knowledge for a non-existent document (expecting VectorStoreError)
            logger.info("\nAttempting to load knowledge for a non-existent document (expecting VectorStoreError)...")
            try:
                vs.load_document_knowledge("non_existent_doc_id_12345")
            except VectorStoreError as e:
                logger.info(f"Caught expected error: {e}")
            except Exception as e:
                logger.error(f"Caught unexpected error: {e}", exc_info=True)

            # Example of searching for a document whose knowledge is NOT loaded (expecting it to be skipped)
            logger.info("\nAttempting to search for an unloaded document (expecting it to be skipped)...")
            skipped_doc_id = "another_unloaded_doc"
            skipped_results = vs.search(query="some query", doc_ids=[skipped_doc_id], top_k=1)
            logger.info(f"Search for unloaded doc '{skipped_doc_id}' returned {len(skipped_results)} results (should be 0).")


        except VectorStoreError as e:
            logger.error(f"A VectorStore error occurred during testing: {e}")
        except Exception as e:
             logger.error(f"An unexpected error occurred during VectorStore test: {e}", exc_info=True)

    logger.info("\n--- End of vector_store.py test ---")