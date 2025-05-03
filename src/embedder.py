# src/embedder.py
"""
This module handles the process of generating and saving embeddings for text chunks,
and building a FAISS index for efficient similarity search.
"""
import os
import json
import numpy as np
import faiss
import logging
from typing import List, Tuple, Optional

# Import SentenceTransformer (likely needed for type hinting or direct use if not via VectorStore)
from sentence_transformers import SentenceTransformer
# Import the Chunk type definition
from src.chunker import Chunk # Import the Chunk TypedDict

# Import paths and config
from src.config import (
    PROCESSED_DATA_DIR,
    KNOWLEDGE_BASE_DIR,
    SENTENCE_TRANSFORMER_MODEL # Import model name for reference/config
)
# Import the VectorStore to access its model instance (singleton pattern)
# Ensure VectorStore handles its own initialization and model loading
from src.vector_store import VectorStore


# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- Custom Exception ---
class EmbeddingError(Exception):
    """Custom exception for errors during the embedding process."""
    pass

# --- Core Functions ---

# Updated function to load List[Chunk]
def load_chunks_for_embedding(doc_id: str) -> List[Chunk]:
    """
    Loads chunks (with metadata) from the processed data directory for embedding.

    Args:
        doc_id: The ID of the document whose chunks should be loaded.

    Returns:
        A list of Chunk dictionaries.

    Raises:
        EmbeddingError: If the chunk file is missing, corrupt, or cannot be read.
    """
    chunk_path = os.path.join(PROCESSED_DATA_DIR, f"{doc_id}_chunks.json")
    logger.info(f"Attempting to load chunks for embedding from: {chunk_path}")

    if not os.path.exists(chunk_path):
        logger.error(f"Chunk file not found for document {doc_id}: {chunk_path}")
        # Wrap in custom exception
        raise EmbeddingError(f"Chunk file not found for document {doc_id}: {chunk_path}")

    try:
        with open(chunk_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Expecting the structure saved by the improved chunker.py
        chunks_data: List[Chunk] = data.get("chunks", []) # Safely get the list of chunks

        if not isinstance(chunks_data, list):
             logger.error(f"Data loaded from {chunk_path} is not a list under 'chunks' key.")
             raise EmbeddingError(f"Invalid format in chunk file for {doc_id}: 'chunks' is not a list.")

        if not chunks_data:
             logger.warning(f"Chunks file {chunk_path} is empty or 'chunks' list is empty.")
             return [] # Return empty list if no chunks found


        # Optional: Add basic validation that items in chunks_data are dict-like
        if chunks_data and not isinstance(chunks_data[0], dict):
             logger.error(f"Items in 'chunks' list in {chunk_path} are not dictionaries.")
             raise EmbeddingError(f"Invalid format in chunk file for {doc_id}: Items in 'chunks' are not dictionaries.")

        logger.info(f"Loaded {len(chunks_data)} chunks for embedding from {chunk_path}.")
        return chunks_data

    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error loading or parsing chunks from {chunk_path} for embedding: {e}", exc_info=True) # Add exc_info
        # Wrap in custom exception
        raise EmbeddingError(f"Error loading/parsing chunk file for {doc_id}: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred loading chunks for embedding {doc_id}: {e}", exc_info=True) # Add exc_info
        # Wrap in custom exception
        raise EmbeddingError(f"Unexpected error loading chunk file for {doc_id}: {e}") from e


# Accept the model instance and List[Chunk] as input
def generate_embeddings(chunks: List[Chunk], model: SentenceTransformer) -> Tuple[np.ndarray, List[Chunk]]:
    """
    Generates embeddings for chunks using a provided SentenceTransformer model instance.
    Returns the embeddings and the original chunks (useful to keep aligned).

    Args:
        chunks: A list of Chunk dictionaries.
        model: An initialized SentenceTransformer model instance.

    Returns:
        A tuple containing:
        - A numpy array of embeddings.
        - The original list of Chunk dictionaries (filtered if any chunks were skipped).

    Raises:
        EmbeddingError: If an error occurs during embedding generation.
    """
    if not chunks:
        logger.warning("No chunks provided to generate embeddings.")
        return np.array([]), []

    # Extract the text content from the Chunk objects
    texts_to_embed: List[str] = [chunk["text"] for chunk in chunks]

    # Add a check for empty texts after extraction
    if not any(texts_to_embed):
        logger.warning(f"All chunks for doc_id '{chunks[0]['doc_id'] if chunks else 'unknown'}' have empty text content. Skipping embedding.")
        return np.array([]), []

    logger.info(f"Generating embeddings for {len(texts_to_embed)} text snippets...")
    try:
        # Use the provided model instance
        # Setting convert_to_numpy=True ensures it returns a numpy array
        embeddings = model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)

        if embeddings.shape[0] != len(texts_to_embed):
             logger.error(f"Mismatch between number of texts ({len(texts_to_embed)}) and generated embeddings ({embeddings.shape[0]}).")
             # Decide how to handle: raise error or return partial? Raising is safer.
             raise EmbeddingError(f"Mismatch between number of texts and generated embeddings.")

        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        # Return both embeddings and the original chunks list (if needed later, e.g., for FAISS index mapping)
        # In this flow, we save embeddings and index separately, so returning chunks here might be redundant
        # unless we want to save the embeddings *with* the chunk metadata somehow.
        # For now, just returning embeddings seems sufficient based on the rest of the code flow.
        # Let's return embeddings and the *original* list of chunks passed in, in case any were filtered.
        # However, the current logic doesn't filter, so let's just return embeddings.
        return embeddings, chunks # Returning chunks too for future flexibility if needed


    except Exception as e: # Catch generic embedding errors
        logger.error(f"Error generating embeddings: {e}", exc_info=True) # Add exc_info
        # Wrap in custom exception
        raise EmbeddingError(f"Error generating embeddings: {e}") from e


def save_embeddings(doc_id: str, embeddings: np.ndarray) -> str:
    """
    Saves embeddings to the knowledge base directory as a .npy file.

    Args:
        doc_id: The ID of the document the embeddings belong to.
        embeddings: The numpy array of embeddings.

    Returns:
        The absolute path where the embeddings were saved.

    Raises:
        EmbeddingError: If there is an error writing the file.
    """
    if embeddings.size == 0:
        logger.warning(f"No embeddings to save for document {doc_id}.")
        # Decide return type: None? Raise? Let's raise if this state shouldn't be reached normally
        # Or return None and let caller handle. Returning None and logging warning seems okay.
        # If we return None, the return type hint should be Optional[str]
        # Let's return None if there are no embeddings to save
        return None # type: ignore # numpy array size 0 check implies no path

    # Ensure directory exists
    try:
        os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
    except OSError as e:
        logger.error(f"❌ Error creating knowledge base directory '{KNOWLEDGE_BASE_DIR}': {e}", exc_info=True)
        raise EmbeddingError(f"Failed to create knowledge base directory: {KNOWLEDGE_BASE_DIR}") from e

    path = os.path.join(KNOWLEDGE_BASE_DIR, f"{doc_id}_embeddings.npy")
    logger.info(f"Attempting to save embeddings for doc_id '{doc_id}' to: {path}")
    try:
        np.save(path, embeddings)
        logger.info(f"✅ Embeddings saved: {path}")
        return path
    except IOError as e:
        logger.error(f"Error saving embeddings to {path}: {e}", exc_info=True) # Add exc_info
        # Wrap in custom exception
        raise EmbeddingError(f"Error saving embeddings for {doc_id}: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred saving embeddings for {doc_id}: {e}", exc_info=True) # Add exc_info
        # Wrap in custom exception
        raise EmbeddingError(f"Unexpected error saving embeddings for {doc_id}: {e}") from e


def build_and_save_faiss_index(doc_id: str, embeddings: np.ndarray) -> str:
    """
    Builds a new FAISS index from embeddings and saves it to disk.

    Args:
        doc_id: The ID of the document the index belongs to.
        embeddings: The numpy array of embeddings.

    Returns:
        The absolute path where the FAISS index was saved.

    Raises:
        EmbeddingError: If an error occurs during index building or saving.
    """
    if embeddings.size == 0:
        logger.warning(f"No embeddings provided to build index for document {doc_id}.")
        # Let's return None if no index was built
        return None # type: ignore # numpy array size 0 check implies no path


    logger.info(f"Building FAISS index for doc_id: {doc_id}...")
    try:
        dim = embeddings.shape[1]
        # Using IndexFlatL2 for simplicity, could use IndexIVFFlat or other
        # depending on the number of vectors and performance requirements.
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        logger.info(f"FAISS index built with {index.ntotal} vectors.")

        index_path = os.path.join(KNOWLEDGE_BASE_DIR, f"{doc_id}_index.faiss")
        logger.info(f"Attempting to save FAISS index to: {index_path}")
        faiss.write_index(index, index_path)
        logger.info(f"✅ FAISS index saved: {index_path}")
        return index_path
    except Exception as e: # FAISS errors might be generic
        logger.error(f"Error building or saving FAISS index for {doc_id}: {e}", exc_info=True) # Add exc_info
        # Wrap in custom exception
        raise EmbeddingError(f"Error building or saving FAISS index for {doc_id}: {e}") from e


def embed_document(doc_id: str) -> Optional[Tuple[str, str]]:
    """
    Orchestrates the embedding process for a document:
    Loads chunks, generates embeddings, saves embeddings, and builds/saves the FAISS index.

    Args:
        doc_id: The ID of the document to embed.

    Returns:
        A tuple containing the paths to the saved embeddings (.npy) and FAISS index (.faiss)
        if the process was successful and embeddings were generated.
        Returns None if no chunks were found or no embeddings were generated.

    Raises:
        EmbeddingError: If any critical error occurs during the embedding steps
                        (loading, generating, saving).
    """
    logger.info(f"Starting embedding process for doc_id: {doc_id}")
    embeddings_path = None
    index_path = None

    try:
        # Get the model instance from the VectorStore singleton
        # Assumes VectorStore has been initialized elsewhere (e.g., application startup)
        vector_store_instance = VectorStore() # Get the singleton instance
        if not hasattr(vector_store_instance, 'model') or vector_store_instance.model is None:
             logger.error("SentenceTransformer model not loaded in VectorStore instance.")
             raise EmbeddingError("Embedding model not initialized.")
        model = vector_store_instance.model
        logger.debug("Accessed SentenceTransformer model from VectorStore.")

        # Step 1: Load chunks (now List[Chunk])
        chunks = load_chunks_for_embedding(doc_id)

        if not chunks:
            logger.warning(f"Skipping embedding for {doc_id} as no chunks were loaded.")
            return None # Indicate that embedding was skipped gracefully

        # Step 2: Generate embeddings (gets List[Chunk] as input)
        embeddings, processed_chunks = generate_embeddings(chunks, model) # Now returns embeddings and chunks

        if embeddings.size == 0:
            logger.warning(f"No embeddings generated for {doc_id} from {len(processed_chunks)} chunks, skipping save/index.")
            return None # Indicate that embedding yielded no results

        # Step 3: Save embeddings
        embeddings_path = save_embeddings(doc_id, embeddings)
        # save_embeddings returns None if no embeddings were passed, but we already checked size > 0

        # Step 4: Build and save FAISS index
        index_path = build_and_save_faiss_index(doc_id, embeddings)
         # build_and_save_faiss_index returns None if no embeddings were passed, but we already checked size > 0


        # If we reached here, both should have been saved successfully (or raised EmbeddingError)
        logger.info(f"✨ Embedding process completed successfully for doc_id: {doc_id}")
        return embeddings_path, index_path # Return the paths

    except EmbeddingError as e:
        # Custom embedding errors are caught and logged within the helper functions
        # Re-raise the specific embedding error
        raise
    except Exception as e:
        # Catch any other unexpected errors during orchestration
        logger.error(f"An unexpected error occurred during embedding document {doc_id}: {e}", exc_info=True)
        # Wrap unexpected errors in the custom exception
        raise EmbeddingError(f"An unexpected error occurred during embedding for {doc_id}: {e}") from e


# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging for standalone execution
    from src.config import setup_logging, ensure_required_dirs
    setup_logging()
    ensure_required_dirs() # Ensure directories exist for this test

    logger.info("--- Testing src/embedder.py ---")

    # --- Setup Test Data ---
    # This requires that you have already processed a document using processor.py
    # and saved chunks for a specific doc_id in the PROCESSED_DATA_DIR.
    # You might need to manually create a dummy chunks file in data/processed
    # if you don't have a full processing pipeline setup for testing.
    # The dummy chunks file should match the structure saved by the improved chunker.py:
    # {
    #   "doc_id": "test_embedder_doc_789",
    #   "total_chunks": 2,
    #   "overlap_used": 50,
    #   "chunks": [
    #     {"chunk_id": "test_embedder_doc_789-0000", "text": "This is the first test chunk."},
    #     {"chunk_id": "test_embedder_doc_789-0001", "text": "This is the second test chunk with some overlap."},
    #   ]
    # }

    test_doc_id = "test_embedder_doc_789" # <<< REPLACE with an actual doc_id that has a chunks file

    # Create dummy chunks file if it doesn't exist for testing
    dummy_chunks_path = os.path.join(PROCESSED_DATA_DIR, f"{test_doc_id}_chunks.json")
    if not os.path.exists(dummy_chunks_path):
         logger.info(f"Creating dummy chunks file for testing at {dummy_chunks_path}")
         os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
         dummy_data = {
             "doc_id": test_doc_id,
             "total_chunks": 2,
             "overlap_used": 50,
             "chunks": [
                 {"chunk_id": f"{test_doc_id}-0000", "text": "This is the first test chunk designed for embedding."},
                 {"chunk_id": f"{test_doc_id}-0001", "text": "This is the second test chunk which overlaps with the first one."},
             ]
         }
         try:
             with open(dummy_chunks_path, "w", encoding="utf-8") as f:
                 json.dump(dummy_data, f, indent=2)
             logger.info("Dummy chunks file created.")
         except IOError as e:
              logger.error(f"Failed to create dummy chunks file: {e}", exc_info=True)


    # --- Run Embedding Process ---
    try:
        # Initialize VectorStore singleton (will load model) - needed before calling embed_document
        # This should happen once at application startup in a real app.
        logger.info("Initializing VectorStore (this loads the SentenceTransformer model)...")
        try:
            vs = VectorStore()
            logger.info("VectorStore initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {e}", exc_info=True)
            raise EmbeddingError("Failed to initialize VectorStore for embedding.") from e


        logger.info(f"\nAttempting to embed document with doc_id: {test_doc_id}")
        result = embed_document(test_doc_id)

        if result:
            emb_path, index_path = result
            logger.info(f"Embedding process finished successfully.")
            logger.info(f"Embeddings saved to: {emb_path}")
            logger.info(f"FAISS index saved to: {index_path}")

            # Optional: Load and check the saved files
            try:
                 loaded_embeddings = np.load(emb_path)
                 logger.info(f"Successfully loaded saved embeddings with shape: {loaded_embeddings.shape}")

                 loaded_index = faiss.read_index(index_path)
                 logger.info(f"Successfully loaded saved FAISS index with {loaded_index.ntotal} vectors.")

            except (IOError, ImportError, RuntimeError, Exception) as e:
                 # Catch potential FAISS or numpy errors during loading
                 logger.error(f"Failed to load saved embedding/index files: {e}", exc_info=True)

        else:
            logger.warning(f"Embedding process did not complete successfully for {test_doc_id} (might have skipped due to no chunks).")


        # Optional: Clean up test files (embeddings, index, dummy chunks)
        # emb_path = os.path.join(KNOWLEDGE_BASE_DIR, f"{test_doc_id}_embeddings.npy")
        # index_path = os.path.join(KNOWLEDGE_BASE_DIR, f"{test_doc_id}_index.faiss")
        # dummy_chunks_path = os.path.join(PROCESSED_DATA_DIR, f"{test_doc_id}_chunks.json")
        #
        # if os.path.exists(emb_path): os.remove(emb_path); logger.info(f"Cleaned up {emb_path}")
        # if os.path.exists(index_path): os.remove(index_path); logger.info(f"Cleaned up {index_path}")
        # if os.path.exists(dummy_chunks_path): os.remove(dummy_chunks_path); logger.info(f"Cleaned up {dummy_chunks_path}")

    except EmbeddingError as e:
        # Specific embedding errors are caught here
        logger.error(f"Embedding process failed for doc_id {test_doc_id}: {e}")
    except Exception as e:
         # Catch any other unexpected errors during the test run
         logger.error(f"Caught an unexpected error during embedder test for doc_id {test_doc_id}: {e}", exc_info=True)


    logger.info("--- End of embedder.py test ---")