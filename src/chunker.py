# src/chunker.py
"""
This module provides functionality to split raw text into smaller,
overlapping chunks suitable for embedding and retrieval, and to save these chunks to disk.
"""
import os
import json
import logging
import re
from typing import List, Dict, Any, TypedDict # Import TypedDict

from src.config import (
    PROCESSED_DATA_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP
)

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- Chunk Data Structure ---
# Define a simple structure for a chunk including metadata
class Chunk(TypedDict):
    """Represents a text chunk with its content and associated metadata."""
    chunk_id: str      # Unique ID for the chunk (e.g., doc_id-chunk_index)
    text: str          # The text content of the chunk
    doc_id: str        # The ID of the original document
    chunk_index: int   # The index of the chunk within the document
    # Add other relevant metadata here, e.g.,
    # original_source: str # Like filename or URL
    # page_number: Optional[int] # If parsing provided this


# --- Core Chunking Function ---

def chunk_text(text: str, doc_id: str, max_chunk_size: int = 200, overlap: int = 50) -> List[Chunk]:
    """
    Splits long document text into overlapping chunks with basic metadata.
    Attempts to split based on headings or paragraph breaks first.

    Args:
        text: The input text to chunk.
        doc_id: The ID of the document the text belongs to (used for chunk_id).
        max_chunk_size: The maximum number of words per chunk.
        overlap: The number of words to overlap between chunks.

    Returns:
        A list of Chunk dictionaries.
    """
    if not isinstance(text, str):
        logger.warning("Input text is not a string. Returning empty list.")
        return []

    text = text.strip()
    if not text:
        logger.warning(f"Input text is empty or whitespace only for doc_id '{doc_id}'. Returning empty list.")
        return []

    # --- Basic Chunking Logic ---
    # Note: This is a simple word-based and newline/header split chunker.
    # More sophisticated chunkers might use sentence tokenization,
    # handle different separators (markdown, HTML), or be aware of LLM token limits.

    # Split by line breaks, section headers (basic pattern), or double newlines
    # Using non-capturing groups `(?:...)` where appropriate, although re.split captures groups anyway.
    # Let's refine the split slightly to potentially keep delimiters if needed, or just split cleaner.
    # A simple split by large whitespace blocks or specific patterns:
    sections: List[str] = re.split(r'\s*\n\s*\n\s*|(?m)^\s*[#=]+\s*.*$', text) # Split by 2+ newlines or lines starting with # or = (basic header attempt)
    sections = [s.strip() for s in sections if s.strip()]

    raw_chunks: List[str] = []
    current_chunk_words: List[str] = []
    current_word_count: int = 0

    for i, section in enumerate(sections):
        # Split section into words
        words: List[str] = section.split()
        section_len: int = len(words)

        if section_len > max_chunk_size:
            # If a section is too long, split it into sub-chunks
            # Add overlap by taking words from the *end* of the previous sub-chunk
            section_sub_chunks: List[str] = []
            for j in range(0, section_len, max_chunk_size - overlap):
                 sub_chunk_words = words[j : j + max_chunk_size]
                 if sub_chunk_words:
                      section_sub_chunks.append(" ".join(sub_chunk_words))

            # Now, add these sub-chunks to the main list of raw_chunks.
            # We need to handle potential overlap with the *previous* section if applicable,
            # but the simple recursive split above doesn't easily carry context forward.
            # A better approach would be to process words iteratively.

            # Let's revise the logic to process words sequentially across sections
            # while respecting section boundaries for better semantic splits initially.

            # Clear current_chunk_words before processing a large section separately
            if current_chunk_words:
                raw_chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []
                current_word_count = 0

            # Process the large section as a series of sub-chunks
            prev_sub_chunk_words: List[str] = []
            for j in range(0, section_len, max_chunk_size - overlap):
                 sub_chunk_words = words[j : j + max_chunk_size]
                 if sub_chunk_words:
                      # Add overlap from the *previous* sub-chunk's end
                      effective_sub_chunk = prev_sub_chunk_words[-overlap:] + sub_chunk_words
                      raw_chunks.append(" ".join(effective_sub_chunk))
                      prev_sub_chunk_words = sub_chunk_words # Store for the next iteration

        else: # Section fits within or extends the current chunk
            if current_word_count + section_len <= max_chunk_size:
                # Add the whole section if it fits
                current_chunk_words.extend(words)
                current_word_count += section_len
            else:
                # Current section doesn't fit, close the current chunk
                if current_chunk_words:
                    raw_chunks.append(" ".join(current_chunk_words))

                # Start a new chunk with overlap from the *end* of the previous chunk
                overlap_words = current_chunk_words[-overlap:] if overlap and current_chunk_words else []
                current_chunk_words = overlap_words + words
                current_word_count = len(current_chunk_words) # Re-calculate based on new list

    # Add the last chunk if it's not empty
    if current_chunk_words:
        raw_chunks.append(" ".join(current_chunk_words))

    # --- Create Chunks with Metadata ---
    chunks_with_metadata: List[Chunk] = []
    for i, chunk_text in enumerate(raw_chunks):
        if chunk_text.strip(): # Ensure chunk is not empty after joining/stripping
            chunks_with_metadata.append(Chunk(
                chunk_id=f"{doc_id}-{i:04d}", # Create a unique ID for each chunk
                text=chunk_text.strip(), # Strip leading/trailing whitespace
                doc_id=doc_id,
                chunk_index=i,
                # Add other metadata here if available from parsing or context
                # original_source="your_source_identifier",
                # page_number=get_page_for_chunk(i) # Hypothetical function
            ))

    logger.info(f"Generated {len(chunks_with_metadata)} chunks for doc_id '{doc_id}'")
    return chunks_with_metadata


# --- Saving Chunks Function ---

# Updated function signature to accept output_dir explicitly
def save_chunks_to_disk(
    chunks: List[Chunk], # Expect List[Chunk] now
    doc_id: str,
    output_dir: str = PROCESSED_DATA_DIR, # Use PROCESSED_DATA_DIR as default, but accept parameter
    overlap_used: int = DEFAULT_CHUNK_OVERLAP # Keep for metadata, default from config
) -> str:
    """
    Saves a list of Chunk dictionaries with metadata to disk as JSON.

    Args:
        chunks: The list of Chunk dictionaries to save.
        doc_id: The ID of the document these chunks belong to.
        output_dir: The directory where the chunk file will be saved. Defaults to PROCESSED_DATA_DIR.
        overlap_used: The actual overlap value used during chunking (for metadata).

    Returns:
        The absolute file path where the chunks were saved.

    Raises:
        IOError: If there is an error writing the file.
        TypeError: If chunk data is not JSON serializable.
        Exception: For other unexpected errors.
    """
    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"❌ Error creating output directory '{output_dir}': {e}", exc_info=True)
        raise IOError(f"Failed to create output directory: {output_dir}") from e


    output_filename = f"{doc_id}_chunks.json"
    output_path = os.path.join(output_dir, output_filename)

    # The chunk_data structure can now be simpler as metadata is in each Chunk
    chunk_file_content: Dict[str, Any] = {
        "doc_id": doc_id,
        "total_chunks": len(chunks),
        "overlap_used": overlap_used, # Store the actual overlap used
        "chunks": chunks # List of Chunk dictionaries
    }

    try:
        # Use 'w' for writing, 'utf-8' encoding is good practice
        with open(output_path, "w", encoding="utf-8") as f:
            # Use json.dump with indent for readability
            json.dump(chunk_file_content, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Saved {len(chunks)} chunks for doc_id '{doc_id}' to {output_path}")
        return output_path

    except (IOError, TypeError, Exception) as e:
        # Catch multiple potential errors and log with exc_info
        logger.error(f"❌ Error saving chunks for doc_id '{doc_id}' to {output_path}: {e}", exc_info=True)
        # Re-raise the original exception
        raise


# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging for standalone execution
    # Note: In a real app, setup_logging should be called once at the app entry point (e.g., app.py)
    from src.config import setup_logging, ensure_required_dirs
    setup_logging()
    ensure_required_dirs() # Ensure directories exist for this test

    logger.info("--- Testing src/chunker.py ---")

    # --- Setup Test Data ---
    test_doc_id = "test_chunker_doc_456"
    long_text = (
        "This is the first paragraph of a test document. It contains several sentences. "
        "We want to see how the chunker handles relatively short paragraphs.\n\n"
        "This is the second paragraph. It is a bit longer than the first one. " * 5 + "\n\n" # Make it longer
        "## Section Header Example\n\n" # Example header
        "This is content under a header. It should ideally start a new chunk or influence splitting. "
        "More content follows to test chunk boundaries and overlap.\n\n"
        "Final paragraph to ensure everything is processed."
    )

    test_chunk_size = DEFAULT_CHUNK_SIZE # Use default from config
    test_overlap = DEFAULT_CHUNK_OVERLAP # Use default from config

    logger.info(f"Input text length: {len(long_text)} characters for doc_id '{test_doc_id}'")
    logger.info(f"Chunking with max_chunk_size={test_chunk_size}, overlap={test_overlap}")

    # --- Run Chunking and Saving ---
    try:
        chunks = chunk_text(long_text, doc_id=test_doc_id, max_chunk_size=test_chunk_size, overlap=test_overlap)
        logger.info(f"Generated {len(chunks)} chunks.")

        if chunks:
            # Specify the output directory for the test
            chunk_file_path = save_chunks_to_disk(
                chunks=chunks,
                doc_id=test_doc_id,
                output_dir=PROCESSED_DATA_DIR, # Use the config directory for testing
                overlap_used=test_overlap
            )
            logger.info(f"Chunks saved to: {chunk_file_path}")

            # Optional: Read and print saved chunks for verification
            try:
                with open(chunk_file_path, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                logger.info(f"Saved chunk data (first chunk): {saved_data['chunks'][0] if saved_data['chunks'] else 'No chunks'}")
                logger.info(f"Saved chunk data (last chunk): {saved_data['chunks'][-1] if len(saved_data['chunks']) > 1 else 'Only one chunk'}")

                # Optional: Clean up the test file
                # if os.path.exists(chunk_file_path):
                #     os.remove(chunk_file_path)
                #     logger.info(f"Cleaned up test chunk file: {chunk_file_path}")

            except (IOError, json.JSONDecodeError) as e:
                logger.error(f"Failed to read or cleanup test chunk file {chunk_file_path}: {e}", exc_info=True)

        else:
            logger.warning("No chunks were generated to save.")

    except Exception as e:
        logger.error(f"An error occurred during chunker utility test: {e}", exc_info=True)

    logger.info("--- End of chunker.py test ---")