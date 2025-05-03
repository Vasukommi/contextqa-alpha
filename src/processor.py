# src/processor.py
"""
This module orchestrates the document processing pipeline:
- Retrieves file information from metadata.
- Parses various file types (PDF, DOCX, TXT).
- Chunks the extracted text into manageable pieces.
- Saves the processed chunks to disk.
"""
import os
import json
import logging
from typing import List, Tuple, Optional # Import Optional for type hints
from src.config import METADATA_FILE # Import METADATA_FILE from config

# Import parsers - ensure these files exist in src/parsers
# It's good practice to wrap these in a try/except if parsers might be optional
# depending on installed dependencies (e.g., docx might need python-docx)
try:
    from src.parsers.pdf_parser import parse_pdf
except ImportError as e:
    # Log an error but don't necessarily fail immediately,
    # only fail if a PDF is actually processed.
    logging.error(f"Could not import PDF parser: {e}. PDF processing will not be available.", exc_info=True)
    parse_pdf = None # Set to None if import fails

try:
    from src.parsers.docx_parser import parse_docx
except ImportError as e:
    logging.error(f"Could not import DOCX parser: {e}. DOCX processing will not be available.", exc_info=True)
    parse_docx = None

try:
    from src.parsers.txt_parser import parse_txt
except ImportError as e:
    logging.error(f"Could not import TXT parser: {e}. TXT processing will not be available.", exc_info=True)
    parse_txt = None


# Import functions from our modules
from src.chunker import chunk_text, save_chunks_to_disk, Chunk # Import Chunk type hint
from src.utils.metadata import load_metadata # Use centralized metadata loader
from src.config import UPLOAD_DIR, PROCESSED_DATA_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP # Use centralized config

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- Custom Exception (Optional but good practice) ---
class DocumentProcessingError(Exception):
    """Custom exception for errors during document processing."""
    pass

# --- Core Functions ---

def get_file_info(doc_id: str) -> Tuple[str, str]:
    """
    Retrieves the file path and type from metadata based on doc_id.

    Args:
        doc_id: The ID of the document to find information for.

    Returns:
        A tuple containing the absolute file path and the file extension (e.g., ('/path/to/file.pdf', 'pdf')).

    Raises:
        DocumentProcessingError: If the metadata cannot be loaded, doc_id is not found, or the associated file is missing/invalid.
    """
    logger.info(f"Attempting to get file info for doc_id: {doc_id}")
    try:
        metadata = load_metadata() # Use the centralized load_metadata function
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        # Wrap loading errors in a more specific processing error
        logger.error(f"Failed to load metadata while getting file info for {doc_id}: {e}", exc_info=True)
        raise DocumentProcessingError(f"Failed to load metadata for doc_id {doc_id}") from e


    for entry in metadata:
        # Use .get() for safer access in case an entry is malformed
        entry_doc_id = entry.get("doc_id")
        entry_file_type = entry.get("file_type")
        original_filename = entry.get("original_filename", "unknown") # Get original name for logging/errors

        if entry_doc_id == doc_id:
            if not entry_file_type:
                 # Use logger.warning for missing file type in metadata
                 logger.warning(f"Metadata entry for doc_id {doc_id} ({original_filename}) is missing 'file_type'. Skipping entry.")
                 continue # Skip invalid entry

            # Construct the expected filename using the doc_id and file_type
            filename = f"{doc_id}.{entry_file_type}"
            file_path = os.path.join(UPLOAD_DIR, filename) # Use centralized UPLOAD_DIR

            # Verify the file exists at the expected path
            if not os.path.exists(file_path):
                 # Log warning, then skip this entry as the file is missing
                 logger.warning(f"Metadata found for doc_id {doc_id} ({original_filename}), but source file does not exist at expected path: {file_path}. Skipping this metadata entry.")
                 continue # Continue searching in case there's another valid entry (unlikely for unique doc_id, but safer)

            logger.info(f"✅ Found file info for {doc_id}: {file_path} (type: {entry_file_type})")
            return file_path, entry_file_type

    # If the loop finishes without finding the doc_id or a valid entry
    logger.error(f"❌ Document ID not found in metadata or source file is missing: {doc_id}")
    # Wrap the error in a more specific processing error
    raise DocumentProcessingError(f"Document ID not found in metadata or source file is missing: {doc_id}")


def read_file_by_type(file_path: str, file_type: str) -> str:
    """
    Dispatches to the appropriate parser based on file type and reads the file.

    Args:
        file_path: The absolute path to the file to read.
        file_type: The extension of the file (e.g., 'pdf', 'docx', 'txt').

    Returns:
        The extracted text content of the file.

    Raises:
        DocumentProcessingError: If the file type is unsupported or parsing fails.
    """
    # Map file types to parser functions
    parsers = {
        "pdf": parse_pdf,
        "docx": parse_docx,
        "txt": parse_txt,
        # Add other parsers here as implemented
        # "xlsx": parse_xlsx,
        # "pptx": parse_pptx,
    }

    parser_func = parsers.get(file_type)

    if not parser_func:
        # Log error and raise specific processing error
        logger.error(f"Unsupported file type attempted for parsing: '{file_type}' from file {file_path}")
        raise DocumentProcessingError(f"Unsupported file type: {file_type}")

    # Check if the parser function was imported successfully
    if parser_func is None:
         # Log error and raise specific processing error if parser import failed earlier
         logger.error(f"Parser for file type '{file_type}' is not available (dependency might be missing?). File: {file_path}")
         raise DocumentProcessingError(f"Parser not available for file type: {file_type}. Check dependencies.")


    try:
        logger.info(f"⚙️ Parsing file: {file_path} (type: {file_type})...")
        raw_text = parser_func(file_path)
        logger.info(f"✅ Successfully parsed file: {file_path}. Extracted {len(raw_text)} characters.")
        return raw_text
    except Exception as e: # Catch any exception during parsing
        # Log error with traceback and wrap in specific processing error
        logger.error(f"❌ Error parsing file {file_path} with type {file_type}: {e}", exc_info=True)
        raise DocumentProcessingError(f"Error parsing file {os.path.basename(file_path)}: {e}") from e


def process_document(doc_id: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> Optional[str]:
    """
    Processes a document identified by its doc_id:
    1. Gets file info from metadata.
    2. Reads the file content using the appropriate parser.
    3. Chunks the raw text into smaller pieces with overlap.
    4. Saves the chunks to disk with metadata.

    Args:
        doc_id: The ID of the document to process.
        chunk_size: The size of chunks (in words). Defaults to DEFAULT_CHUNK_SIZE from config.
        chunk_overlap: The overlap between chunks (in words). Defaults to DEFAULT_CHUNK_OVERLAP from config.

    Returns:
        The absolute file path where the chunks were saved, or None if no chunks were generated.

    Raises:
        DocumentProcessingError: If any critical error occurs during the processing steps
                                 (metadata lookup, parsing, or saving chunks).
    """
    logger.info(f"Starting document processing for doc_id: {doc_id}")
    logger.info(f"Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    # Step 1: Get file info
    try:
        file_path, file_type = get_file_info(doc_id)
    except DocumentProcessingError as e:
        # Error is already logged in get_file_info
        raise # Re-raise the specific processing error

    # Step 2: Read file content
    raw_text: str
    try:
        raw_text = read_file_by_type(file_path, file_type)
    except DocumentProcessingError as e:
         # Error is already logged in read_file_by_type
         raise # Re-raise the specific processing error


    if not raw_text or not raw_text.strip(): # Check for empty or whitespace-only text
        logger.warning(f"⚠️ No significant content extracted from: {file_path}. No chunks will be generated.")
        # Decide how to handle: return None, save an empty file, or raise error?
        # Returning None signals that processing finished but yielded no content.
        return None


    # Step 3: Chunk the text
    chunks: List[Chunk]
    try:
        # Chunk size and overlap are passed from the function parameters
        chunks = chunk_text(raw_text, doc_id=doc_id, max_chunk_size=chunk_size, overlap=chunk_overlap)
        logger.info(f"✂️ Chunked into {len(chunks)} blocks from {len(raw_text)} characters.")
    except Exception as e: # Catch errors from chunk_text
        # Log error with traceback and wrap in specific processing error
        logger.error(f"❌ Error during chunking of {file_path}: {e}", exc_info=True)
        raise DocumentProcessingError(f"Error during chunking of {os.path.basename(file_path)}: {e}") from e

    if not chunks:
         logger.warning(f"⚠️ Chunking resulted in no chunks for {file_path}. No chunk file will be saved.")
         # Returning None also handles the case where chunking yields an empty list
         return None


    # Step 4: Save the chunks
    try:
        # Pass the actual overlap used to save_chunks_to_disk
        # Ensure PROCESSED_DATA_DIR exists - could add a check here or rely on ensure_required_dirs
        # os.makedirs(PROCESSED_DATA_DIR, exist_ok=True) # Redundant if ensure_required_dirs is called on startup

        chunk_file_path = save_chunks_to_disk(
            chunks=chunks,
            doc_id=doc_id,
            output_dir=PROCESSED_DATA_DIR, # Pass the output directory from config
            overlap_used=chunk_overlap
        )
        logger.info(f"✅ Chunks saved to: {chunk_file_path}")
        logger.info(f"✨ Document processing completed successfully for doc_id: {doc_id}")
        return chunk_file_path
    except IOError as e:
         # Log error with traceback and wrap in specific processing error
         logger.error(f"❌ Failed to save chunks for {doc_id} to {PROCESSED_DATA_DIR}: {e}", exc_info=True)
         raise DocumentProcessingError(f"Failed to save chunks for {doc_id}: {e}") from e


# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging for standalone execution
    # Note: In a real app, setup_logging should be called once at the app entry point (e.g., app.py)
    from src.config import setup_logging, ensure_required_dirs
    setup_logging()
    ensure_required_dirs() # Ensure directories exist for this test

    logger.info("--- Testing src/processor.py ---")

    # --- Setup Test Data ---
    # Create dummy files and metadata for testing if they don't exist
    dummy_doc_id = "test_doc_123"
    dummy_filename_pdf = f"{dummy_doc_id}.pdf"
    dummy_filepath_pdf = os.path.join(UPLOAD_DIR, dummy_filename_pdf)
    dummy_doc_id_txt = "test_doc_txt_456"
    dummy_filename_txt = f"{dummy_doc_id_txt}.txt"
    dummy_filepath_txt = os.path.join(UPLOAD_DIR, dummy_filename_txt)


    # Create dummy PDF content (requires a minimal PDF structure or a library)
    # For simplicity in this example, let's create a dummy text file and metadata entry for it.
    # If you have a real PDF, place it in data/uploads and update the doc_id in metadata.json
    if not os.path.exists(dummy_filepath_txt):
        dummy_txt_content = "This is a test document for processing. It contains several sentences. We will check if the chunking works correctly. Overlap is also important to consider for contextual information between chunks. This is the end of the document."
        try:
            with open(dummy_filepath_txt, "w") as f:
                f.write(dummy_txt_content)
            logger.info(f"Created dummy TXT file: {dummy_filepath_txt}")

            # Add dummy metadata entry (requires src/utils/metadata.py to handle append/update)
            # For a simple test, manually update metadata.json or use a helper function if available
            try:
                metadata_entries = load_metadata()
            except (FileNotFoundError, json.JSONDecodeError):
                 metadata_entries = [] # Start with empty if file doesn't exist or is bad

            # Check if entry already exists
            if not any(entry.get("doc_id") == dummy_doc_id_txt for entry in metadata_entries):
                new_entry = {
                    "doc_id": dummy_doc_id_txt,
                    "original_filename": dummy_filename_txt,
                    "file_type": "txt",
                    "upload_time": "TEST_TIMESTAMP", # Replace with actual time in a real scenario
                    "status": "uploaded" # Or whatever status is appropriate
                }
                metadata_entries.append(new_entry)

                try:
                    with open(METADATA_FILE, "w") as f:
                        json.dump(metadata_entries, f, indent=4)
                    logger.info(f"Added dummy metadata entry for {dummy_doc_id_txt} to {METADATA_FILE}")
                except IOError as e:
                    logger.error(f"Failed to write dummy metadata file {METADATA_FILE}: {e}", exc_info=True)

        except IOError as e:
             logger.error(f"Failed to create dummy TXT file {dummy_filepath_txt}: {e}", exc_info=True)

    # --- Run Processing ---
    # Use the dummy doc_id created above or one from your existing metadata.json
    doc_id_to_process = dummy_doc_id_txt # Or use a doc_id from your real files

    if doc_id_to_process:
        try:
            logger.info(f"\nAttempting to process document with doc_id: {doc_id_to_process}")
            processed_path = process_document(doc_id_to_process)

            if processed_path:
                logger.info(f"Processing complete. Chunks saved to: {processed_path}")
                # Optional: Read and print saved chunks for verification
                # try:
                #     with open(processed_path, 'r') as f:
                #         saved_chunks_data = json.load(f)
                #     logger.info(f"Saved chunks data sample: {saved_chunks_data[:2]}...") # Print first 2 chunks
                # except (IOError, json.JSONDecodeError) as e:
                #     logger.error(f"Failed to read saved chunk file {processed_path}: {e}", exc_info=True)
            else:
                 logger.info(f"Processing complete for doc_id: {doc_id_to_process}, but no chunks were generated.")


        except DocumentProcessingError as e:
            # Specific processing errors are caught here
            logger.error(f"Document processing failed for doc_id {doc_id_to_process}: {e}")
        except Exception as e:
            # Catch any other unexpected errors during the test
            logger.error(f"An unexpected error occurred during processing test for doc_id {doc_id_to_process}: {e}", exc_info=True)

    else:
         logger.warning("No document ID specified for testing. Skipping processing test.")

    logger.info("--- End of processor.py test ---")