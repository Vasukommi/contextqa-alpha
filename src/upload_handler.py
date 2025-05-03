# src/upload_handler.py
"""
This module handles the process of receiving a document file, assigning a unique ID,
storing it in the designated upload directory, and updating the application's metadata file.
"""
import os
import uuid
import json
import shutil
import logging
from datetime import datetime
from typing import List, Dict, Any # Import for type hinting
from src.config import (
    UPLOAD_DIR,
    METADATA_FILE,
    DATA_DIR, # Need this to ensure data/ exists for metadata.json
    TEMP_UPLOAD_DIR # Import TEMP_UPLOAD_DIR if you plan to use it
)
from src.utils.metadata import load_metadata, save_metadata # Use centralized metadata handlers

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- Directory Management ---

def ensure_upload_dirs() -> None:
    """
    Ensures the necessary directories for uploads and data storage exist.
    Uses paths configured in src.config.
    """
    # Use centralized paths from config
    dirs_to_ensure = [
        UPLOAD_DIR,
        DATA_DIR, # Ensure data/ exists for metadata file
        # TEMP_UPLOAD_DIR # Uncomment if handle_upload or other functions use a temporary upload area
    ]

    try:
        logger.info("Ensuring upload and data directories exist...")
        for dir_path in dirs_to_ensure:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
        logger.info("✅ Ensured necessary directories exist.")
    except OSError as e:
        logger.critical(f"❌ Critical error ensuring upload directories exist: {e}", exc_info=True) # Use critical for setup failure
        raise # Re-raise the exception, as app cannot function without these dirs

# --- File Upload Handling ---

# Optional: Define a custom exception for upload handling errors
# class UploadHandlingError(Exception):
#     """Custom exception for errors during document upload handling."""
#     pass


def handle_upload(file_path: str) -> str:
    """
    Handles the final step of processing an uploaded document file:
    - Assigns a unique document ID (UUID).
    - Determines file type and performs basic validation.
    - Copies the file to the permanent upload directory using the doc_id as filename.
    - Updates the global metadata.json file with the new document's information.

    Note: This function expects `file_path` to be the source location of the file
    after it has been initially received (e.g., from a web request body saved
    to a temporary location or passed directly).

    Args:
        file_path: The absolute or relative path to the source file to be handled.

    Returns:
        The generated document ID (str) if handling is successful.

    Raises:
        FileNotFoundError: If the source file does not exist at the provided `file_path`.
        ValueError: If the file has an invalid/missing extension or an unsupported file type.
        IOError: If there is an error during file copying or metadata saving.
        json.JSONDecodeError: If the metadata file is corrupt during loading.
        # Or UploadHandlingError: If using a custom exception
    """
    logger.info(f"Attempting to handle upload for source file: {file_path}")

    # Ensure directories are ready before proceeding
    # Can be called here or once at application startup.
    # Calling here adds robustness if directories are removed.
    ensure_upload_dirs()

    if not os.path.exists(file_path):
        logger.error(f"❌ Source file not found for upload handling: {file_path}", exc_info=True) # Add exc_info
        raise FileNotFoundError(f"Source file not found: {file_path}")

    # Get original filename and extension robustly
    original_filename: str = os.path.basename(file_path)
    name, ext = os.path.splitext(original_filename)

    if not ext or len(ext) < 2: # Check if extension exists and has at least one character after the dot
        logger.warning(f"File has invalid or missing extension: {original_filename}")
        raise ValueError(f"File '{original_filename}' has invalid or missing extension.")

    # Remove leading dot from extension and make lowercase
    file_extension: str = ext[1:].lower()

    # Basic check for supported file types
    supported_extensions: List[str] = ['pdf', 'docx', 'txt'] # Add others as you implement parsers
    if file_extension not in supported_extensions:
        logger.warning(f"Unsupported file type for upload: '{file_extension}' from file {original_filename}")
        raise ValueError(f"Unsupported file type: '{file_extension}' for file '{original_filename}'. Supported types are: {', '.join(supported_extensions)}")


    doc_id: str = str(uuid.uuid4()) # Generate a unique ID
    logger.info(f"Generated doc_id: {doc_id} for file {original_filename} (type: {file_extension})")

    new_filename: str = f"{doc_id}.{file_extension}"
    dest_path: str = os.path.join(UPLOAD_DIR, new_filename)
    logger.debug(f"Constructed destination path: {dest_path}")

    try:
        logger.info(f"Copying '{original_filename}' to permanent storage at '{dest_path}'...")
        # Use shutil.copy2 to preserve metadata like timestamps if desired
        shutil.copy2(file_path, dest_path)
        logger.info(f"✅ Successfully copied file: '{original_filename}' → '{new_filename}' to {dest_path}")

        # Optional: Clean up the original source file if it was a temporary file
        # if file_path.startswith(TEMP_UPLOAD_DIR): # Basic check, or pass a flag
        #     try:
        #         os.remove(file_path)
        #         logger.debug(f"Cleaned up temporary file: {file_path}")
        #     except OSError as cleanup_error:
        #         logger.warning(f"Failed to clean up temporary file {file_path}: {cleanup_error}", exc_info=True)


    except shutil.Error as e:
        logger.error(f"❌ Shutil error copying file {file_path} to {dest_path}: {e}", exc_info=True)
        raise IOError(f"Error copying file '{original_filename}': {e}") from e # Chain exception
    except IOError as e:
        logger.error(f"❌ IO error copying file {file_path} to {dest_path}: {e}", exc_info=True)
        raise IOError(f"Error copying file '{original_filename}': {e}") from e # Chain exception
    except Exception as e: # Catch any other unexpected copy errors
        logger.error(f"❌ An unexpected error occurred during file copy from {file_path} to {dest_path}: {e}", exc_info=True)
        raise IOError(f"Unexpected error copying file '{original_filename}': {e}") from e # Chain exception


    # --- Update Metadata ---
    try:
        logger.debug("Loading existing metadata...")
        # Use the centralized load_metadata function
        metadata: List[Dict[str, Any]] = load_metadata()
        logger.debug(f"Loaded {len(metadata)} existing metadata entries.")
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        logger.error(f"❌ Error loading metadata during upload handling of {original_filename}: {e}", exc_info=True)
        # If metadata loading fails, the uploaded file is orphaned. Raise error.
        raise # Re-raise the specific error


    # Append new document's metadata entry
    new_entry: Dict[str, Any] = {
        "doc_id": doc_id,
        "original_filename": original_filename,
        "file_type": file_extension,
        "upload_time": datetime.utcnow().isoformat() + "Z", # Use ISO format with Z for UTC
        "status": "uploaded" # Add an initial status
        # Add other metadata here if available/relevant, e.g., size, uploader ID
    }
    metadata.append(new_entry)
    logger.debug(f"Appended new metadata entry for doc_id {doc_id}.")

    try:
        logger.debug(f"Saving updated metadata to {METADATA_FILE}...")
        # Use the centralized save_metadata function
        save_metadata(metadata)
        logger.info(f"✅ Successfully updated metadata.json for doc_id: {doc_id}")
    except (IOError, TypeError) as e:
        logger.error(f"❌ Error saving metadata for doc_id {doc_id}: {e}", exc_info=True)
        # If metadata fails, the file is copied but not tracked. This is critical.
        # Could attempt to delete the copied file here to clean up, but raising is simpler.
        raise IOError(f"Error saving metadata for uploaded file '{original_filename}': {e}") from e # Chain exception
    except Exception as e: # Catch any other unexpected metadata saving errors
         logger.error(f"❌ An unexpected error occurred saving metadata for doc_id {doc_id}: {e}", exc_info=True)
         raise IOError(f"Unexpected error saving metadata for uploaded file '{original_filename}': {e}") from e # Chain exception


    logger.info(f"✅ Document upload handled successfully. Assigned doc_id: {doc_id}")
    return doc_id


# --- CLI Test Example ---
if __name__ == "__main__":
    # Configure logging if running this script directly
    # Note: In a real application, logging setup should be handled once at the application entry point (e.g., app.py).
    setup_logging()
    logger.info("--- Running upload_handler CLI test ---")

    # Ensure test directories exist
    try:
        ensure_upload_dirs() # Ensure upload/data directories are ready
    except Exception as e:
        logger.error(f"Failed to ensure upload directories for test: {e}")
        exit(1) # Exit if directories cannot be set up

    # Define a path for a dummy test file
    # Use a path in the current directory for simplicity, or in TEMP_UPLOAD_DIR if testing that flow
    test_file_dir = "."
    # test_file_dir = TEMP_UPLOAD_DIR # Uncomment if you want to test using temp uploads as source
    dummy_test_filename = "dummy_upload_test_file.txt"
    dummy_test_file_path = os.path.join(test_file_dir, dummy_test_filename)
    dummy_file_content = "This is a dummy file created for testing the upload handler.\nIt has some text content."


    # --- Test Case 1: Successful Upload ---
    logger.info("\n--- Test Case 1: Successful Upload ---")
    # Create the dummy test file if it doesn't exist
    if not os.path.exists(dummy_test_file_path):
        logger.info(f"Creating dummy test file for successful upload test: {dummy_test_file_path}")
        try:
            os.makedirs(test_file_dir, exist_ok=True) # Ensure test dir exists
            with open(dummy_test_file_path, "w", encoding="utf-8") as f:
                f.write(dummy_file_content)
            logger.info("Dummy test file created.")
        except IOError as e:
            logger.error(f"Failed to create dummy test file {dummy_test_file_path}: {e}", exc_info=True)
            dummy_test_file_path = None # Indicate failure

    if dummy_test_file_path and os.path.exists(dummy_test_file_path):
        logger.info(f"Attempting to handle upload for: {dummy_test_file_path}")
        uploaded_doc_id_1 = None
        try:
            uploaded_doc_id_1 = handle_upload(dummy_test_file_path)
            logger.info(f"✅ Test Case 1 Successful: Upload handled with doc_id: {uploaded_doc_id_1}")

        except (FileNotFoundError, ValueError, IOError, json.JSONDecodeError, Exception) as e:
            logger.error(f"❌ Test Case 1 Failed: Error during successful upload test: {e}", exc_info=True)

        finally:
             # Optional: Clean up the dummy source file
             if os.path.exists(dummy_test_file_path) and test_file_dir == ".": # Only remove if created in current dir
                 try:
                      os.remove(dummy_test_file_path)
                      logger.debug(f"Cleaned up dummy source file: {dummy_test_file_path}")
                 except OSError as cleanup_error:
                      logger.warning(f"Failed to clean up dummy source file {dummy_test_file_path}: {cleanup_error}", exc_info=True)

             # Optional: Clean up the uploaded file and metadata entry
             if uploaded_doc_id_1:
                  try:
                       uploaded_file_path = os.path.join(UPLOAD_DIR, f"{uploaded_doc_id_1}.txt")
                       if os.path.exists(uploaded_file_path):
                            os.remove(uploaded_file_path)
                            logger.debug(f"Cleaned up uploaded file: {uploaded_file_path}")

                       # Remove from metadata
                       metadata = load_metadata()
                       metadata = [entry for entry in metadata if entry.get("doc_id") != uploaded_doc_id_1]
                       save_metadata(metadata)
                       logger.debug(f"Removed metadata entry for doc_id {uploaded_doc_id_1}")

                  except Exception as cleanup_error:
                       logger.warning(f"Failed to clean up uploaded artifacts for {uploaded_doc_id_1}: {cleanup_error}", exc_info=True)

    else:
         logger.warning("Skipping Test Case 1 as dummy test file could not be created or found.")


    # --- Test Case 2: File Not Found ---
    logger.info("\n--- Test Case 2: File Not Found ---")
    non_existent_file = "./this_file_does_not_exist_12345.pdf"
    logger.info(f"Attempting to handle upload for non-existent file: {non_existent_file} (expecting FileNotFoundError)")
    try:
        handle_upload(non_existent_file)
        logger.error("❌ Test Case 2 Failed: FileNotFoundError was NOT raised for non-existent file.")
    except FileNotFoundError as e:
        logger.info(f"✅ Test Case 2 Successful: Caught expected error: {e}")
    except Exception as e:
        logger.error(f"❌ Test Case 2 Failed: Caught unexpected error: {e}", exc_info=True)


    # --- Test Case 3: Invalid Extension ---
    logger.info("\n--- Test Case 3: Invalid Extension ---")
    invalid_ext_file = "./file_without_ext"
    # Create a dummy file without an extension
    if not os.path.exists(invalid_ext_file):
        try:
            with open(invalid_ext_file, "w") as f:
                f.write("This file has no extension.")
        except IOError as e:
            logger.warning(f"Failed to create dummy file without extension {invalid_ext_file}: {e}", exc_info=True)
            invalid_ext_file = None

    if invalid_ext_file and os.path.exists(invalid_ext_file):
        logger.info(f"Attempting to handle upload for file with invalid extension: {invalid_ext_file} (expecting ValueError)")
        try:
            handle_upload(invalid_ext_file)
            logger.error("❌ Test Case 3 Failed: ValueError was NOT raised for invalid extension.")
        except ValueError as e:
            logger.info(f"✅ Test Case 3 Successful: Caught expected error: {e}")
        except Exception as e:
            logger.error(f"❌ Test Case 3 Failed: Caught unexpected error: {e}", exc_info=True)
        finally:
             # Clean up the dummy file
             try:
                  if os.path.exists(invalid_ext_file):
                       os.remove(invalid_ext_file)
                       logger.debug(f"Cleaned up dummy file: {invalid_ext_file}")
             except OSError as cleanup_error:
                  logger.warning(f"Failed to clean up dummy file {invalid_ext_file}: {cleanup_error}", exc_info=True)

    else:
         logger.warning("Skipping Test Case 3 as dummy invalid extension file could not be created.")


    # --- Test Case 4: Unsupported Extension ---
    logger.info("\n--- Test Case 4: Unsupported Extension ---")
    unsupported_file = "./unsupported_file.zip"
    # Create a dummy file with an unsupported extension
    if not os.path.exists(unsupported_file):
        try:
            with open(unsupported_file, "w") as f:
                f.write("This file has an unsupported extension.")
        except IOError as e:
            logger.warning(f"Failed to create dummy file with unsupported extension {unsupported_file}: {e}", exc_info=True)
            unsupported_file = None

    if unsupported_file and os.path.exists(unsupported_file):
        logger.info(f"Attempting to handle upload for file with unsupported extension: {unsupported_file} (expecting ValueError)")
        try:
            handle_upload(unsupported_file)
            logger.error("❌ Test Case 4 Failed: ValueError was NOT raised for unsupported extension.")
        except ValueError as e:
            logger.info(f"✅ Test Case 4 Successful: Caught expected error: {e}")
        except Exception as e:
            logger.error(f"❌ Test Case 4 Failed: Caught unexpected error: {e}", exc_info=True)
        finally:
             # Clean up the dummy file
             try:
                  if os.path.exists(unsupported_file):
                       os.remove(unsupported_file)
                       logger.debug(f"Cleaned up dummy file: {unsupported_file}")
             except OSError as cleanup_error:
                  logger.warning(f"Failed to clean up dummy file {unsupported_file}: {cleanup_error}", exc_info=True)
    else:
         logger.warning("Skipping Test Case 4 as dummy unsupported extension file could not be created.")


    # Note: Testing metadata errors (IOError, JSONDecodeError during load/save)
    # is harder in a simple test without mocking file system operations or
    # intentionally corrupting the metadata file.

    logger.info("\n--- End of upload_handler CLI test ---")