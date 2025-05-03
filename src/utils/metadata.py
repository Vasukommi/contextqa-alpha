# src/utils/metadata.py
import json
import os
import logging # Import the logging module
from typing import List, Dict, Any
from src.config import METADATA_FILE # Import the metadata file path from config

# Get a logger instance for this module
logger = logging.getLogger(__name__)


def load_metadata() -> List[Dict[str, Any]]:
    """
    Loads metadata from the centralized metadata file.

    Returns:
        A list of metadata dictionaries.
        Returns an empty list if the file does not exist or is empty.

    Raises:
        json.JSONDecodeError: If the metadata file contains invalid JSON.
        IOError: For other file reading errors.
    """
    # No logging needed here for non-existent or empty file,
    # as it's a normal state for a new application.
    if not os.path.exists(METADATA_FILE) or os.path.getsize(METADATA_FILE) == 0:
        return []

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            # No logging for successful read at this level, maybe debug if needed
            return json.load(f)
    except json.JSONDecodeError as e:
        # Use logger.error instead of print
        logger.error(f"Error decoding metadata JSON from {METADATA_FILE}: {e}")
        raise # Re-raise the exception after logging
    except IOError as e:
        # Use logger.error instead of print
        logger.error(f"Error reading metadata file {METADATA_FILE}: {e}")
        raise # Re-raise the exception after logging
    except Exception as e:
         # Use logger.error instead of print for generic catch-all
         logger.error(f"An unexpected error occurred loading metadata from {METADATA_FILE}: {e}")
         raise # Re-raise the exception after logging


def save_metadata(metadata_list: List[Dict[str, Any]]):
    """
    Saves the metadata list to the centralized metadata file.

    Args:
        metadata_list: The list of metadata dictionaries to save.

    Raises:
        IOError: For file writing errors.
        TypeError: If metadata_list is not JSON serializable.
    """
    # Ensure the directory for the metadata file exists (config.ensure_required_dirs() also does this)
    # No logging needed for os.makedirs with exist_ok=True
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)

    try:
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False) # ensure_ascii=False for non-ASCII chars
        # No logging for successful save at this level, maybe info if needed
        # logger.info(f"Successfully saved metadata to {METADATA_FILE}")
    except IOError as e:
        # Use logger.error instead of print
        logger.error(f"Error saving metadata to {METADATA_FILE}: {e}")
        raise # Re-raise the exception after logging
    except TypeError as e:
         # Use logger.error instead of print
         logger.error(f"Metadata list is not JSON serializable: {e}")
         raise # Re-raise the exception after logging
    except Exception as e:
         # Use logger.error instead of print for generic catch-all
         logger.error(f"An unexpected error occurred saving metadata to {METADATA_FILE}: {e}")
         raise # Re-raise the exception after logging

# Example Usage:
# import logging
# from src.config import setup_logging # Assuming you want to run this directly for testing
# from src.utils.metadata import load_metadata, save_metadata
#
# if __name__ == "__main__":
#    setup_logging() # Configure logging before any logging calls
#    logger.info("Testing metadata utility functions.")
#    try:
#        # Test loading
#        current_metadata = load_metadata()
#        logger.info(f"Loaded metadata: {current_metadata}")
#
#        # Test saving (add a dummy entry)
#        dummy_entry = {"doc_id": "test_doc_123", "file_type": "txt", "upload_time": "2023-10-27T10:00:00Z"}
#        current_metadata.append(dummy_entry)
#        save_metadata(current_metadata)
#        logger.info("Saved dummy metadata entry.")
#
#        # Verify saving by loading again
#        verified_metadata = load_metadata()
#        logger.info(f"Verified metadata after saving: {verified_metadata}")
#
#        # Clean up the dummy entry (optional)
#        verified_metadata = [entry for entry in verified_metadata if entry.get("doc_id") != "test_doc_123"]
#        save_metadata(verified_metadata)
#        logger.info("Cleaned up dummy metadata entry.")
#
#    except Exception as e:
#        logger.error(f"An error occurred during metadata utility test: {e}")