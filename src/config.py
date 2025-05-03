# src/config.py
import os
import logging
import sys
from dotenv import load_dotenv # Import load_dotenv

# --- Load Environment Variables ---
# This should be done early to make variables available for configuration
load_dotenv()

# --- Basic Configuration ---

# Determine the base directory of the project
# Assumes config.py is in backend/src/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --- Directory Definitions ---
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge")
TEMP_UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads") # Added based on folder structure

# Define specific file paths
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

# --- Application Settings (Configurable via Environment Variables with Defaults) ---

# Chunking settings
# Get from env or use default
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
SOURCE_SNIPPET_LENGTH = int(os.getenv("SOURCE_SNIPPET_LENGTH", 200)) # Used in agent to show source snippets

# Embedding settings
# Get from env or use default
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

# OpenAI settings
# Get from env or use default
DEFAULT_GPT_MODEL = os.getenv("GPT_MODEL", "gpt-3.5-turbo")
DEFAULT_GPT_TEMPERATURE = float(os.getenv("GPT_TEMPERATURE", 0.2))
DEFAULT_GPT_MAX_TOKENS = int(os.getenv("GPT_MAX_TOKENS", 500))

# IMPORTANT: OpenAI API Key should NEVER be hardcoded.
# Load it from environment variables.
# Example: OPENAI_API_KEY="your_api_key_here" in your .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Use logging instead of print for consistency
    logging.warning("OPENAI_API_KEY not found in environment variables.")
    # Depending on your app's requirements, you might want to raise an error or exit here.
    # raise ValueError("OPENAI_API_KEY is not set.")

# Retriever settings
DEFAULT_TOP_K = int(os.getenv("RETRIEVER_TOP_K", 3))

# --- Logging Settings ---
# Get logging level from env (e.g., "INFO", "DEBUG") or use default
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
# Map string level to logging constant
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- Directory Initialization Function ---

def ensure_required_dirs() -> None:
    """Ensures all necessary directories defined in config exist."""
    required_dirs = [
        UPLOAD_DIR,
        PROCESSED_DATA_DIR,
        KNOWLEDGE_BASE_DIR,
        TEMP_UPLOAD_DIR, # Ensure temp upload dir exists
        os.path.dirname(METADATA_FILE), # Ensure data/ exists
    ]
    logger = logging.getLogger(__name__) # Get logger for this module

    logger.info("Ensuring required directories exist...")
    try:
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
        logger.info("✅ Ensured required directories exist.")
    except OSError as e:
        logger.error(f"❌ Error ensuring required directories exist: {e}", exc_info=True)
        # Depending on severity, you might want to exit or raise the error
        # raise

# --- Logging Setup Function ---

def setup_logging() -> None:
    """Configures the basic logging for the application."""
    # Get the root logger
    root_logger = logging.getLogger()
    # Set the overall minimum level based on LOG_LEVEL
    root_logger.setLevel(LOG_LEVEL)

    # Prevent duplicate handlers if setup is called multiple times
    if not root_logger.handlers:
        # Create a console handler, outputting to stdout
        console_handler = logging.StreamHandler(sys.stdout)
        # Set the level for the handler (optional, typically root level is sufficient)
        # console_handler.setLevel(LOG_LEVEL)

        # Create and set the formatter
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        console_handler.setFormatter(formatter)

        # Add the handler to the root logger
        root_logger.addHandler(console_handler)

        # Optional: Add a file handler (uncomment and configure if needed)
        # try:
        #     log_file_path = os.path.join(BASE_DIR, 'app.log')
        #     file_handler = logging.FileHandler(log_file_path)
        #     file_handler.setFormatter(formatter)
        #     root_logger.addHandler(file_handler)
        #     logging.info(f"Logging to console and file: {log_file_path}")
        # except Exception as e:
        #     logging.error(f"Could not set up file logging: {e}")


    # Use logging instead of print for consistency
    logging.info(f"✅ Logging configured with level: {logging.getLevelName(LOG_LEVEL)}")


# --- Example Usage (for reference, keep commented out) ---
# from src.config import PROCESSED_DATA_DIR, DEFAULT_CHUNK_SIZE
# chunk_path = os.path.join(PROCESSED_DATA_DIR, f"{doc_id}_chunks.json")

# To use logging in another module:
# import logging
# logger = logging.getLogger(__name__) # Get logger specific to the module
# logger.info("This is an info message from another module")
# logger.error("This is an error message with context", exc_info=True) # exc_info=True logs traceback


# --- Initialization Calls (Optional - consider calling elsewhere) ---
# Calling setup_logging() here means logging is configured as soon as config.py is imported.
# This is often desired.
setup_logging()

# Calling ensure_required_dirs() here means directories are created on import.
# Consider if this is the desired behavior or if it should be part of your application's startup logic (e.g., in app.py).
# ensure_required_dirs()