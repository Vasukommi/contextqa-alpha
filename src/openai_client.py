# src/openai_client.py
"""
This module provides a client for interacting with the OpenAI API,
specifically for generating text completions using chat models based on provided context.
It handles API key loading, client initialization, and error handling.
"""
import os
import openai
import logging
from typing import List, Dict, Any, Optional

# Import config values
from src.config import (
    DEFAULT_GPT_MODEL,
    DEFAULT_GPT_TEMPERATURE,
    DEFAULT_GPT_MAX_TOKENS,
    # Optional: Add other config values like TOP_P if needed
    # DEFAULT_GPT_TOP_P = 1.0 # Example
)

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- Custom Exception ---
class OpenAIError(Exception):
    """Custom exception for errors related to OpenAI API interactions."""
    pass

# --- OpenAI Client Initialization (Singleton-like at module level) ---

# Set your OpenAI API key from env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Use logger.critical and raise EnvironmentError if the key is not set.
# This ensures the application fails early if a critical dependency is missing.
if not OPENAI_API_KEY:
    logger.critical("❌ OPENAI_API_KEY environment variable not set. Cannot initialize OpenAI client.")
    # Raising EnvironmentError is appropriate for configuration issues
    raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

# Initialize the OpenAI client once and reuse it.
# This is done at the module level, effectively making it a singleton for this module's usage.
openai_client: Optional[openai.OpenAI] = None # Type hint for clarity

try:
    logger.info("Attempting to initialize OpenAI client.")
    # Instantiate the client using the API key
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Optional: Verify connection/auth by making a small dummy call
    # This adds startup delay but confirms the API key is valid and connection works.
    # try:
    #     logger.debug("Verifying OpenAI client connection...")
    #     openai_client.models.list()
    #     logger.debug("OpenAI client connection verified.")
    # except Exception as verify_error:
    #      logger.warning(f"OpenAI client verification failed: {verify_error}", exc_info=True)
    #      # Decide if verification failure should be critical. For now, just log warning.


    logger.info("✅ OpenAI client initialized successfully.")
except Exception as e:
    # Use logger.critical as client initialization failure is critical
    logger.critical(f"❌ Failed to initialize OpenAI client: {e}", exc_info=True) # Add exc_info
    # If the client fails to initialize, the module is unusable.
    # Raise a custom exception wrapping the original error.
    raise OpenAIError(f"Failed to initialize OpenAI client: {e}") from e # Chain exception


# --- Ask GPT Function ---

def ask_gpt(context: str, question: str, model: str = DEFAULT_GPT_MODEL) -> str:
    """
    Sends a question to OpenAI's GPT model with provided document context
    and returns the generated answer.

    Args:
        context: Relevant extracted content from documents to use as context.
        question: The user's query.
        model: The OpenAI chat model to use (e.g., "gpt-4", "gpt-3.5-turbo").
               Defaults to DEFAULT_GPT_MODEL from config.

    Returns:
        The GPT-generated answer as a string.

    Raises:
        OpenAIError: If the OpenAI client is not initialized, context is empty,
                     the API call fails, or the response structure is unexpected.
    """
    # Check if the client was initialized successfully
    if openai_client is None:
        logger.error("❌ OpenAI client is not initialized. Cannot ask GPT.")
        # Raise custom exception
        raise OpenAIError("OpenAI client is not initialized.")


    # Basic context validation
    if not context or not context.strip():
        logger.warning("ask_gpt called with empty or whitespace-only context.")
        # Raising an error is clearer than returning a specific string
        raise OpenAIError("Cannot answer question: Empty or no relevant context provided.")


    logger.info(f"Sending question to LLM (model: {model}): '{question[:min(len(question), 100)]}...'")
    # Optional: Log context at DEBUG level (be mindful of logging large contexts)
    # logger.debug(f"Context sent to LLM:\n{context}")


    # Prepare messages payload for the chat completion API
    # The system message sets the persona and core instructions.
    # The user message contains the specific task (question + context).
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant working at a company. "
                "Answer the user's question using only the provided context. "
                "If you cannot find the answer within the context, state that you don't have enough information. "
                "Do not use external knowledge or invent facts. "
                "Be concise and directly answer the question based on the context."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n\n{context}\n\nQuestion:\n{question}"
        }
    ]

    try:
        # Use the pre-initialized client instance to create a chat completion
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages, # Use the structured messages list
            temperature=DEFAULT_GPT_TEMPERATURE, # Use config values for tuning
            max_tokens=DEFAULT_GPT_MAX_TOKENS,   # Use config values to control response length
            top_p=1.0, # Example hardcoded, could be in config
            frequency_penalty=0.0, # Example hardcoded
            presence_penalty=0.0,  # Example hardcoded
            # Add a timeout for the API request if desired (highly recommended)
            # timeout=60.0, # Example timeout in seconds
        )

        # Process the response
        # Check if the response structure is as expected and extract the content
        if response and response.choices and response.choices[0].message and response.choices[0].message.content is not None:
            logger.info("✅ Received response from LLM.")
            # Optional: Log the full response object at DEBUG level for inspection
            # logger.debug(f"Full LLM response object: {response}")

            # Optional: Log token usage if available
            # if response.usage:
            #      logger.info(f"Token Usage: Prompt={response.usage.prompt_tokens}, Completion={response.usage.completion_tokens}, Total={response.usage.total_tokens}")

            # Return the cleaned-up response content
            return response.choices[0].message.content.strip()
        else:
            # This indicates an unexpected response format from the API despite no exception
            logger.warning(f"OpenAI returned an unexpected response structure for question: '{question[:min(len(question), 100)]}...'. Response: {response}")
            # Raise custom exception for unexpected format
            raise OpenAIError("Received an unexpected response structure from the LLM.")


    # --- Specific OpenAI API Exception Handling ---
    except openai.APIStatusError as e:
        # Handle API errors specifically (e.g., 401 Unauthorized, 404 Not Found, 429 Rate Limited, 500 Internal Server Error)
        logger.error(f"OpenAI API Status Error ({e.status_code}): {e.response.text}", exc_info=True) # Log traceback
        # Raise custom exception wrapping the API error
        raise OpenAIError(f"OpenAI API error (Status {e.status_code}): {e.response.text}") from e

    except openai.APITimeoutError as e:
        logger.error(f"OpenAI API Timeout Error: {e}", exc_info=True) # Log traceback
        # Raise custom exception wrapping the timeout error
        raise OpenAIError(f"OpenAI API request timed out: {e}") from e

    except openai.APIConnectionError as e:
        logger.error(f"OpenAI API Connection Error: {e}", exc_info=True) # Log traceback
        # Raise custom exception wrapping the connection error
        raise OpenAIError(f"Could not connect to the OpenAI API: {e}") from e

    # --- Catch-all for Other Exceptions during the API Call ---
    except Exception as e:
        # Catch any other unexpected exceptions during the API call process
        logger.error(f"An unexpected error occurred during OpenAI API call for question: '{question[:min(len(question), 100)]}...': {e}", exc_info=True) # Log traceback
        # Raise custom exception wrapping the unexpected error
        raise OpenAIError(f"An unexpected error occurred during LLM interaction: {e}") from e


# --- CLI Test Example ---
if __name__ == "__main__":
    # Configure logging if running this script directly
    # Note: In a real application, logging setup should be handled once at the application entry point (e.g., app.py).
    from src.config import setup_logging
    setup_logging()
    logger.info("--- Running openai_client CLI test ---")

    # Ensure OPENAI_API_KEY is set in your environment before running
    # The module-level initialization will raise EnvironmentError if not set,
    # so the script will exit early if the key is missing.

    # --- Test Case 1: Successful Call ---
    logger.info("\n--- Test Case 1: Successful API Call ---")
    context_success = "The capital of France is Paris. The Eiffel Tower is located in Paris."
    question_success = "What is the capital of France?"
    try:
        logger.info(f"Attempting successful API call with context: '{context_success[:50]}...' and question: '{question_success}'")
        answer = ask_gpt(context_success, question_success)
        logger.info(f"✅ Test Case 1 Successful. Received answer: {answer}")

    except OpenAIError as e:
        logger.error(f"❌ Test Case 1 Failed: Caught expected OpenAIError: {e}")
    except Exception as e:
        logger.error(f"❌ Test Case 1 Failed: Caught unexpected error: {e}", exc_info=True)

    # --- Test Case 2: Empty Context ---
    logger.info("\n--- Test Case 2: Empty Context ---")
    context_empty = ""
    question_empty = "What's the policy?"
    logger.info(f"Attempting API call with empty context (expecting OpenAIError)...")
    try:
        ask_gpt(context_empty, question_empty)
        logger.error("❌ Test Case 2 Failed: OpenAIError was NOT raised for empty context.")
    except OpenAIError as e:
        logger.info(f"✅ Test Case 2 Successful: Caught expected OpenAIError: {e}")
    except Exception as e:
        logger.error(f"❌ Test Case 2 Failed: Caught unexpected error: {e}", exc_info=True)

    # --- Test Case 3: API Error (e.g., invalid model name - might vary based on OpenAI lib version) ---
    # Note: This test might require commenting out the initial critical error check for API_KEY
    # or temporarily providing a key but intentionally using an invalid model name if your OpenAI client allows that.
    # A more reliable way to test specific API errors requires mocking the API.
    # This example is illustrative.
    # logger.info("\n--- Test Case 3: API Error (Illustrative) ---")
    # try:
    #     logger.info("Attempting API call with potentially invalid settings (expecting OpenAIError)...")
    #     # This might fail if "invalid-model" is not a valid model
    #     ask_gpt("some context", "some question", model="invalid-model-name-for-test")
    #     logger.error("❌ Test Case 3 Failed: OpenAIError was NOT raised for simulated API error.")
    # except OpenAIError as e:
    #     logger.info(f"✅ Test Case 3 Successful: Caught expected OpenAIError: {e}")
    #     # Check the error message or wrapped exception type to confirm it's an API error
    # except Exception as e:
    #      logger.error(f"❌ Test Case 3 Failed: Caught unexpected error: {e}", exc_info=True)


    logger.info("\n--- End of openai_client CLI test ---")