# src/agent.py
"""
This module defines the CompanyAgent, which acts as a conversational agent
to answer user questions by retrieving relevant information from a VectorStore
and using an LLM (via openai_client) to synthesize the answer.
"""
import logging
from typing import Dict, Any, List, Optional, TypedDict, Union # Import Union

# Import configuration values
from src.config import DEFAULT_TOP_K, SOURCE_SNIPPET_LENGTH

# Import necessary components
# Expecting VectorStore to raise VectorStoreError and ask_gpt to raise OpenAIError
from src.vector_store import VectorStore, VectorStoreError
from src.openai_client import ask_gpt, OpenAIError

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- Custom Exception ---
class AgentError(Exception):
    """Custom exception for errors occurring within the CompanyAgent."""
    pass

# --- Return Type Structure ---
class AnswerResult(TypedDict):
    """Represents the structure of the response from the CompanyAgent."""
    answer: str
    sources: List[Dict[str, Union[str, int]]] # List of dictionaries for sources

# Define the structure for each source snippet
class SourceSnippet(TypedDict):
    """Represents details for a source snippet."""
    doc_id: str
    chunk_id: str
    chunk_index: int
    snippet_text: str
    # Add other relevant metadata if available, e.g., page_number


# --- Company Agent Class ---

class CompanyAgent:
    """
    Agent for answering questions based on one or more company documents
    using a VectorStore for retrieval and an LLM client for generation.
    Assumes the VectorStore singleton is managed and knowledge is loaded elsewhere.
    """
    def __init__(self):
        """
        Initializes the CompanyAgent by obtaining the VectorStore instance.

        Raises:
            AgentError: If the VectorStore instance cannot be obtained (e.g.,
                        if the SentenceTransformer model failed to load during
                        VectorStore initialization).
        """
        logger.info("Initializing CompanyAgent (multi-document capable).")
        try:
            # Access the VectorStore singleton instance.
            # VectorStore's __init__ handles its own model loading and errors.
            self._vector_store: VectorStore = VectorStore()
            logger.info("Successfully obtained VectorStore instance.")
        except VectorStoreError as e:
            # Catch specific VectorStore initialization errors and wrap in AgentError
            logger.critical(f"❌ Failed to get VectorStore instance during Agent initialization: {e}", exc_info=True)
            raise AgentError(f"Failed to initialize Agent: Could not get VectorStore instance. {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during initialization
            logger.critical(f"❌ An unexpected error occurred during Agent initialization: {e}", exc_info=True)
            raise AgentError(f"An unexpected error occurred during Agent initialization: {e}") from e


    def answer(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        doc_ids: Optional[List[str]] = None
    ) -> AnswerResult:
        """
        Answers a question using multiple documents if available by performing
        retrieval and then generation.

        Args:
            question: The user's question string. Must not be empty or whitespace.
            top_k: The number of top search results to use as context for the LLM.
                   Defaults to DEFAULT_TOP_K from config.
            doc_ids: Optional list of document IDs to perform the search within.
                     If None, searches all documents currently loaded in the VectorStore.

        Returns:
            An AnswerResult dictionary containing the generated 'answer' and
            a list of 'sources' (SnippetSource dictionaries).

        Raises:
            ValueError: If the question is empty or whitespace only.
            AgentError: If an error occurs during the retrieval or generation steps.
        """
        # Input validation
        if not question or not question.strip():
            logger.warning("Agent answer called with empty or whitespace-only question.")
            raise ValueError("Question cannot be empty.")


        logger.info(f"Agent answering: '{question[:min(len(question), 100)]}...' (top_k={top_k}) across docs: {doc_ids or 'ALL'}")

        # --- Step 1: Retrieval ---
        results: List[Dict[str, Any]] # Type hint for clarity
        try:
            # VectorStore.search now returns Dict including chunk metadata
            results = self._vector_store.search(query=question, top_k=top_k, doc_ids=doc_ids)
            logger.info(f"Retrieval successful, found {len(results)} total results.")
        except VectorStoreError as e:
            # Catch specific VectorStore errors during search and wrap in AgentError
            logger.error(f"❌ Error during retrieval: {e}", exc_info=True)
            raise AgentError(f"Retrieval failed: Could not search the knowledge base. {e}") from e
        except Exception as e:
            # Catch any other unexpected retrieval errors
            logger.error(f"❌ An unexpected error occurred during retrieval: {e}", exc_info=True)
            raise AgentError(f"An unexpected error occurred during retrieval: {e}") from e


        if not results:
            logger.warning("No relevant chunks found during retrieval.")
            # Return a specific message indicating no relevant info was found
            return AnswerResult(
                 answer="I couldn't find relevant information in the documents provided.",
                 sources=[]
            )

        # --- Step 2: Context Preparation ---
        # Extract text from the 'chunk' key in the result dictionaries
        context_chunks: List[str] = [r.get("chunk", "") for r in results if isinstance(r, dict)] # Ensure 'chunk' key exists and item is dict
        context = "\n\n".join(context_chunks)

        if not context.strip():
            logger.warning("Empty context generated after extracting text from retrieved chunks.")
            # Return a specific message if retrieved chunks had no text content
            return AnswerResult(
                 answer="The retrieved content was empty or unhelpful.",
                 sources=[]
            )

        logger.debug(f"Prepared context from {len(context_chunks)} chunks (total chars: {len(context)}).")
        # Optional: Log context at DEBUG level (be mindful of size and sensitivity)
        # logger.debug(f"Context sent to LLM:\n{context}")


        # --- Step 3: Generation ---
        response: str
        try:
            logger.info("Calling LLM for generation using context...")
            # ask_gpt now raises OpenAIError
            response = ask_gpt(context, question)
            logger.info("✅ LLM generation successful.")
        except OpenAIError as e:
            # Catch specific OpenAI errors and wrap in AgentError
            logger.error(f"❌ LLM call failed: {e}", exc_info=True)
            raise AgentError(f"LLM generation failed: Could not get response from AI model. {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during LLM call
            logger.error(f"❌ An unexpected error occurred during LLM call: {e}", exc_info=True)
            raise AgentError(f"An unexpected error occurred during LLM interaction: {e}") from e


        # --- Step 4: Source Snippet Formatting ---
        source_snippets: List[SourceSnippet] = []
        for r in results:
            # Safely get metadata from the result dictionary (provided by VectorStore.search)
            doc_id = r.get('doc_id', 'unknown')
            chunk_id = r.get('chunk_id', 'N/A') # Include chunk_id
            chunk_index = r.get('chunk_index', -1) # Include chunk_index
            full_chunk_text = r.get("chunk", "") # Get the full text again for consistent snippet generation

            # Generate snippet text
            snippet_text = full_chunk_text[:SOURCE_SNIPPET_LENGTH]
            if len(full_chunk_text) > SOURCE_SNIPPET_LENGTH:
                snippet_text += "..."

            # Append SourceSnippet dictionary
            source_snippets.append(SourceSnippet(
                doc_id=doc_id,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                snippet_text=snippet_text.strip() # Strip snippet text
                # Add other metadata from the chunk result if available and relevant
                # e.g., "page_number": r.get("page_number")
            ))
        logger.debug(f"Formatted {len(source_snippets)} source snippets.")


        # --- Step 5: Return Final Result ---
        return AnswerResult(
            answer=response.strip(), # Ensure final answer is stripped
            sources=source_snippets
        )


# --- Example usage (CLI test) ---
if __name__ == "__main__":
    # Configure logging if running this script directly
    # Note: In a real application, logging setup should be handled once at the application entry point (e.g., app.py).
    from src.config import setup_logging, ensure_required_dirs
    from src.vector_store import VectorStore # Need VectorStore to load knowledge
    import os # Needed for example file paths if manually setting up

    setup_logging()
    # ensure_required_dirs() # Uncomment if running this script stand-alone and need directories


    logger.info("--- Running CompanyAgent CLI test ---")

    # --- Setup for Test ---
    # !!! IMPORTANT !!!
    # For this test to work, you MUST have:
    # 1. The VectorStore singleton class accessible and its __init__ successfully loading the model.
    # 2. Knowledge (embeddings, chunks, index) loaded into the VectorStore for the test_doc_id(s).
    # This typically means running the full processing/embedding pipeline for documents
    # and ensuring VectorStore().load_document_knowledge() is called for those doc_ids
    # somewhere before this test runs. This script only *uses* the loaded knowledge.

    # Replace with doc_ids for which you have loaded knowledge in your VectorStore setup
    test_doc_id_single = "your_test_doc_id_here" # <<< REPLACE with an actual doc_id
    # Example for multi-doc test, replace with actual loaded doc IDs
    test_doc_ids_multi = ["doc_id_a", "doc_id_b"] # <<< REPLACE


    # Attempt to initialize VectorStore (loads model) and load knowledge if doc_ids are set up
    vs_instance: Optional[VectorStore] = None
    loaded_doc_ids_for_test: List[str] = []

    try:
        logger.info("Attempting to get VectorStore instance (loads model)...")
        vs_instance = VectorStore()
        logger.info("VectorStore instance obtained.")

        # Load knowledge for test documents if they are configured and files exist
        potential_test_doc_ids = [test_doc_id_single] + test_doc_ids_multi
        for doc_id_to_load in potential_test_doc_ids:
             if doc_id_to_load and doc_id_to_load != "your_test_doc_id_here" and doc_id_to_load not in ["doc_id_a", "doc_id_b"]:
                 # Check if the necessary files for this doc_id exist before attempting to load
                 emb_path = os.path.join(VectorStore.KNOWLEDGE_BASE_DIR, f"{doc_id_to_load}_embeddings.npy") # Access config via class or import
                 chunk_path = os.path.join(VectorStore.PROCESSED_DATA_DIR, f"{doc_id_to_load}_chunks.json") # Access config via class or import
                 index_path = os.path.join(VectorStore.KNOWLEDGE_BASE_DIR, f"{doc_id_to_load}_index.faiss") # Access config via class or import

                 if os.path.exists(emb_path) and os.path.exists(chunk_path) and os.path.exists(index_path):
                      try:
                          logger.info(f"Attempting to load knowledge for '{doc_id_to_load}' for test...")
                          vs_instance.load_document_knowledge(doc_id_to_load)
                          logger.info(f"Knowledge for '{doc_id_to_load}' loaded for test.")
                          loaded_doc_ids_for_test.append(doc_id_to_load)
                      except VectorStoreError as load_error:
                           logger.error(f"Failed to load knowledge for {doc_id_to_load} for test: {load_error}", exc_info=True)
                      except Exception as load_error:
                           logger.error(f"Unexpected error loading knowledge for {doc_id_to_load} for test: {load_error}", exc_info=True)
                 else:
                      logger.warning(f"Skipping loading knowledge for '{doc_id_to_load}' for test: Knowledge files not found.")
             else:
                  logger.debug(f"Skipping attempt to load knowledge for placeholder doc_id: {doc_id_to_load}")


    except VectorStoreError as vs_init_error:
        logger.critical(f"Failed to initialize VectorStore for test: {vs_init_error}")
        vs_instance = None # Ensure vs_instance is None if init fails
    except Exception as unexpected_init_error:
         logger.critical(f"Unexpected error during VectorStore initialization for test: {unexpected_init_error}", exc_info=True)
         vs_instance = None # Ensure vs_instance is None if init fails


    # --- Initialize Agent ---
    agent_instance: Optional[CompanyAgent] = None
    if vs_instance and loaded_doc_ids_for_test:
        try:
             logger.info("\nAttempting to initialize CompanyAgent...")
             agent_instance = CompanyAgent() # Should succeed if VectorStore init was okay
             logger.info("CompanyAgent initialized.")
        except AgentError as agent_init_error:
             logger.error(f"Failed to initialize CompanyAgent: {agent_init_error}")
             agent_instance = None
        except Exception as unexpected_agent_init_error:
             logger.error(f"Unexpected error during Agent initialization: {unexpected_agent_init_error}", exc_info=True)
             agent_instance = None
    else:
         if not vs_instance:
              logger.error("Skipping Agent initialization and tests: VectorStore failed to initialize.")
         elif not loaded_doc_ids_for_test:
              logger.error("Skipping Agent initialization and tests: No knowledge loaded for testing.")


    # --- Run Agent Tests ---
    if agent_instance and loaded_doc_ids_for_test:
        # Use the first loaded doc_id for single-doc test
        test_doc_id_for_search = loaded_doc_ids_for_test[0]

        # Test Case 1: Successful Answer (Single Document)
        logger.info(f"\n--- Test Case 1: Successful Answer (Single Doc: {test_doc_id_for_search}) ---")
        test_question_single = "What is the company?" # Replace with a relevant query for your test doc
        try:
            logger.info(f"Asking agent about doc '{test_doc_id_for_search}': '{test_question_single}'")
            answer_result: AnswerResult = agent_instance.answer(question=test_question_single, doc_ids=[test_doc_id_for_search], top_k=3)

            logger.info("✅ Test Case 1 Successful.")
            logger.info(f"Answer: {answer_result['answer']}")
            logger.info("Sources:")
            for source in answer_result['sources']:
                 logger.info(f"  - Doc ID: {source.get('doc_id')}, Chunk ID: {source.get('chunk_id')}, Index: {source.get('chunk_index')}, Snippet: '{source.get('snippet_text')}'")

        except (ValueError, AgentError, Exception) as e:
            logger.error(f"❌ Test Case 1 Failed: Caught error during answer: {e}", exc_info=True)


        # Test Case 2: Answer with No Relevant Chunks
        logger.info("\n--- Test Case 2: Answer with No Relevant Chunks ---")
        test_question_no_results = "Tell me about space travel?" # Assume this isn't in your company docs
        try:
            logger.info(f"Asking agent an irrelevant question: '{test_question_no_results}'")
            answer_result_no_results: AnswerResult = agent_instance.answer(question=test_question_no_results, doc_ids=loaded_doc_ids_for_test, top_k=3)

            logger.info("✅ Test Case 2 Successful.")
            logger.info(f"Answer: {answer_result_no_results['answer']}") # Expecting "I couldn't find relevant information..."
            logger.info(f"Sources: {answer_result_no_results['sources']}") # Expecting empty list

        except (ValueError, AgentError, Exception) as e:
            logger.error(f"❌ Test Case 2 Failed: Caught error during answer with no results: {e}", exc_info=True)


        # Test Case 3: Empty Question
        logger.info("\n--- Test Case 3: Empty Question ---")
        test_question_empty = ""
        logger.info("Asking agent with an empty question (expecting ValueError)...")
        try:
            agent_instance.answer(question=test_question_empty, doc_ids=loaded_doc_ids_for_test)
            logger.error("❌ Test Case 3 Failed: ValueError was NOT raised for empty question.")
        except ValueError as e:
            logger.info(f"✅ Test Case 3 Successful: Caught expected error: {e}")
        except (AgentError, Exception) as e:
             logger.error(f"❌ Test Case 3 Failed: Caught unexpected error: {e}", exc_info=True)


        # Add more test cases here, e.g., multi-document search if you set up more docs

    else:
        logger.error("\nSkipping Agent tests because Agent could not be initialized or no knowledge was loaded.")


    logger.info("\n--- End of CompanyAgent CLI test ---")