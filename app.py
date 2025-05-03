import streamlit as st
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import backend components and their custom exceptions
from src.upload_handler import handle_upload, ensure_upload_dirs
from src.processor import process_document, DocumentProcessingError
from src.embedder import embed_document, EmbeddingError
from src.agent import CompanyAgent, AgentError
from src.vector_store import VectorStore, VectorStoreError
from src.utils.metadata import load_metadata, save_metadata # Import save_metadata if needed directly
from src.config import (
    METADATA_FILE,
    UPLOAD_DIR,
    PROCESSED_DATA_DIR,
    KNOWLEDGE_BASE_DIR,
    TEMP_UPLOAD_DIR, # Use TEMP_UPLOAD_DIR from config
    setup_logging,
    DEFAULT_TOP_K # Import DEFAULT_TOP_K if used in Ask section directly
)

# --- Setup Logging (Should be called once at the application entry point) ---
setup_logging()
logger = logging.getLogger(__name__)
logger.info("Streamlit app initializing.")

# --- Ensure Required Directories Exist ---
# Call this once at the very start to ensure all necessary directories are in place
try:
    ensure_upload_dirs() # This now includes TEMP_UPLOAD_DIR and DATA_DIR
    # Note: ensure_required_dirs in config also covers PROCESSED_DATA_DIR and KNOWLEDGE_BASE_DIR
    # Calling it here or in config's setup is fine, just ensure it happens.
    # Let's rely on ensure_upload_dirs from upload_handler for app startup.
    # Or import ensure_required_dirs from config and call that. Let's use the one from config.
    from src.config import ensure_required_dirs as ensure_all_required_dirs
    ensure_all_required_dirs()
    logger.info("App: Ensured all required directories exist.")
except Exception as e:
    logger.critical(f"App: Critical error during directory setup: {e}", exc_info=True)
    st.error(f"Critical error during directory setup: {e}. Please check logs.")
    st.stop() # Stop the app if directories cannot be set up


st.set_page_config(page_title="Flynkle Alpha v2", layout="wide")
st.title("🤖 Flynkle — Internal Alpha v2")


# --- Cached metadata loader ---
@st.cache_data
def cached_load_metadata() -> List[Dict[str, Any]]:
    """Loads metadata using Streamlit's data caching."""
    logger.info("Loading metadata via st.cache_data.")
    try:
        metadata = load_metadata()
        logger.info(f"Loaded {len(metadata)} metadata entries.")
        return metadata
    except FileNotFoundError:
        logger.info("Metadata file not found. Returning empty list.")
        return [] # Return empty list if metadata file doesn't exist yet
    except Exception as e:
        logger.error(f"❌ Failed to load metadata: {e}", exc_info=True)
        st.error(f"Failed to load document metadata. Error: {e}")
        # Decide if metadata loading failure should stop the app.
        # For now, return empty and log error.
        return []

# --- Cached VectorStore Initialization and Initial Knowledge Loading ---
@st.cache_resource
def get_vector_store(initial_metadata: List[Dict[str, Any]]):
    """
    Initializes the VectorStore and loads knowledge for existing documents
    from metadata using Streamlit's resource caching.
    """
    logger.info("Initializing VectorStore via st.cache_resource.")
    try:
        # VectorStore __init__ loads the SentenceTransformer model
        vs = VectorStore()
        logger.info("VectorStore instance created.")

        # Load knowledge for documents found in the initial metadata
        logger.info(f"Attempting to load knowledge for {len(initial_metadata)} existing documents.")
        for entry in initial_metadata:
            doc_id = entry.get("doc_id")
            if doc_id:
                try:
                    # VectorStore.load_document_knowledge handles if already loaded
                    vs.load_document_knowledge(doc_id)
                    logger.debug(f"Attempted to load knowledge for doc_id: {doc_id}")
                except VectorStoreError as load_error:
                    logger.warning(f"Skipping loading knowledge for doc_id {doc_id} due to error: {load_error}", exc_info=True)
                    # Log warning but continue, as this specific doc might be missing files
                except Exception as load_error:
                    logger.warning(f"Skipping loading knowledge for doc_id {doc_id} due to unexpected error: {load_error}", exc_info=True)
            else:
                 logger.warning(f"Skipping loading knowledge for metadata entry missing doc_id: {entry}")

        logger.info("✅ VectorStore initialized and initial knowledge loading attempted.")
        return vs
    except VectorStoreError as e:
        logger.critical(f"❌ Failed to initialize VectorStore: {e}", exc_info=True)
        st.error(f"Failed to initialize the knowledge base. Error: {e}. Please check logs.")
        st.stop() # Stop the app if VectorStore init fails
    except Exception as e:
        logger.critical(f"❌ Failed to initialize VectorStore due to unexpected error: {e}", exc_info=True)
        st.error(f"Failed to initialize the knowledge base. Unexpected Error: {e}. Please check logs.")
        st.stop()

# Load initial metadata for the cached VectorStore function
initial_metadata_for_vs = cached_load_metadata()
# Get the cached VectorStore instance, passing initial metadata
vector_store = get_vector_store(initial_metadata_for_vs)


# --- File Upload Section ---
st.header("📤 Upload Documents")
# Track uploaded file names in session state to know which ones have been uploaded in the current session
if 'uploaded_file_names_in_session' not in st.session_state:
    st.session_state['uploaded_file_names_in_session'] = []
    logger.debug("Initialized st.session_state['uploaded_file_names_in_session']")


uploaded_files = st.file_uploader(
    "Upload one or more files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="file_uploader"
)

# Identify new files that have been selected in the uploader
# Compare current selection to the list of file names already processed in this session
current_selected_file_names = [file.name for file in uploaded_files]
new_files_selected = [
    file for file in uploaded_files
    if file.name not in st.session_state['uploaded_file_names_in_session']
]


if new_files_selected:
    logger.info(f"Detected {len(new_files_selected)} new files selected by user to process.")
    with st.status("Processing new files...", expanded=True) as status:
        processed_successfully_count = 0
        for file in new_files_selected:
            file_name = file.name
            temp_path: Optional[str] = None
            doc_id: Optional[str] = None

            try:
                status.write(f"Starting processing for **{file_name}**...")
                logger.info(f"Starting processing pipeline for file: {file_name}")

                # Step 1: Save temporary file
                os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True) # Ensure temp dir exists
                temp_path = os.path.join(TEMP_UPLOAD_DIR, file_name)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                logger.info(f"Saved temporary file: {temp_path}")
                status.write(f"Saved temporary file: **{file_name}**")

                # Step 2: Handle Upload (assign doc_id, copy to UPLOAD_DIR, update metadata)
                status.write(f"Handling upload for **{file_name}**...")
                doc_id = handle_upload(temp_path) # handle_upload now raises exceptions
                status.write(f"Assigned Doc ID: `{doc_id}`")
                logger.info(f"Assigned Doc ID: {doc_id} for {file_name}")

                # Step 3: Process Document (parse, chunk)
                status.write(f"Processing document **{file_name}** (chunking)...")
                process_document(doc_id) # process_document now raises exceptions
                status.write(f"Document processed: **{file_name}**")
                logger.info(f"Document processed for doc_id: {doc_id}")

                # Step 4: Embed Document (generate embeddings, build index)
                status.write(f"Embedding document **{file_name}**...")
                embed_document(doc_id) # embed_document now raises exceptions
                status.write(f"Document embedded: **{file_name}**")
                logger.info(f"Document embedded for doc_id: {doc_id}")

                # Step 5: Load Knowledge into VectorStore
                status.write(f"Loading knowledge into VectorStore for **{file_name}**...")
                vector_store.load_document_knowledge(doc_id) # load_document_knowledge now raises exceptions
                status.write(f"Knowledge loaded: **{file_name}**")
                logger.info(f"Knowledge loaded into VectorStore for doc_id: {doc_id}")


                logger.info(f"✅ Finished processing pipeline for: {file_name}")
                status.write(f"✅ Successfully processed and loaded: **{file_name}**")

                # Add file name to session state list of processed files
                st.session_state['uploaded_file_names_in_session'].append(file_name)
                processed_successfully_count += 1

                # Clear cached metadata and vector store after successful processing
                # This forces reload and ensures new documents appear in the list and are loaded
                cached_load_metadata.clear()
                # get_vector_store.clear() # Clearing VectorStore cache forces model reload - usually not desired unless config changes
                                           # Reloading metadata and relying on vs.load_document_knowledge in cached_get_vector_store is better


            # --- Specific Exception Handling for Processing Pipeline ---
            except (FileNotFoundError, ValueError, IOError, json.JSONDecodeError) as e:
                logger.error(f"❌ File/Metadata Error processing {file_name}: {e}", exc_info=True)
                status.error(f"❌ Error processing **{file_name}**: File or metadata issue. {e}")
            except DocumentProcessingError as e:
                 logger.error(f"❌ Document Processing Error for {file_name}: {e}", exc_info=True)
                 status.error(f"❌ Error processing **{file_name}**: Document processing failed. {e}")
            except EmbeddingError as e:
                 logger.error(f"❌ Embedding Error for {file_name}: {e}", exc_info=True)
                 status.error(f"❌ Error embedding **{file_name}**: Embedding failed. {e}")
            except VectorStoreError as e:
                 logger.error(f"❌ Vector Store Error for {file_name}: {e}", exc_info=True)
                 status.error(f"❌ Error loading knowledge for **{file_name}**: Vector store failed. {e}")
            except Exception as e:
                logger.error(f"❌ An unexpected error occurred during pipeline for {file_name}: {e}", exc_info=True)
                status.error(f"❌ An unexpected error occurred while processing **{file_name}**. {e}")

            finally:
                # Clean up the temporary file regardless of success or failure
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                        logger.debug(f"Cleaned up temporary file: {temp_path}")
                    except OSError as cleanup_e:
                        logger.warning(f"Failed to remove temporary file {temp_path}: {cleanup_e}", exc_info=True)

        status.update(label=f"✅ Processed {processed_successfully_count} new file(s)!", state="complete", expanded=False)
        # Rerun to update the list of available documents in the Ask section
        st.rerun()


# --- Display Available Documents ---
st.markdown("---")
st.header("📚 Available Documents")

# Reload metadata to show newly added files if any
metadata_after_upload = cached_load_metadata()

if not metadata_after_upload:
    st.info("No documents uploaded yet.")
else:
    # Filter metadata to only show documents that have actually been loaded into the VectorStore
    loaded_doc_ids = vector_store._knowledge.keys() # Access loaded knowledge keys
    loaded_documents_metadata = [m for m in metadata_after_upload if m.get("doc_id") in loaded_doc_ids]

    if not loaded_documents_metadata:
        st.info("Documents have been uploaded, but none were successfully processed and loaded.")
    else:
        st.write(f"Successfully processed and loaded {len(loaded_documents_metadata)} document(s):")
        for doc_meta in loaded_documents_metadata:
            original_filename = doc_meta.get("original_filename", "Unknown file")
            upload_time_str = doc_meta.get("upload_time", "Unknown time")
            try:
                 upload_time = datetime.fromisoformat(upload_time_str.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M UTC')
            except ValueError:
                 upload_time = upload_time_str # Fallback if parsing fails
            st.markdown(f"- **{original_filename}** (Uploaded: {upload_time})")


# --- Ask Loaded Documents Section ---
st.markdown("---")
st.header("🧠 Ask Your Documents")

# Use the list of document IDs that were successfully loaded into the VectorStore
available_doc_ids_for_query = list(vector_store._knowledge.keys())

if not available_doc_ids_for_query:
    st.info("No documents successfully loaded into the knowledge base to ask questions from.")
else:
    # Allow user to select specific documents to ask (Optional feature)
    # selected_doc_ids_for_query = st.multiselect(
    #     "Select documents to ask (optional, leave blank to ask all loaded)",
    #     options=available_doc_ids_for_query,
    #     format_func=lambda doc_id: next((m.get("original_filename", doc_id) for m in metadata_after_upload if m.get("doc_id") == doc_id), doc_id),
    #     key="doc_selector"
    # )
    # Use all available docs if none are selected
    # query_doc_ids = selected_doc_ids_for_query if selected_doc_ids_for_query else available_doc_ids_for_query

    # For simplicity initially, always ask all loaded documents
    query_doc_ids = available_doc_ids_for_query


    question = st.text_input("Ask a question about the available documents:", key="global_question_input")

    # Use a form to group question input and button for better interaction
    with st.form("question_form"):
        submitted = st.form_submit_button("Ask")

        if submitted and question.strip():
            with st.spinner("Thinking..."):
                try:
                    # Initialize Agent instance (it will get the cached VectorStore)
                    agent = CompanyAgent() # Agent init might raise AgentError if VS init failed earlier

                    # Call the agent's answer method (it now raises AgentError or ValueError)
                    # Pass the list of doc_ids derived from successfully loaded knowledge
                    answer_result = agent.answer(
                         question=question,
                         doc_ids=query_doc_ids, # Pass the list of doc_ids to search
                         top_k=DEFAULT_TOP_K # Use config value
                    )

                    st.subheader("📜 Answer")
                    # The answer key is guaranteed to exist by AnswerResult TypedDict
                    st.markdown(answer_result['answer'])

                    if answer_result['sources']:
                        st.subheader("🔍 Sources")
                        # source['snippet_text'] and other keys are guaranteed by SourceSnippet TypedDict
                        for i, source in enumerate(answer_result['sources'], 1):
                            st.markdown(f"**{i}.** Doc: `{source.get('doc_id', 'N/A')}` | Chunk: `{source.get('chunk_id', 'N/A')}` (Index {source.get('chunk_index', 'N/A')})")
                            st.text(source.get('snippet_text', '')) # Use st.text for preformatted snippet text


                # --- Specific Exception Handling for Question Answering ---
                except ValueError as e:
                    logger.warning(f"Validation Error during answer: {e}")
                    st.warning(f"Please enter a valid question. {e}")
                except AgentError as e:
                    logger.error(f"❌ Agent Error during answer: {e}", exc_info=True)
                    st.error(f"An error occurred while generating the answer. {e}")
                except Exception as e:
                    logger.error(f"❌ An unexpected error occurred during answering: {e}", exc_info=True)
                    st.error(f"An unexpected error occurred. {e}. Please check logs.")
        elif submitted and not question.strip():
             st.warning("Please enter a question.")

logger.info("Streamlit app finished execution cycle.")