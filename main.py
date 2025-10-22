import streamlit as st
import tempfile
import os
import traceback

# Import Rag classes.
from rag import RAG

#Display all messages stored in session_state
def display_messages():
  for message in st.session_state.messages:
    with st.chat_message(message['role']):
      st.markdown(message['content'])

# Start a new session
def start_new_session():
  """Delete the current collection and start a fresh session"""
  if "assistant" in st.session_state and st.session_state["assistant"]:
    # Delete the old collection
    st.session_state["assistant"].delete_collection()
  
  # Create new RAG instance and clear messages
  st.session_state["assistant"] = RAG()
  st.session_state.messages = []
  st.success("New session started! Upload documents to begin.")
      
def process_file():
  st.session_state["assistant"].clear()
  st.session_state.messages = []

  for file in st.session_state["file_uploader"]:
    # Store the file at tem location
    # of your system to feed to our vector storage.
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      tf.write(file.getbuffer())
      file_path = tf.name

    #feed the file to the vector storage.
    with st.session_state["feeder_spinner"], st.spinner("Uploading the file"):
      st.session_state["assistant"].feed(file_path)
    os.remove(file_path)
    

def process_input():
  # See if user has typed in any message and assign to prompt.
  if prompt := st.chat_input("Ask a question about your documents..."):
    with st.chat_message("user"):
      st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response and write back to the chat container.
    with st.spinner("Thinking..."):
      response = st.session_state["assistant"].ask(prompt)
    with st.chat_message("assistant"):
      st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


def main():
  st.title("DocueMentor - RAG Assistant")
  st.markdown("Upload PDF documents and ask questions about their content.")

  # Initialize the session_state.
  if len(st.session_state) == 0:
    try:
      st.session_state["assistant"] = RAG()
      st.session_state.messages = []
    except Exception as e:
      st.error(f"⚠️ **Error initializing RAG system**")
      st.error(f"**Error message:** {str(e)}")
      
      with st.expander("🔍 Click to see detailed error trace"):
        st.code(traceback.format_exc())
      
      st.warning("""
      **Common issues:**
      1. Missing `.env` file with `XAI_API_KEY`
      2. Ollama is not running (required for embeddings)
      3. Missing required model: `nomic-embed-text:latest`
      
      **Quick fix:**
      - Create a `.env` file with your XAI_API_KEY
      - Run: `ollama serve` in a separate terminal
      - Run: `ollama pull nomic-embed-text:latest`
      """)
      return

  # Sidebar for session management
  with st.sidebar:
    st.header("Session Management")
    
    # Display session info
    if st.session_state.get("assistant"):
      collection_info = st.session_state["assistant"].get_collection_info()
      st.info(f"**Session ID:** {collection_info['collection_name'][:8]}...")
      st.metric("Documents Uploaded", collection_info['document_count'])
    
    # New Session button
    if st.button("🔄 Start New Session", use_container_width=True):
      start_new_session()
      st.rerun()
    
    st.divider()
    
    st.markdown("""
    ### How to use:
    1. Upload PDF documents
    2. Ask questions about them
    3. Start a new session to clear everything
    """)

  # Code for file upload functionality.
  st.file_uploader(
      "Upload PDF documents",
      type = ["pdf"],
      key = "file_uploader",
      on_change=process_file,
      label_visibility="visible",
      accept_multiple_files=True,
    )

  st.session_state["feeder_spinner"] = st.empty()

  display_messages()
  process_input()

if __name__ == "__main__":
  main()