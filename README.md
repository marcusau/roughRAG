# DocueMentor - RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents and ask questions about their content. Built with Streamlit, ChromaDB, Ollama embeddings, and XAI's Grok model.

## Features

- 📄 Upload multiple PDF documents
- 🔍 Ask questions about document content
- 🧠 Powered by XAI's Grok-3-mini model
- 📊 Vector-based document retrieval
- 💬 Interactive chat interface
- 🔄 Session management

## Prerequisites

- Python 3.8 or higher
- XAI API key
- Ollama installed locally

## Setup Instructions

### 1. Set up Virtual Environment

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

### 3. Create Environment File

Create a `.env` file in the project root directory and add your XAI API key:

```bash
# Create .env file
touch .env
```

Add the following content to your `.env` file:

```env
XAI_API_KEY=your_xai_api_key_here
```

**Note:** Replace `your_xai_api_key_here` with your actual XAI API key. You can get one from [XAI's website](https://x.ai/).

### 4. Install Ollama and Download Embedding Model

#### Install Ollama

Visit [Ollama's website](https://ollama.ai/) and download the appropriate version for your operating system.

#### Download the Embedding Model

After installing Ollama, download the required embedding model:

```bash
ollama pull nomic-embed-text:latest
```

#### Start Ollama Service

In a separate terminal window, start the Ollama service:

```bash
ollama serve
```

**Important:** Keep this terminal window open while using the application, as the embedding service needs to be running.

### 5. Run the Application

Start the Streamlit application:

```bash
STREAMLIT_BROWSER_GATHERUSAGESTATS=false STREAMLIT_SERVER_HEADLESS=true streamlit run main.py --server.port 8502
```

### 6. Access the Application

Open your web browser and navigate to:

```
http://localhost:8502
```

## Usage

1. **Upload Documents**: Use the file uploader to select one or more PDF documents
2. **Ask Questions**: Type your questions in the chat input field
3. **Start New Session**: Use the sidebar to clear the current session and start fresh

## Troubleshooting

### Common Issues

1. **"Error initializing RAG system"**
   - Ensure your `.env` file exists with a valid `XAI_API_KEY`
   - Make sure Ollama is running (`ollama serve`)
   - Verify the embedding model is downloaded (`ollama pull nomic-embed-text:latest`)

2. **Embedding errors**
   - Check that Ollama service is running
   - Verify the `nomic-embed-text:latest` model is available

3. **API key issues**
   - Double-check your XAI API key in the `.env` file
   - Ensure the key is valid and has sufficient credits

### Getting Help

If you encounter issues:

1. Check the error messages in the Streamlit interface
2. Verify all prerequisites are installed correctly
3. Ensure all services (Ollama) are running
4. Check that your API key is valid

## Project Structure

```
MarcusRAG/
├── main.py                 # Streamlit application
├── rag.py                  # RAG implementation
├── provider.py             # LLM and embedding providers
├── chunk_vector_store.py   # Vector storage management
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── documents/              # Sample documents
├── utils/                  # Utility functions
└── scripts/               # Helper scripts
```

## Dependencies

Key dependencies include:
- `streamlit` - Web application framework
- `chromadb` - Vector database
- `langchain-ollama` - Ollama integration
- `langchain-xai` - XAI integration
- `pymupdf4llm` - PDF processing
- `python-dotenv` - Environment variable management

## License

This project is for educational and research purposes.
