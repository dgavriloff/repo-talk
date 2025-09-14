import os
import shutil
import argparse
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import logging

from git import Repo
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE_FILE = Path("prompt_template.txt")
REWRITE_PROMPT_TEMPLATE_FILE = Path("rewrite_prompt_template.txt")

# --- INITIAL SETUP ---
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_interactions.log")
    ]
)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY must be set in the .env file")

# --- CONFIGURATION ---
DB_FILE = "cli_demo.db"
REPO_CACHE_DIR = Path("repo_cache")
VECTOR_STORE_DIR = Path("vector_stores")

# --- DATABASE LOGIC ---
def init_db():
    """Initializes the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS repositories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        github_url TEXT NOT NULL UNIQUE
    )
    """
    )
    conn.commit()
    conn.close()

# --- CORE FUNCTIONS ---
def index_repository(repo_url: str):
    """Clones a repo, chunks code, creates embeddings, and saves them to disk."""
    repo_name = repo_url.split("/")[-1]
    logging.info(f"Starting indexing for '{repo_name}'...")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO repositories (name, github_url) VALUES (?, ?)", (repo_name, repo_url))
    conn.commit()
    conn.close()

    local_path = REPO_CACHE_DIR / repo_name
    if local_path.exists():
        shutil.rmtree(local_path)
    
    logging.info(f"1/4: Cloning repository from {repo_url}...")
    Repo.clone_from(repo_url, local_path, depth=1)

    logging.info("2/4: Loading documents...")
    documents = []
    for pattern in ["*.py", "*.md", "*.js", "*.ts", "*.tsx"]:
        for file_path in local_path.rglob(pattern):
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
            except Exception:
                logging.warning(f"  - Skipping unreadable file: {file_path}")

    if not documents:
        logging.info("No indexable documents found. Aborting.")
        return

    logging.info("3/4: Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    logging.info("4/4: Creating and saving vector store via OpenAI API...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Batch processing
    batch_size = 500
    vector_store = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)
        logging.info(f"  - Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")

    vector_store_path = VECTOR_STORE_DIR / repo_name
    vector_store.save_local(str(vector_store_path))
    
    logging.info(f"\n✅ Indexing complete for '{repo_name}'.")

def chat_with_repo(repo_name: str, question: str):
    """Uses a multi-step chain to answer a question about a repository."""
    # --- 0. Load Vector Store and LLM ---
    vector_store_path = VECTOR_STORE_DIR / repo_name
    if not vector_store_path.exists():
        logging.error(f"Error: No index found for '{repo_name}'. Please index it first.")
        return

    logging.info(f"Loading index for '{repo_name}'...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local(str(vector_store_path), embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0)

    # --- 1. Rewrite the User's Question for Better Retrieval ---
    if not REWRITE_PROMPT_TEMPLATE_FILE.exists():
        logging.error(f"Error: Rewrite prompt template file '{REWRITE_PROMPT_TEMPLATE_FILE}' not found.")
        return
    with open(REWRITE_PROMPT_TEMPLATE_FILE, 'r') as f:
        rewrite_template_str = f.read()
    
    rewrite_prompt = PromptTemplate.from_template(rewrite_template_str)
    rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)
    
    logging.info(f"Original user question: {question}")
    rewritten_question = rewrite_chain.invoke({"question": question})["text"]
    logging.info(f"Rewritten question for vector search: {rewritten_question}")

    # --- 2. Retrieve Context from the Vector Database ---
    retriever = vector_store.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(rewritten_question)
    
    logging.info("\n--- Retrieved Context ---")
    for doc in retrieved_docs:
        logging.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        logging.info(f"Content:\n{doc.page_content}\n")
    logging.info("-------------------------\n")

    # --- 3. Synthesize the Final Answer ---
    if not PROMPT_TEMPLATE_FILE.exists():
        logging.error(f"Error: Prompt template file '{PROMPT_TEMPLATE_FILE}' not found.")
        return
    with open(PROMPT_TEMPLATE_FILE, 'r') as f:
        answer_template_str = f.read()

    answer_prompt = PromptTemplate.from_template(answer_template_str)
    answer_chain = LLMChain(llm=llm, prompt=answer_prompt)

    context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    final_answer = answer_chain.invoke({
        "context": context_str,
        "question": question # Use the original question for the final answer
    })["text"]

    logging.info("\n--- Final Answer ---")
    logging.info(final_answer)
    logging.info("--------------------\n")


# --- CLI COMMAND PARSER ---
def main():
    parser = argparse.ArgumentParser(description="A CLI to chat with GitHub repositories using OpenAI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a GitHub repository.")
    index_parser.add_argument("url", type=str, help="The full HTTPS URL of the GitHub repository.")
    
    chat_parser = subparsers.add_parser("chat", help="Chat with an indexed repository.")
    chat_parser.add_argument("repo_name", type=str, help="The name of the repository (e.g., 'fastapi').")
    chat_parser.add_argument("question", type=str, help="The question you want to ask the repository.")
    
    subparsers.add_parser("list", help="List all indexed repositories.")

    args = parser.parse_args()

    REPO_CACHE_DIR.mkdir(exist_ok=True)
    VECTOR_STORE_DIR.mkdir(exist_ok=True)

    if args.command == "index":
        index_repository(args.url)
    elif args.command == "chat":
        chat_with_repo(args.repo_name, args.question)
    elif args.command == "list":
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT name, github_url FROM repositories")
        repos = cursor.fetchall()
        conn.close()
        if not repos:
            logging.info("No repositories have been indexed yet.")
        else:
            logging.info("Indexed Repositories:")
            for name, url in repos:
                logging.info(f"  - {name} ({url})")

if __name__ == "__main__":
    init_db()
    main()