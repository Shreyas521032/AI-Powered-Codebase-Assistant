import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import os
import git
import sqlite3
import hashlib
import json
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
import requests
import time
import shutil
from typing import List, Dict, Optional
import tree_sitter
from tree_sitter import Language, Parser
import subprocess
import tempfile

# Page config
st.set_page_config(
    page_title="🤖 AI Codebase Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .code-block {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .sidebar-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.pinecone_connected = False
    st.session_state.model_loaded = False
    st.session_state.repositories = []
    st.session_state.chat_history = []

class CodebaseAssistant:
    def __init__(self):
        self.embedding_model = None
        self.pinecone_client = None
        self.pinecone_index = None
        self.db_path = "data/metadata.db"
        self.repos_path = "data/repos"
        self.setup_directories()
        self.setup_database()
    
    def setup_directories(self):
        """Create necessary directories"""
        Path("data").mkdir(exist_ok=True)
        Path(self.repos_path).mkdir(exist_ok=True)
    
    def setup_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_chunks (
                id INTEGER PRIMARY KEY,
                repo_name TEXT,
                file_path TEXT,
                chunk_content TEXT,
                chunk_type TEXT,
                language TEXT,
                vector_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS repositories (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                url TEXT,
                status TEXT,
                chunks_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def initialize_pinecone(self, api_key: str, environment: str = "gcp-starter"):
        """Initialize Pinecone connection"""
        try:
            # Initialize Pinecone client
            self.pinecone_client = Pinecone(api_key=api_key)
            
            # Create index if it doesn't exist
            index_name = "codebase-assistant"
            existing_indexes = self.pinecone_client.list_indexes().names()
            
            if index_name not in existing_indexes:
                # Create serverless index
                self.pinecone_client.create_index(
                    name=index_name,
                    dimension=512,  # all-MiniLM-L6-v2 dimension
                    metric="cosine"
                )
                # Wait for index to be ready
                while not self.pinecone_client.describe_index(index_name).status['ready']:
                    time.sleep(1)
            
            self.pinecone_index = self.pinecone_client.Index(index_name)
            return True, "Pinecone connected successfully!"
        except Exception as e:
            return False, f"Pinecone connection failed: {str(e)}"
    
    def load_embedding_model(self):
        """Load sentence transformer model"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            return True, "Embedding model loaded successfully!"
        except Exception as e:
            return False, f"Failed to load embedding model: {str(e)}"
    
    def clone_repository(self, repo_url: str, repo_name: str = None):
        """Clone a repository"""
        try:
            if not repo_name:
                repo_name = repo_url.split('/')[-1].replace('.git', '')
            
            repo_path = Path(self.repos_path) / repo_name
            
            # Remove existing repo if it exists
            if repo_path.exists():
                shutil.rmtree(repo_path)
            
            # Clone repository
            git.Repo.clone_from(repo_url, repo_path)
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO repositories (name, url, status)
                VALUES (?, ?, ?)
            ''', (repo_name, repo_url, 'cloned'))
            conn.commit()
            conn.close()
            
            return True, f"Repository '{repo_name}' cloned successfully!"
        except Exception as e:
            return False, f"Failed to clone repository: {str(e)}"
    
    def extract_code_chunks(self, repo_name: str):
        """Extract code chunks from repository"""
        try:
            repo_path = Path(self.repos_path) / repo_name
            code_chunks = []
            
            # Supported file extensions
            code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt'}
            
            for file_path in repo_path.rglob('*'):
                if file_path.is_file() and file_path.suffix in code_extensions:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Simple chunking by functions/classes
                        chunks = self.simple_chunk_code(content, str(file_path.relative_to(repo_path)))
                        code_chunks.extend(chunks)
                    except Exception as e:
                        continue
            
            return True, code_chunks, f"Extracted {len(code_chunks)} code chunks"
        except Exception as e:
            return False, [], f"Failed to extract code chunks: {str(e)}"
    
    def simple_chunk_code(self, content: str, file_path: str):
        """Simple code chunking by functions and classes"""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        chunk_type = 'code'
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Start of function or class
            if (stripped.startswith('def ') or stripped.startswith('class ') or 
                stripped.startswith('function ') or stripped.startswith('export function')):
                
                # Save previous chunk if it exists
                if current_chunk and len('\n'.join(current_chunk).strip()) > 50:
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'file_path': file_path,
                        'type': chunk_type,
                        'language': self.get_language_from_extension(file_path)
                    })
                
                current_chunk = [line]
                chunk_type = 'function' if 'def ' in stripped or 'function' in stripped else 'class'
            else:
                current_chunk.append(line)
            
            # Save chunk if it gets too long
            if len(current_chunk) > 100:
                if len('\n'.join(current_chunk).strip()) > 50:
                    chunks.append({
                        'content': '\n'.join(current_chunk),
                        'file_path': file_path,
                        'type': chunk_type,
                        'language': self.get_language_from_extension(file_path)
                    })
                current_chunk = []
                chunk_type = 'code'
        
        # Save final chunk
        if current_chunk and len('\n'.join(current_chunk).strip()) > 50:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'file_path': file_path,
                'type': chunk_type,
                'language': self.get_language_from_extension(file_path)
            })
        
        return chunks
    
    def get_language_from_extension(self, file_path: str):
        """Get programming language from file extension"""
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        ext = Path(file_path).suffix
        return ext_to_lang.get(ext, 'unknown')
    
    def embed_and_store_chunks(self, repo_name: str, chunks: List[Dict]):
        """Generate embeddings and store in Pinecone"""
        try:
            if not self.embedding_model or not self.pinecone_index:
                return False, "Models not initialized"
            
            # Generate embeddings
            contents = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_model.encode(contents).tolist()
            
            # Prepare vectors for Pinecone
            vectors = []
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{repo_name}_{i}_{hashlib.md5(chunk['content'].encode()).hexdigest()[:8]}"
                
                # Metadata for Pinecone
                metadata = {
                    'repo_name': repo_name,
                    'file_path': chunk['file_path'],
                    'chunk_type': chunk['type'],
                    'language': chunk['language'],
                    'content': chunk['content'][:1000]  # Limit content in metadata
                }
                
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
                
                # Store in SQLite
                cursor.execute('''
                    INSERT INTO code_chunks (repo_name, file_path, chunk_content, chunk_type, language, vector_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (repo_name, chunk['file_path'], chunk['content'], chunk['type'], chunk['language'], vector_id))
            
            # Batch upsert to Pinecone
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.pinecone_index.upsert(vectors=batch)
            
            # Update repository status
            cursor.execute('''
                UPDATE repositories SET status = ?, chunks_count = ?
                WHERE name = ?
            ''', ('indexed', len(chunks), repo_name))
            
            conn.commit()
            conn.close()
            
            return True, f"Successfully indexed {len(chunks)} chunks to Pinecone"
        except Exception as e:
            return False, f"Failed to embed and store chunks: {str(e)}"
    
    def search_code(self, query: str, top_k: int = 5, repo_filter: str = None):
        """Search for relevant code chunks"""
        try:
            if not self.embedding_model or not self.pinecone_index:
                return False, [], "Models not initialized"
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in Pinecone
            filter_dict = {}
            if repo_filter:
                filter_dict['repo_name'] = repo_filter
            
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Format results
            search_results = []
            for match in results['matches']:
                search_results.append({
                    'score': match['score'],
                    'repo_name': match['metadata']['repo_name'],
                    'file_path': match['metadata']['file_path'],
                    'chunk_type': match['metadata']['chunk_type'],
                    'language': match['metadata']['language'],
                    'content': match['metadata']['content']
                })
            
            return True, search_results, f"Found {len(search_results)} relevant chunks"
        except Exception as e:
            return False, [], f"Search failed: {str(e)}"
    
    def generate_response_with_ollama(self, query: str, context_chunks: List[Dict]):
        """Generate response using Ollama (fallback to simple response if not available)"""
        try:
            # Prepare context
            context = "\n\n".join([
                f"File: {chunk['file_path']}\nLanguage: {chunk['language']}\n```{chunk['language']}\n{chunk['content']}\n```"
                for chunk in context_chunks[:3]  # Limit context
            ])
            
            # Try Ollama first
            try:
                prompt = f"""Based on the following code context, answer the user's question:

Context:
{context}

Question: {query}

Please provide a helpful answer based on the code context above."""

                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={
                        'model': 'mistral:7b',
                        'prompt': prompt,
                        'stream': False
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()['response']
            except:
                pass
            
            # Fallback response
            if context_chunks:
                return f"""Based on the code context found, here are the most relevant pieces:

{chr(10).join([f"**{chunk['file_path']}** ({chunk['language']}):{chr(10)}```{chunk['language']}{chr(10)}{chunk['content'][:500]}...{chr(10)}```{chr(10)}" for chunk in context_chunks[:2]])}

This code appears to be related to your query: "{query}". You can examine the full context above for more details."""
            else:
                return f"I couldn't find specific code related to '{query}' in the indexed repositories. Try refining your search query or make sure the relevant repositories are indexed."
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_repositories(self):
        """Get list of repositories"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT name, url, status, chunks_count, created_at FROM repositories ORDER BY created_at DESC')
        repos = cursor.fetchall()
        conn.close()
        return repos

# Initialize the assistant
@st.cache_resource
def get_assistant():
    return CodebaseAssistant()

assistant = get_assistant()

# Main UI
def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Codebase Assistant</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by Mistral 7B, Pinecone & Streamlit**")
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚙️ Configuration")
        
        # Pinecone Setup
        st.subheader("🌲 Pinecone Setup")
        pinecone_api_key = st.text_input("Pinecone API Key", type="password", 
                                       help="Get your free API key from pinecone.io")
        
        # Note: Environment is now handled automatically by serverless
        st.info("💡 Using Pinecone Serverless (GCP us-central1)")
        
        if st.button("Connect to Pinecone", type="primary"):
            if pinecone_api_key:
                with st.spinner("Connecting to Pinecone..."):
                    success, message = assistant.initialize_pinecone(pinecone_api_key)
                    if success:
                        st.session_state.pinecone_connected = True
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.error("Please enter your Pinecone API key")
        
        # Model Setup
        st.subheader("🧠 Model Setup")
        if st.button("Load Embedding Model", type="secondary"):
            with st.spinner("Loading embedding model..."):
                success, message = assistant.load_embedding_model()
                if success:
                    st.session_state.model_loaded = True
                    st.success(message)
                else:
                    st.error(message)
        
        # Status indicators
        st.markdown("### 📊 Status")
        if st.session_state.pinecone_connected:
            st.markdown('<div class="status-success">✅ Pinecone Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ Pinecone Not Connected</div>', unsafe_allow_html=True)
        
        if st.session_state.model_loaded:
            st.markdown('<div class="status-success">✅ Model Loaded</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ Model Not Loaded</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Ollama Info
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("🦙 Ollama Setup (Optional)")
        st.markdown("""
        For best results, install Ollama locally:
        ```bash
        # Install Ollama
        curl -fsSL https://ollama.ai/install.sh | sh
        
        # Pull Mistral 7B
        ollama pull mistral:7b
        
        # Start Ollama server
        ollama serve
        ```
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📂 Repository Manager", "🔍 Code Search", "💬 Chat Assistant"])
    
    with tab1:
        st.header("📊 Dashboard")
        
        # Get repository stats
        repos = assistant.get_repositories()
        total_repos = len(repos)
        total_chunks = sum([repo[3] for repo in repos if repo[3]])
        indexed_repos = len([repo for repo in repos if repo[2] == 'indexed'])
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <h3>{total_repos}</h3>
                <p>Total Repositories</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <h3>{indexed_repos}</h3>
                <p>Indexed Repositories</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <h3>{total_chunks:,}</h3>
                <p>Code Chunks</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            status = "🟢 Ready" if (st.session_state.pinecone_connected and st.session_state.model_loaded) else "🔴 Setup Required"
            st.markdown(f'''
            <div class="metric-card">
                <h3>{status}</h3>
                <p>System Status</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Recent repositories
        if repos:
            st.subheader("📈 Recent Repositories")
            repo_data = []
            for repo in repos[:10]:
                repo_data.append({
                    "Repository": repo[0],
                    "Status": "✅ Indexed" if repo[2] == 'indexed' else "⏳ Cloned" if repo[2] == 'cloned' else "❌ Error",
                    "Chunks": repo[3] or 0,
                    "Added": repo[4][:10] if repo[4] else "Unknown"
                })
            st.dataframe(repo_data, use_container_width=True)
    
    with tab2:
        st.header("📂 Repository Manager")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("➕ Add New Repository")
            repo_url = st.text_input("Repository URL", 
                                   placeholder="https://github.com/username/repository.git",
                                   help="Enter the Git URL of the repository you want to index")
            repo_name = st.text_input("Repository Name (Optional)", 
                                    help="Leave empty to use the repository name from URL")
            
            if st.button("🚀 Clone & Index Repository", type="primary", disabled=not (st.session_state.pinecone_connected and st.session_state.model_loaded)):
                if repo_url:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Clone repository
                    status_text.text("Step 1/3: Cloning repository...")
                    progress_bar.progress(10)
                    success, message = assistant.clone_repository(repo_url, repo_name)
                    
                    if success:
                        actual_repo_name = repo_name if repo_name else repo_url.split('/')[-1].replace('.git', '')
                        progress_bar.progress(33)
                        
                        # Step 2: Extract code chunks
                        status_text.text("Step 2/3: Extracting code chunks...")
                        success, chunks, extract_message = assistant.extract_code_chunks(actual_repo_name)
                        
                        if success:
                            progress_bar.progress(66)
                            
                            # Step 3: Embed and store
                            status_text.text("Step 3/3: Generating embeddings and storing...")
                            success, store_message = assistant.embed_and_store_chunks(actual_repo_name, chunks)
                            
                            if success:
                                progress_bar.progress(100)
                                status_text.text("✅ Repository successfully indexed!")
                                st.success(f"🎉 {store_message}")
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error(f"❌ {store_message}")
                        else:
                            st.error(f"❌ {extract_message}")
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.error("Please enter a repository URL")
        
        with col2:
            st.subheader("📋 Setup Checklist")
            checklist_items = [
                ("Pinecone Connected", st.session_state.pinecone_connected),
                ("Embedding Model Loaded", st.session_state.model_loaded),
                ("Repository URL Entered", bool(st.session_state.get('repo_url', ''))),
            ]
            
            for item, status in checklist_items:
                icon = "✅" if status else "⭕"
                st.markdown(f"{icon} {item}")
        
        # Repository list
        st.subheader("📚 Existing Repositories")
        repos = assistant.get_repositories()
        
        if repos:
            for repo in repos:
                with st.expander(f"📁 {repo[0]} - {repo[2].title()}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**URL:** {repo[1]}")
                    with col2:
                        st.write(f"**Chunks:** {repo[3] or 0}")
                    with col3:
                        st.write(f"**Added:** {repo[4][:10] if repo[4] else 'Unknown'}")
        else:
            st.info("No repositories added yet. Add your first repository above!")
    
    with tab3:
        st.header("🔍 Code Search")
        
        if not (st.session_state.pinecone_connected and st.session_state.model_loaded):
            st.warning("⚠️ Please complete the setup in the sidebar first!")
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("🔍 Search your codebase", 
                                       placeholder="e.g., 'authentication function', 'database connection', 'error handling'",
                                       help="Enter natural language queries to find relevant code")
        
        with col2:
            repos = assistant.get_repositories()
            repo_names = ["All Repositories"] + [repo[0] for repo in repos if repo[2] == 'indexed']
            selected_repo = st.selectbox("Repository Filter", repo_names)
        
        if st.button("🚀 Search", type="primary") and search_query:
            with st.spinner("Searching through your codebase..."):
                repo_filter = None if selected_repo == "All Repositories" else selected_repo
                success, results, message = assistant.search_code(search_query, top_k=10, repo_filter=repo_filter)
                
                if success and results:
                    st.success(f"🎯 {message}")
                    
                    for i, result in enumerate(results):
                        with st.expander(f"📄 {result['file_path']} (Score: {result['score']:.3f})", expanded=i==0):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Repository:** {result['repo_name']}")
                            with col2:
                                st.write(f"**Language:** {result['language']}")
                            with col3:
                                st.write(f"**Type:** {result['chunk_type']}")
                            
                            st.markdown("**Code:**")
                            st.code(result['content'], language=result['language'])
                else:
                    st.error(f"❌ {message}")
    
    with tab4:
        st.header("💬 Chat Assistant")
        
        if not (st.session_state.pinecone_connected and st.session_state.model_loaded):
            st.warning("⚠️ Please complete the setup in the sidebar first!")
            return
        
        # Chat interface
        st.subheader("🤖 Ask questions about your codebase")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
        
        # Chat input
        user_question = st.text_input("💭 Ask a question about your code", 
                                    placeholder="e.g., 'How does the authentication work?', 'Show me the database models'")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Send", type="primary") and user_question:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                with st.spinner("🧠 Thinking..."):
                    # Search for relevant context
                    success, results, _ = assistant.search_code(user_question, top_k=3)
                    
                    if success:
                        # Generate response
                        response = assistant.generate_response_with_ollama(user_question, results)
                        
                        # Add assistant response to history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        st.rerun()
                    else:
                        st.error("Failed to search for relevant context")
        
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()
