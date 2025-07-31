import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class LocalVectorRAG:
    """Local Vector Retrieval-Augmented Generation tool for document search"""

    def __init__(self, docs_folder: str = "data/company_docs", model_name: str = "paraphrase-MiniLM-L6-v2", chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the Local Vector RAG system

        Args:
            docs_folder: Path to the documents folder
            model_name: Name of the sentence transformer model to use
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
        """
        self.docs_folder = Path(docs_folder)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs = []
        self.embeddings = []
        self.model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded sentence transformer model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.model = None
        else:
            logger.warning("sentence-transformers not available, falling back to simple text search")
        
        self._load_documents()

    def _load_documents(self) -> None:
        """Load documents from the specified folder"""
        if not self.docs_folder.exists():
            logger.warning(f"Documents folder {self.docs_folder} does not exist")
            return

        supported_extensions = {'.txt', '.md', '.json'}
        
        for file_path in self.docs_folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    content = self._read_file_content(file_path)
                    if content.strip():
                        chunks = self._split_text_into_chunks(content)
                        for i, chunk in enumerate(chunks):
                            doc_info = {
                                "content": chunk,
                                "source": str(file_path.relative_to(self.docs_folder)),
                                "full_path": str(file_path),
                                "chunk_id": i,
                                "total_chunks": len(chunks)
                            }
                            self.docs.append(doc_info)
                        logger.debug(f"Loaded document: {file_path} ({len(chunks)} chunks)")
                except Exception as e:
                    logger.error(f"Failed to load document {file_path}: {e}")

        logger.info(f"Loaded {len(self.docs)} document chunks from {self.docs_folder}")
        
        if self.model and self.docs:
            self._generate_embeddings()

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence or paragraph boundaries
            chunk = text[start:end]
            
            # Look for good break points
            break_points = ['\n\n', '\n', '. ', '。', '！', '？']
            best_break = -1
            
            for break_point in break_points:
                last_break = chunk.rfind(break_point)
                if last_break > self.chunk_size * 0.7:  # Don't break too early
                    best_break = last_break + len(break_point)
                    break
            
            if best_break > 0:
                chunks.append(text[start:start + best_break])
                start = start + best_break - self.chunk_overlap
            else:
                chunks.append(chunk)
                start = end - self.chunk_overlap
        
        return chunks

    def _read_file_content(self, file_path: Path) -> str:
        """Read content from a file with proper encoding handling"""
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'big5']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try binary mode
        with open(file_path, 'rb') as f:
            content = f.read()
            return content.decode('utf-8', errors='ignore')

    def _generate_embeddings(self) -> None:
        """Generate embeddings for all documents"""
        if not self.model:
            return
            
        try:
            texts = [doc["content"] for doc in self.docs]
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
            logger.info(f"Generated embeddings for {len(self.embeddings)} document chunks")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            self.embeddings = []

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant documents based on query

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of relevant document chunks with metadata
        """
        if not self.docs:
            logger.warning("No documents available for search")
            return []

        if self.model and len(self.embeddings) > 0:
            return self._vector_search(query, top_k)
        else:
            return self._simple_text_search(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform vector-based semantic search"""
        try:
            query_embedding = self.model.encode([query])[0]
            scores = np.dot(self.embeddings, query_embedding)
            
            # Get top k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                result = self.docs[idx].copy()
                result["score"] = float(scores[idx])
                result["search_type"] = "vector"
                results.append(result)
            
            logger.debug(f"Vector search found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self._simple_text_search(query, top_k)

    def _simple_text_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback simple text-based search"""
        query_lower = query.lower()
        scored_docs = []
        
        for doc in self.docs:
            content_lower = doc["content"].lower()
            score = content_lower.count(query_lower)
            
            # Add bonus for matches in filename
            if query_lower in doc["source"].lower():
                score += 10
            
            if score > 0:
                result = doc.copy()
                result["score"] = score
                result["search_type"] = "text"
                scored_docs.append(result)
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        logger.debug(f"Text search found {len(scored_docs)} results for query: {query}")
        return scored_docs[:top_k]

    def reload_documents(self) -> None:
        """Reload all documents from the folder"""
        self.docs = []
        self.embeddings = []
        self._load_documents()

    def get_document_count(self) -> int:
        """Get the total number of loaded document chunks"""
        return len(self.docs)

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return ['.txt', '.md', '.json']
