"""
Qdrant indexing operations for document ingestion and collection management.

This module provides functionality for creating collections, chunking documents,
generating embeddings, and indexing content into Qdrant vector database.
"""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict, Callable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ..embedding.client import OllamaEmbeddingClient
from ..embedding.formatter import format_document


def generate_chunk_id(source: str, chunk_index: int) -> int:
    """
    Generate a deterministic chunk ID based on source and chunk index.
    
    This enables proper upsert behavior - the same content will always
    produce the same ID, allowing incremental updates without duplicates.
    
    Args:
        source: Source identifier (file path, URL, or title)
        chunk_index: Index of the chunk within the document
        
    Returns:
        A positive integer ID derived from MD5 hash
    """
    # Create a unique string combining source and chunk index
    unique_str = f"{source}:chunk:{chunk_index}"
    # Generate MD5 hash and take first 16 hex chars (64 bits)
    hash_hex = hashlib.md5(unique_str.encode('utf-8')).hexdigest()[:16]
    # Convert to positive integer
    return int(hash_hex, 16)

logger = logging.getLogger(__name__)


class QdrantIndexer:
    """Main class for indexing documents into Qdrant collections."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_client: OllamaEmbeddingClient,
        embedding_model: str = "embeddinggemma:latest",
    ):
        """
        Initialize the Qdrant indexer.

        Args:
            qdrant_client: Initialized QdrantClient instance
            embedding_client: Initialized OllamaEmbeddingClient instance
            embedding_model: Name of the embedding model to use
        """
        self.qdrant_client = qdrant_client
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 768,
        distance: Distance = Distance.COSINE,
        on_disk_payload: bool = False,
    ) -> bool:
        """
        Create a new Qdrant collection.

        Args:
            collection_name: Name of the collection to create
            vector_size: Dimension of the embedding vectors
            distance: Distance metric to use
            on_disk_payload: Whether to store payload on disk

        Returns:
            True if collection was created successfully, False otherwise
        """
        try:
            # Check if collection already exists
            collections = self.qdrant_client.get_collections()
            existing_names = [col.name for col in collections.collections]

            if collection_name in existing_names:
                logger.info(f"Collection '{collection_name}' already exists")
                return True

            # Create collection with specified parameters
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance,
                    on_disk=True,  # Store vectors on disk for better memory usage
                ),
                on_disk_payload=on_disk_payload,
            )

            logger.info(
                f"✅ Created collection '{collection_name}' (vector_size={vector_size})"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create collection '{collection_name}': {e}")
            return False

    def chunk_text(
        self, text: str, chunk_size: int = 150, overlap: int = 30
    ) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: The text to chunk
            chunk_size: Number of words per chunk (default 150 words)
            overlap: Number of words to overlap between chunks (default 30 words)

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Split text into words
        words = text.split()

        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            # Move forward by (chunk_size - overlap) words
            if end >= len(words):
                break
            start += chunk_size - overlap

        return chunks

    def create_chunk_objects(
        self,
        article: Dict,
        chunk_size: int = 150,
        chunk_overlap: int = 30,
        max_chunks_per_article: int = 10,
    ) -> List[Dict]:
        """
        Create chunk objects from an article.

        Args:
            article: Article dictionary with id, title and content
            chunk_size: Number of words per chunk
            chunk_overlap: Number of overlapping words between chunks
            max_chunks_per_article: Maximum number of chunks per article

        Returns:
            List of chunk objects ready for indexing
        """
        content = article.get("content", "")
        title = article.get("title", "")
        article_id = article.get("id")

        chunks = self.chunk_text(content, chunk_size, chunk_overlap)

        # Limit chunks per article to avoid overwhelming the index
        chunks = chunks[:max_chunks_per_article]

        # Use file_path if available, otherwise fall back to title or article_id
        source_identifier = (
            article.get("file_path") 
            or article.get("source") 
            or title 
            or str(article_id)
        )

        chunk_objects = []
        for i, chunk in enumerate(chunks):
            # Skip very short chunks
            words = chunk.split(" ")
            if len(words) < 10:
                continue

            # Use the embedding formatter to format text according to model requirements
            formatted_text = format_document(title, chunk, self.embedding_model)
            print("Formatted text:", formatted_text)

            # Generate deterministic chunk ID based on source and index
            # This enables proper incremental updates via upsert
            chunk_id = generate_chunk_id(source_identifier, i)

            chunk_objects.append(
                {
                    "chunk_id": chunk_id,
                    "article_id": article_id,
                    "chunk_index": i,
                    "title": title,
                    "content": chunk,
                    "text": formatted_text,  # Model-specific formatted text for embedding
                    "source": article.get("source", source_identifier),
                }
            )

        return chunk_objects

    def embed_batch_concurrent(
        self,
        texts: List[str],
        max_workers: int = 4,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts concurrently."""
        results = [None] * len(texts)  # Pre-allocate results list

        def embed_single(index_text_pair):
            index, text = index_text_pair
            try:
                embedding = self.embedding_client.embed_text(text, self.embedding_model)
                return index, embedding
            except Exception as e:
                raise RuntimeError(f"Failed to embed text at index {index}: {e}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(embed_single, (i, text)): i
                for i, text in enumerate(texts)
            }

            # Collect results
            for future in as_completed(future_to_index):
                try:
                    index, embedding = future.result()
                    results[index] = embedding
                except Exception as e:
                    raise e

        return results

    def index_chunks(
        self,
        collection_name: str,
        chunks: List[Dict],
        batch_size: int = 50,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """
        Index chunks into Qdrant collection.

        Args:
            collection_name: Name of the collection to index into
            chunks: List of chunk objects to index
            batch_size: Number of chunks to process in each batch
            max_workers: Number of concurrent embedding workers
            progress_callback: Optional callback for progress reporting

        Returns:
            True if indexing succeeded, False otherwise
        """
        if not chunks:
            logger.warning("No chunks to index")
            return True

        try:
            total_chunks = len(chunks)
            logger.info(
                f"📝 Indexing {total_chunks} chunks into '{collection_name}'..."
            )

            # Process chunks in batches
            for batch_start in range(0, total_chunks, batch_size):
                batch_end = min(batch_start + batch_size, total_chunks)
                batch_chunks = chunks[batch_start:batch_end]

                # Extract texts for embedding
                texts = [chunk["text"] for chunk in batch_chunks]

                # Generate embeddings concurrently
                embeddings = self.embed_batch_concurrent(texts, max_workers)

                # Create points for Qdrant
                points = []
                for chunk, embedding in zip(batch_chunks, embeddings):
                    if embedding is None:
                        logger.warning(
                            f"Skipping chunk {chunk['chunk_id']} due to embedding failure"
                        )
                        continue

                    point = PointStruct(
                        id=chunk["chunk_id"],
                        vector=embedding,
                        payload={
                            "article_id": chunk["article_id"],
                            "chunk_index": chunk["chunk_index"],
                            "title": chunk["title"],
                            "content": chunk["content"],
                            "source": chunk["source"],
                        },
                    )
                    points.append(point)

                # Upload batch to Qdrant
                if points:
                    self.qdrant_client.upsert(
                        collection_name=collection_name,
                        points=points,
                    )

                # Report progress
                if progress_callback:
                    progress_callback(batch_end, total_chunks)
                else:
                    logger.info(f"📊 Indexed {batch_end}/{total_chunks} chunks...")

            logger.info(f"✅ Successfully indexed {total_chunks} chunks")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to index chunks: {e}")
            return False

    def index_documents(
        self,
        collection_name: str,
        documents: List[Dict],
        chunk_size: int = 150,
        chunk_overlap: int = 30,
        max_chunks_per_article: int = 10,
        batch_size: int = 50,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """
        Index documents into Qdrant collection (full pipeline).

        Args:
            collection_name: Name of the collection to index into
            documents: List of document dictionaries
            chunk_size: Number of words per chunk
            chunk_overlap: Number of overlapping words between chunks
            max_chunks_per_article: Maximum number of chunks per article
            batch_size: Number of chunks to process in each batch
            max_workers: Number of concurrent embedding workers
            progress_callback: Optional callback for progress reporting

        Returns:
            True if indexing succeeded, False otherwise
        """
        logger.info(
            f"🚀 Starting document indexing pipeline for {len(documents)} documents..."
        )

        # Step 1: Create chunks from documents
        logger.info("📄 Creating chunks from documents...")
        all_chunks = []
        for doc in documents:
            doc_chunks = self.create_chunk_objects(
                doc, chunk_size, chunk_overlap, max_chunks_per_article
            )
            all_chunks.extend(doc_chunks)

        logger.info(
            f"📝 Created {len(all_chunks)} chunks from {len(documents)} documents"
        )

        # Step 2: Index chunks
        return self.index_chunks(
            collection_name, all_chunks, batch_size, max_workers, progress_callback
        )


# Document reading utilities
def read_markdown_files(
    directory_path: str, max_docs: Optional[int] = None
) -> List[Dict]:
    """
    Read and parse markdown files from a directory.

    Args:
        directory_path: Path to directory containing markdown files
        max_docs: Maximum number of documents to process

    Returns:
        List of document dictionaries with id, title, and content
    """
    documents = []
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    # Find all markdown files recursively
    md_files = sorted(directory.rglob("*.md"))

    if not md_files:
        raise ValueError(f"No markdown files found in: {directory_path}")

    logger.info(f"Found {len(md_files)} markdown files in {directory_path}")

    if max_docs:
        md_files = md_files[:max_docs]
        logger.info(f"Limited to {len(md_files)} files (max_docs={max_docs})")

    for idx, md_file in enumerate(md_files):
        try:
            # Read file content
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                logger.warning(f"Skipping empty file: {md_file}")
                continue

            # Extract title from first H1 heading or use filename
            title = extract_title_from_markdown(content)
            if not title:
                # Use filename without extension as title
                title = md_file.stem.replace("_", " ").replace("-", " ")

            # Create document with unique ID based on file index
            doc = {
                "id": idx + 1,  # Start from 1
                "title": title,
                "content": content,
                "file_path": str(md_file.relative_to(directory)),
                "category": md_file.parent.name
                if md_file.parent != directory
                else "root",
            }

            documents.append(doc)

            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(md_files)} files")

        except Exception as e:
            logger.error(f"Error reading file {md_file}: {e}")
            continue

    logger.info(f"Successfully loaded {len(documents)} markdown documents")
    return documents


def read_json_files(directory_path: str, max_docs: Optional[int] = None) -> List[Dict]:
    """
    Read and parse JSON files from a directory.

    Args:
        directory_path: Path to directory containing JSON files
        max_docs: Maximum number of documents to process

    Returns:
        List of document dictionaries with id, title, and content
    """
    documents = []
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    # Find all JSON files recursively
    json_files = sorted(directory.rglob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in: {directory_path}")

    logger.info(f"Found {len(json_files)} JSON files in {directory_path}")

    if max_docs:
        logger.info(f"Will process up to {max_docs} documents total from JSON files")

    for json_file in json_files:
        if max_docs and len(documents) >= max_docs:
            logger.info(f"Reached maximum document limit ({max_docs}), stopping")
            break

        try:
            # Read JSON file content
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both single objects and arrays
            if isinstance(data, dict):
                # Single JSON object
                json_documents = [data]
            elif isinstance(data, list):
                # Array of JSON objects
                json_documents = data
            else:
                logger.warning(
                    f"Skipping {json_file}: Invalid JSON structure (not object or array)"
                )
                continue

            for item in json_documents:
                if max_docs and len(documents) >= max_docs:
                    break

                # Validate required fields
                if not isinstance(item, dict):
                    continue

                article_id = item.get("article_id") or item.get("id")
                title = item.get("title", "")
                content = item.get("content", "")

                if not article_id:
                    continue

                doc = {
                    "id": article_id,
                    "title": title,
                    "content": content,
                    "source": item.get("source", "json"),
                }

                documents.append(doc)

        except Exception as e:
            logger.error(f"Error reading file {json_file}: {e}")
            continue

    logger.info(f"Successfully loaded {len(documents)} JSON documents")
    return documents


def extract_title_from_markdown(content: str) -> Optional[str]:
    """
    Extract title from markdown content.
    Looks for the first H1 heading (# Title) or underlined title.

    Args:
        content: Markdown content

    Returns:
        Title string or None if not found
    """
    lines = content.split("\n")

    # Check for H1 heading (# Title)
    for line in lines:
        line = line.strip()
        if line.startswith("# ") and len(line) > 2:
            # Found H1 heading
            return line[2:].strip()

    # Check for underlined title (Title\n=====)
    for i, line in enumerate(lines[:-1]):
        next_line = lines[i + 1].strip()
        if next_line and all(c == "=" for c in next_line) and len(next_line) >= 3:
            # Previous line is likely a title
            if line.strip():
                return line.strip()

    return None


# Legacy functions for backward compatibility
def embed_batch_concurrent(
    texts: List[str],
    model: str,
    ollama_url: str,
    max_workers: int = 4,
    timeout: int = 120,
) -> List[List[float]]:
    """Legacy function for backward compatibility."""
    client = OllamaEmbeddingClient(ollama_url, timeout)
    indexer = QdrantIndexer(
        None, client, model
    )  # qdrant_client can be None for this function
    return indexer.embed_batch_concurrent(texts, max_workers)


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    """Legacy function for backward compatibility."""
    indexer = QdrantIndexer(None, None, "")  # Parameters not needed for this function
    return indexer.chunk_text(text, chunk_size, overlap)


def create_chunk_objects(
    article: Dict,
    chunk_size: int = 150,
    chunk_overlap: int = 30,
    max_chunks_per_article: int = 10,
    model_name: str = "embeddinggemma:latest",
) -> List[Dict]:
    """
    Legacy function for backward compatibility.
    
    Note: Now uses deterministic chunk IDs based on source/file_path,
    enabling proper incremental indexing via upsert.
    """
    indexer = QdrantIndexer(None, None, model_name)
    return indexer.create_chunk_objects(
        article, chunk_size, chunk_overlap, max_chunks_per_article
    )
