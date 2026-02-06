#!/usr/bin/env python3
"""
Indexing CLI for the Qdrant RAG system.

This script provides command-line access to document indexing functionality
using the shared library components.
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance

# Import from shared library (use PYTHONPATH environment variable)
from lib.embedding.client import OllamaEmbeddingClient
from lib.embedding.models import get_model_registry
from lib.qdrant.indexing import (
    QdrantIndexer,
    read_markdown_files,
    read_json_files,
    read_single_markdown_file,
)
from lib.qdrant.search import get_collection_stats
from lib.utils.config import get_config
from lib.utils.exclude import load_exclude_patterns

logger = logging.getLogger(__name__)


def progress_callback(current: int, total: int) -> None:
    """Progress callback for indexing operations."""
    percentage = (current / total) * 100
    print(f"📊 Progress: {current}/{total} chunks indexed ({percentage:.1f}%)")


def create_collection_if_needed(
    qdrant_client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance_metric: str = "cosine",
    recreate: bool = False,
    verbose: bool = False,
) -> bool:
    """Create collection if it doesn't exist."""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        existing_names = [col.name for col in collections.collections]

        if collection_name in existing_names:
            if recreate:
                log_print(f"🗑️  Deleting existing collection '{collection_name}'...", verbose)
                qdrant_client.delete_collection(collection_name)
                log_print(f"✅ Collection '{collection_name}' deleted", verbose)
            else:
                log_print(f"📦 Collection '{collection_name}' already exists", verbose)
                return True

        # Map string distance to Qdrant Distance enum
        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        distance = distance_map.get(distance_metric.lower(), Distance.COSINE)

        log_print(
            f"🏗️  Creating collection '{collection_name}' with {vector_size} dimensions...", verbose
        )

        indexer = QdrantIndexer(
            qdrant_client, None, ""
        )  # embedding client not needed for collection creation
        success = indexer.create_collection(collection_name, vector_size, distance)

        if success:
            log_print(f"✅ Collection '{collection_name}' created successfully", verbose)
            return True
        else:
            print(f"❌ Failed to create collection '{collection_name}'")
            return False

    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return False


def load_documents(
    input_path: str, max_docs: Optional[int] = None, file_type: str = "auto",
    base_dir: Optional[str] = None, exclude_patterns: Optional[List[str]] = None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """Load documents from input path (file or directory)."""
    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Handle single file input
    if input_path_obj.is_file():
        if input_path.endswith(".md"):
            log_print(f"📄 Loading single markdown file: {input_path}", verbose)
            doc = read_single_markdown_file(input_path, base_dir, exclude_patterns)
            return [doc]
        elif input_path.endswith(".json"):
            # For JSON, we still need to handle it - could be single doc or array
            log_print(f"📄 Loading single JSON file: {input_path}", verbose)
            return read_json_files(str(input_path_obj.parent), max_docs)
        else:
            raise ValueError(f"Unsupported file type: {input_path}")

    # Handle directory input
    input_dir = input_path_obj

    # Auto-detect file type if not specified
    if file_type == "auto":
        json_files = list(input_dir.rglob("*.json"))
        md_files = list(input_dir.rglob("*.md"))

        if json_files and not md_files:
            file_type = "json"
        elif md_files and not json_files:
            file_type = "markdown"
        elif json_files and md_files:
            log_print(
                f"📁 Found both JSON ({len(json_files)}) and Markdown ({len(md_files)}) files", verbose
            )
            file_type = "json"  # Prefer JSON if both exist
        else:
            raise ValueError(f"No supported files found in: {input_path}")

    log_print(f"📂 Loading {file_type} documents from: {input_path}", verbose)

    if file_type == "json":
        return read_json_files(str(input_dir), max_docs)
    elif file_type == "markdown":
        return read_markdown_files(str(input_dir), max_docs, exclude_patterns)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def log_print(message: str, verbose: bool = False) -> None:
    """Print message only in verbose mode."""
    if verbose:
        print(message)


def index_documents(args) -> None:
    """Main indexing function."""
    verbose = args.verbose
    log_print("🚀 Starting document indexing...", verbose)
    start_time = time.time()

    # Load configuration
    config = get_config(args.config_file) if args.config_file else get_config()
    
    # Load exclude patterns
    exclude_patterns = []
    if not args.no_exclude:
        exclude_patterns = load_exclude_patterns(args.exclude_file)
        if exclude_patterns:
            log_print(f"🚫 Loaded {len(exclude_patterns)} exclude patterns", verbose)

    # Initialize clients
    log_print("🔗 Connecting to services...", verbose)
    qdrant_client = QdrantClient(url=args.qdrant_url, timeout=60.0)
    embedding_client = OllamaEmbeddingClient(
        ollama_url=args.ollama_url,
        timeout=args.connection_timeout,
    )

    try:
        # Get vector size for the model
        log_print(f"🤖 Getting model information for: {args.model}", verbose)
        registry = get_model_registry(args.ollama_url)
        vector_size = registry.get_or_detect_vector_size(args.model)

        if not vector_size:
            print(f"❌ Could not determine vector size for model: {args.model}")
            print("Available models in registry:")
            for model in registry.list_available_models()[:5]:
                print(
                    f"  • {model['name']} ({model.get('vector_size', 'unknown')} dims)"
                )
            sys.exit(1)

        log_print(f"✅ Model '{args.model}' uses {vector_size} dimensional vectors", verbose)

        # Create collection
        if not create_collection_if_needed(
            qdrant_client,
            args.collection,
            vector_size,
            args.distance_metric,
            args.recreate,
            verbose,
        ):
            sys.exit(1)

        # Load documents
        log_print(f"📚 Loading documents from: {args.input_path}", verbose)
        documents = load_documents(
            args.input_path, args.max_docs, args.file_type, 
            exclude_patterns=exclude_patterns,
            verbose=verbose
        )

        if not documents:
            print("❌ No documents found to index")
            sys.exit(1)

        log_print(f"📖 Loaded {len(documents)} documents", verbose)

        # Show collection stats before indexing
        if verbose:
            stats = get_collection_stats(qdrant_client, args.collection)
            print("📊 Collection stats before indexing:")
            for key, value in stats.items():
                print(f"   {key}: {value}")

        # Initialize indexer
        indexer = QdrantIndexer(
            qdrant_client=qdrant_client,
            embedding_client=embedding_client,
            embedding_model=args.model,
        )

        # Index documents
        log_print(f"⚙️  Starting indexing with {args.workers} workers...", verbose)
        success = indexer.index_documents(
            collection_name=args.collection,
            documents=documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_chunks_per_article=args.max_chunks_per_article,
            batch_size=args.batch_size,
            max_workers=args.workers,
            progress_callback=progress_callback if verbose else None,
        )

        if not success:
            print("❌ Indexing failed")
            sys.exit(1)

        # Show final stats (verbose only)
        if verbose:
            print("\n📊 Final collection statistics:")
            stats = get_collection_stats(qdrant_client, args.collection)
            for key, value in stats.items():
                print(f"   {key}: {value}")

        # Calculate timing
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60

        # Essential output - always show success summary
        if not verbose:
            # Minimal output (default mode)
            stats = get_collection_stats(qdrant_client, args.collection)
            print(f"✅ Indexed {len(documents)} docs → {args.collection} ({stats.get('points_count', '?')} chunks, {minutes}m {seconds:.1f}s)")
        else:
            print("\n✅ Indexing completed successfully!")
            print(f"⏱️  Total time: {minutes}m {seconds:.1f}s")
            print(f"📈 Indexed {len(documents)} documents")
            print(f"🎯 Collection: {args.collection}")
            print(f"🤖 Model: {args.model}")

    except KeyboardInterrupt:
        print("\n⚠️  Indexing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        qdrant_client.close()


def show_collection_info(args) -> None:
    """Show information about collections."""
    try:
        qdrant_client = QdrantClient(url=args.qdrant_url, timeout=30.0)

        if args.collection:
            # Show specific collection stats
            print(f"📊 Statistics for collection '{args.collection}':")
            stats = get_collection_stats(qdrant_client, args.collection)
            for key, value in stats.items():
                print(f"   {key}: {value}")
        else:
            # List all collections
            collections = qdrant_client.get_collections()
            print(f"📦 Found {len(collections.collections)} collections:")

            for collection in collections.collections:
                print(f"\n  📁 {collection.name}")
                stats = get_collection_stats(qdrant_client, collection.name)
                for key, value in stats.items():
                    print(f"     {key}: {value}")

        qdrant_client.close()

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


def delete_by_source(args) -> None:
    """Delete all chunks for a specific source/file from the collection."""
    print(f"🗑️  Deleting chunks for: {args.delete}")
    print(f"🎯 Collection: {args.collection}")

    try:
        qdrant_client = QdrantClient(url=args.qdrant_url, timeout=30.0)

        # Get stats before deletion
        stats_before = get_collection_stats(qdrant_client, args.collection)
        points_before = stats_before.get("points_count", 0)

        # Create indexer (embedding client not needed for delete)
        indexer = QdrantIndexer(qdrant_client, None, "")

        # Delete chunks by source
        indexer.delete_by_source(args.collection, args.delete)

        # Get stats after deletion
        stats_after = get_collection_stats(qdrant_client, args.collection)
        points_after = stats_after.get("points_count", 0)

        deleted_count = points_before - points_after
        print(f"✅ Deleted {deleted_count} chunks")
        print(f"📊 Collection now has {points_after} chunks")

        qdrant_client.close()

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    # Load configuration
    config = get_config()

    parser = argparse.ArgumentParser(
        description="Index documents into Qdrant vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input-path ./documents --collection docs
  %(prog)s --input-path ./single-file.md --collection docs  # Index single file
  %(prog)s --input-path ./articles.json --collection articles --recreate
  %(prog)s --input-path ./data --model bge-m3:567m --chunk-size 200
  %(prog)s --delete MEMORY.md --collection docs  # Delete chunks for a file
  %(prog)s --info  # Show all collections
  %(prog)s --info --collection docs  # Show specific collection stats

Supported file types: JSON, Markdown (.md)
File type is auto-detected based on file extensions.
Single file indexing uses deterministic IDs for proper upsert (update in place).
    """,
    )

    # Main operation mode
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument(
        "--input-path", "-i", help="Path to documents directory or file to index"
    )
    operation.add_argument(
        "--info", action="store_true", help="Show collection information and statistics"
    )
    operation.add_argument(
        "--delete", "-d", metavar="FILE_PATH",
        help="Delete all chunks for a specific file/source from the collection"
    )

    # Connection settings
    parser.add_argument(
        "--qdrant-url",
        default=config.database.url,
        help=f"Qdrant server URL (default: {config.database.url})",
    )
    parser.add_argument(
        "--ollama-url",
        default=config.embedding.url,
        help=f"Ollama API URL (default: {config.embedding.url})",
    )
    parser.add_argument(
        "--collection", "-c", default="docs", help="Collection name (default: docs)"
    )

    # Model settings
    parser.add_argument(
        "--model",
        "-m",
        default=config.embedding.model,
        help=f"Embedding model to use (default: {config.embedding.model})",
    )
    parser.add_argument(
        "--distance-metric",
        choices=["cosine", "dot", "euclidean"],
        default="cosine",
        help="Distance metric for similarity (default: cosine)",
    )

    # Exclude patterns
    parser.add_argument(
        "--exclude-file",
        help="Path to exclude file (default: .exclude in qras directory)",
    )
    parser.add_argument(
        "--no-exclude",
        action="store_true",
        help="Disable exclude patterns (ignore .exclude file)",
    )

    # Indexing parameters
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection from scratch (deletes existing data)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        help="Maximum number of documents to index (useful for testing)",
    )
    parser.add_argument(
        "--file-type",
        choices=["auto", "json", "markdown"],
        default="auto",
        help="Input file type (default: auto-detect)",
    )

    # Chunking parameters
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=config.indexing.chunk_size,
        help=f"Number of words per chunk (default: {config.indexing.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=config.indexing.chunk_overlap,
        help=f"Number of overlapping words between chunks (default: {config.indexing.chunk_overlap})",
    )
    parser.add_argument(
        "--max-chunks-per-article",
        type=int,
        default=config.indexing.max_chunks_per_article,
        help=f"Maximum number of chunks per article (default: {config.indexing.max_chunks_per_article})",
    )

    # Performance parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.embedding.batch_size,
        help=f"Number of chunks to process in each batch (default: {config.embedding.batch_size})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.embedding.max_workers,
        help=f"Number of concurrent embedding workers (default: {config.embedding.max_workers})",
    )
    parser.add_argument(
        "--connection-timeout",
        type=int,
        default=config.embedding.timeout,
        help=f"Connection timeout in seconds (default: {config.embedding.timeout})",
    )

    # Output options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress and debug output (default: minimal output)",
    )
    parser.add_argument("--config-file", help="Path to configuration file")

    args = parser.parse_args()

    # Setup logging - use ERROR level by default, INFO only in verbose mode
    log_level = logging.INFO if args.verbose else logging.ERROR
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    
    # Suppress library logs unless verbose
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        logging.getLogger("qdrant_client").setLevel(logging.ERROR)

    # Validate arguments
    if not args.input_path and not args.info and not args.delete:
        parser.error(
            "Must provide --input-path to index, --delete to remove, or --info to show collection information"
        )

    if args.input_path and not Path(args.input_path).exists():
        parser.error(f"Input path does not exist: {args.input_path}")

    try:
        if args.info:
            show_collection_info(args)
        elif args.delete:
            delete_by_source(args)
        else:
            index_documents(args)

    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
