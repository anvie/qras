"""
Shared configuration management for the Qdrant RAG system.

This module provides centralized configuration handling for both CLI scripts
and web interface, with support for environment variables and default values.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field

# Try to import dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    url: str = "http://localhost:6333"
    timeout: int = 30
    collection_prefix: str = ""

    @classmethod
    def from_env(cls, prefix: str = "QDRANT_") -> "DatabaseConfig":
        """Create database config from environment variables."""
        return cls(
            url=os.getenv(f"{prefix}URL", cls.url),
            timeout=int(os.getenv(f"{prefix}TIMEOUT", cls.timeout)),
            collection_prefix=os.getenv(
                f"{prefix}COLLECTION_PREFIX", cls.collection_prefix
            ),
        )


@dataclass
class EmbeddingConfig:
    """Embedding service configuration settings."""

    url: str = "http://localhost:11434"
    model: str = "embeddinggemma:latest"
    timeout: int = 120
    max_workers: int = 4
    batch_size: int = 50

    @classmethod
    def from_env(cls, prefix: str = "OLLAMA_") -> "EmbeddingConfig":
        """Create embedding config from environment variables."""
        return cls(
            url=os.getenv(f"{prefix}URL", cls.url),
            model=os.getenv(f"{prefix}MODEL", cls.model),
            timeout=int(os.getenv(f"{prefix}TIMEOUT", cls.timeout)),
            max_workers=int(os.getenv(f"{prefix}MAX_WORKERS", cls.max_workers)),
            batch_size=int(os.getenv(f"{prefix}BATCH_SIZE", cls.batch_size)),
        )


@dataclass
class IndexingConfig:
    """Document indexing configuration settings."""

    chunk_size: int = 150
    chunk_overlap: int = 30
    max_chunks_per_article: int = 10
    min_chunk_words: int = 10

    @classmethod
    def from_env(cls, prefix: str = "INDEXING_") -> "IndexingConfig":
        """Create indexing config from environment variables."""
        return cls(
            chunk_size=int(os.getenv(f"{prefix}CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv(f"{prefix}CHUNK_OVERLAP", cls.chunk_overlap)),
            max_chunks_per_article=int(
                os.getenv(f"{prefix}MAX_CHUNKS_PER_ARTICLE", cls.max_chunks_per_article)
            ),
            min_chunk_words=int(
                os.getenv(f"{prefix}MIN_CHUNK_WORDS", cls.min_chunk_words)
            ),
        )


@dataclass
class SearchConfig:
    """Search configuration settings."""

    default_limit: int = 10
    max_limit: int = 100
    min_score: float = 0.0
    fusion_method: str = "rrf"  # "rrf" or "dbsf"
    enable_hybrid_search: bool = True
    enable_sparse: bool = True  # Enable sparse vector search (BM25)

    @classmethod
    def from_env(cls, prefix: str = "SEARCH_") -> "SearchConfig":
        """Create search config from environment variables."""
        return cls(
            default_limit=int(os.getenv(f"{prefix}DEFAULT_LIMIT", cls.default_limit)),
            max_limit=int(os.getenv(f"{prefix}MAX_LIMIT", cls.max_limit)),
            min_score=float(os.getenv(f"{prefix}MIN_SCORE", cls.min_score)),
            fusion_method=os.getenv(f"{prefix}FUSION_METHOD", cls.fusion_method),
            enable_hybrid_search=os.getenv(f"{prefix}ENABLE_HYBRID", "true").lower()
            == "true",
            enable_sparse=os.getenv(f"{prefix}ENABLE_SPARSE", "true").lower()
            == "true",
        )


@dataclass
class LoggingConfig:
    """Logging configuration settings."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None

    @classmethod
    def from_env(cls, prefix: str = "LOG_") -> "LoggingConfig":
        """Create logging config from environment variables."""
        return cls(
            level=os.getenv(f"{prefix}LEVEL", cls.level),
            format=os.getenv(f"{prefix}FORMAT", cls.format),
            file_path=os.getenv(f"{prefix}FILE_PATH", cls.file_path),
        )


@dataclass
class RAGConfig:
    """Complete RAG system configuration."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Global settings
    data_directory: str = "./data"
    temp_directory: str = "./temp"
    max_concurrent_operations: int = 4

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create complete config from environment variables.

        If python-dotenv is available, loads .env file first to make its
        variables available as environment variables.
        """
        # Load .env file if dotenv is available
        if DOTENV_AVAILABLE:
            # Look for .env in current directory and parent directories
            env_path = Path.cwd() / ".env"
            if not env_path.exists():
                # Try parent directory (common in project structures)
                env_path = Path.cwd().parent / ".env"

            if env_path.exists():
                load_dotenv(env_path, override=False)
                logger.debug(f"Loaded environment variables from {env_path}")

        return cls(
            database=DatabaseConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            indexing=IndexingConfig.from_env(),
            search=SearchConfig.from_env(),
            logging=LoggingConfig.from_env(),
            data_directory=os.getenv("DATA_DIRECTORY", cls.data_directory),
            temp_directory=os.getenv("TEMP_DIRECTORY", cls.temp_directory),
            max_concurrent_operations=int(
                os.getenv("MAX_CONCURRENT_OPERATIONS", cls.max_concurrent_operations)
            ),
        )

    @classmethod
    def from_file(cls, config_path: str) -> "RAGConfig":
        """Load configuration from a file (JSON, YAML, or TOML)."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Determine file format and load
        if config_path.endswith(".json"):
            import json

            with open(config_file, "r") as f:
                data = json.load(f)
        elif config_path.endswith((".yml", ".yaml")):
            try:
                import yaml

                with open(config_file, "r") as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML configuration files")
        elif config_path.endswith(".toml"):
            try:
                import tomllib

                with open(config_file, "rb") as f:
                    data = tomllib.load(f)
            except ImportError:
                raise ImportError(
                    "tomllib or toml is required to load TOML configuration files"
                )
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path}")

        # Convert loaded data to config objects
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        """Create config from dictionary data."""
        config = cls()

        # Update database config
        if "database" in data:
            db_data = data["database"]
            config.database = DatabaseConfig(**db_data)

        # Update embedding config
        if "embedding" in data:
            emb_data = data["embedding"]
            config.embedding = EmbeddingConfig(**emb_data)

        # Update indexing config
        if "indexing" in data:
            idx_data = data["indexing"]
            config.indexing = IndexingConfig(**idx_data)

        # Update search config
        if "search" in data:
            search_data = data["search"]
            config.search = SearchConfig(**search_data)

        # Update logging config
        if "logging" in data:
            log_data = data["logging"]
            config.logging = LoggingConfig(**log_data)

        # Update global settings
        for key in ["data_directory", "temp_directory", "max_concurrent_operations"]:
            if key in data:
                setattr(config, key, data[key])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "database": {
                "url": self.database.url,
                "timeout": self.database.timeout,
                "collection_prefix": self.database.collection_prefix,
            },
            "embedding": {
                "url": self.embedding.url,
                "model": self.embedding.model,
                "timeout": self.embedding.timeout,
                "max_workers": self.embedding.max_workers,
                "batch_size": self.embedding.batch_size,
            },
            "indexing": {
                "chunk_size": self.indexing.chunk_size,
                "chunk_overlap": self.indexing.chunk_overlap,
                "max_chunks_per_article": self.indexing.max_chunks_per_article,
                "min_chunk_words": self.indexing.min_chunk_words,
            },
            "search": {
                "default_limit": self.search.default_limit,
                "max_limit": self.search.max_limit,
                "min_score": self.search.min_score,
                "fusion_method": self.search.fusion_method,
                "enable_hybrid_search": self.search.enable_hybrid_search,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file_path": self.logging.file_path,
            },
            "data_directory": self.data_directory,
            "temp_directory": self.temp_directory,
            "max_concurrent_operations": self.max_concurrent_operations,
        }

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to a file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        if config_path.endswith(".json"):
            import json

            with open(config_file, "w") as f:
                json.dump(data, f, indent=2)
        elif config_path.endswith((".yml", ".yaml")):
            try:
                import yaml

                with open(config_file, "w") as f:
                    yaml.dump(data, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML is required to save YAML configuration files")
        elif config_path.endswith(".toml"):
            try:
                import tomllib

                # Note: tomllib is read-only, need toml for writing
                import toml

                with open(config_file, "w") as f:
                    toml.dump(data, f)
            except ImportError:
                raise ImportError("toml is required to save TOML configuration files")
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path}")


# Global configuration instance
_global_config: Optional[RAGConfig] = None


def get_config(config_path: Optional[str] = None) -> RAGConfig:
    """
    Get the global configuration instance.

    Args:
        config_path: Optional path to configuration file

    Returns:
        RAGConfig instance
    """
    global _global_config

    if _global_config is None:
        if config_path and Path(config_path).exists():
            _global_config = RAGConfig.from_file(config_path)
        else:
            _global_config = RAGConfig.from_env()

    return _global_config


def set_config(config: RAGConfig) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Setup logging based on configuration."""
    if config is None:
        config = get_config().logging

    # Configure logging
    logging_kwargs = {
        "level": getattr(logging, config.level.upper()),
        "format": config.format,
    }

    if config.file_path:
        # Log to file
        log_file = Path(config.file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging_kwargs["filename"] = config.file_path

    logging.basicConfig(**logging_kwargs)


def create_sample_config(output_path: str) -> None:
    """Create a sample configuration file."""
    config = RAGConfig()
    config.save_to_file(output_path)
    logger.info(f"Sample configuration saved to: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    config = RAGConfig.from_env()

    print("Current configuration:")
    print(f"  Database URL: {config.database.url}")
    print(f"  Embedding Model: {config.embedding.model}")
    print(f"  Chunk Size: {config.indexing.chunk_size}")
    print(f"  Search Limit: {config.search.default_limit}")
    print(f"  Log Level: {config.logging.level}")

    # Test configuration saving/loading
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        config.save_to_file(temp_path)
        loaded_config = RAGConfig.from_file(temp_path)

        print("\nConfiguration save/load test:")
        print(f"  Original model: {config.embedding.model}")
        print(f"  Loaded model: {loaded_config.embedding.model}")
        print(f"  Match: {config.embedding.model == loaded_config.embedding.model}")

    finally:
        # Clean up
        Path(temp_path).unlink()
