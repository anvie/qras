"""
Exclude file handling for QRAS indexing.

Supports .gitignore-style patterns using fnmatch.
"""

import fnmatch
import logging
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


def load_exclude_patterns(exclude_file: Optional[str] = None) -> List[str]:
    """
    Load exclude patterns from a file.
    
    Args:
        exclude_file: Path to exclude file. If None, looks for .exclude
                     in the script directory.
    
    Returns:
        List of exclude patterns (empty if file not found)
    """
    patterns = []
    
    if exclude_file is None:
        # Look for .exclude in the qras directory
        script_dir = Path(__file__).parent.parent.parent
        exclude_file = script_dir / ".exclude"
    else:
        exclude_file = Path(exclude_file)
    
    if not exclude_file.exists():
        logger.debug(f"No exclude file found at {exclude_file}")
        return patterns
    
    try:
        with open(exclude_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)
        
        if patterns:
            logger.info(f"Loaded {len(patterns)} exclude patterns from {exclude_file}")
        
        return patterns
    
    except Exception as e:
        logger.warning(f"Error reading exclude file {exclude_file}: {e}")
        return []


def should_exclude(file_path: str, patterns: List[str]) -> bool:
    """
    Check if a file should be excluded based on patterns.
    
    Supports:
    - Exact filename match: IDENTITY.md
    - Glob patterns: *.log, test_*.md
    - Directory patterns: docs/*.md, **/temp/*
    
    Args:
        file_path: Path to check (can be relative or absolute)
        patterns: List of exclude patterns
    
    Returns:
        True if file should be excluded
    """
    if not patterns:
        return False
    
    # Get just the filename for simple matches
    path = Path(file_path)
    filename = path.name
    
    for pattern in patterns:
        # Try matching against just the filename
        if fnmatch.fnmatch(filename, pattern):
            logger.debug(f"Excluding {file_path} (matched pattern: {pattern})")
            return True
        
        # Try matching against the full path (for directory patterns)
        if fnmatch.fnmatch(str(file_path), pattern):
            logger.debug(f"Excluding {file_path} (matched pattern: {pattern})")
            return True
        
        # Try matching with ** for recursive patterns
        if "**" in pattern:
            # Convert ** to match any path segment
            pattern_parts = pattern.replace("**", "*")
            if fnmatch.fnmatch(str(file_path), pattern_parts):
                logger.debug(f"Excluding {file_path} (matched pattern: {pattern})")
                return True
    
    return False


def filter_files(file_paths: List[Path], patterns: List[str]) -> List[Path]:
    """
    Filter a list of file paths based on exclude patterns.
    
    Args:
        file_paths: List of Path objects to filter
        patterns: List of exclude patterns
    
    Returns:
        Filtered list with excluded files removed
    """
    if not patterns:
        return file_paths
    
    filtered = []
    excluded_count = 0
    
    for path in file_paths:
        if should_exclude(str(path), patterns):
            excluded_count += 1
        else:
            filtered.append(path)
    
    if excluded_count > 0:
        logger.info(f"Excluded {excluded_count} files based on patterns")
    
    return filtered
