"""
Sparse vector generation for BM25-style keyword search.

This module provides functionality to generate sparse vectors from text
for use in hybrid search combining semantic (dense) and keyword (sparse) matching.
"""

import re
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


def stem_indonesian(word: str) -> str:
    """
    Simple Indonesian stemmer for suffix removal.
    
    Handles common suffixes:
    - Possessive: -nya, -ku, -mu
    - Verb suffixes: -kan, -an
    - Particles: -lah, -kah, -pun
    
    This is a lightweight stemmer focused on matching variations.
    For full morphological analysis, use a proper library like Sastrawi.
    
    Note: Conservative approach - only removes suffixes when result is >= 3 chars
    and the suffix is clearly detachable.
    """
    if len(word) < 4:
        return word
    
    # Common words that should NOT be stemmed (false positives)
    # These look like they have suffixes but they don't
    NO_STEM = {
        'punya', 'hanya', 'dunia', 'rahasia', 'biasa', 'rasa', 
        'masa', 'bisa', 'jasa', 'desa', 'bahasa', 'angkasa',
        'siapa', 'apa', 'dimana', 'kapan', 'mengapa', 'kenapa',
        'saya', 'anda', 'dia', 'kita', 'kami', 'mereka',
    }
    
    if word in NO_STEM:
        return word
    
    original = word
    
    # Remove possessive suffixes first (most common cause of mismatch)
    # Only if remaining stem is >= 3 chars
    if word.endswith('nya') and len(word) > 5:
        word = word[:-3]
    elif word.endswith('ku') and len(word) > 4:
        word = word[:-2]
    elif word.endswith('mu') and len(word) > 4:
        word = word[:-2]
    
    # Remove particles (only if result >= 3 chars)
    if word.endswith('lah') and len(word) > 5:
        word = word[:-3]
    elif word.endswith('kah') and len(word) > 5:
        word = word[:-3]
    elif word.endswith('pun') and len(word) > 5:
        word = word[:-3]
    
    # Remove verb/noun suffixes (conservative - only -kan and -an)
    # Skip -i suffix as it causes too many false positives
    if word.endswith('kan') and len(word) > 5:
        word = word[:-3]
    elif word.endswith('an') and len(word) > 4:
        # Be extra careful with -an, lots of root words end in -an
        # Only remove if word is long enough
        candidate = word[:-2]
        if len(candidate) >= 3:
            word = candidate
    
    return word


# Simple stopwords for filtering (multilingual: EN + ID)
STOPWORDS = {
    # English
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "this",
    "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "also", "now", "here", "there", "then",
    # Indonesian
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "pada", "adalah", "atau", "juga", "sudah", "akan", "bisa", "ada",
    "tidak", "saya", "anda", "kita", "mereka", "dia", "kami", "kalau",
    "jika", "maka", "seperti", "karena", "tetapi", "namun", "hanya",
    "lebih", "sangat", "banyak", "satu", "dua", "tiga", "semua",
}

# Vocabulary size for hash-based indexing
VOCAB_SIZE = 30000


@dataclass
class SparseVector:
    """Sparse vector representation with indices and values."""
    indices: List[int]
    values: List[float]
    
    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary format for Qdrant."""
        return {
            "indices": self.indices,
            "values": self.values
        }


def tokenize(text: str, min_length: int = 2, stem: bool = True) -> List[str]:
    """
    Tokenize text into words with optional Indonesian stemming.
    
    Args:
        text: Input text to tokenize
        min_length: Minimum token length to keep
        stem: Apply Indonesian suffix stemming (default: True)
        
    Returns:
        List of lowercase tokens (stemmed if enabled)
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Split into tokens
    tokens = text.split()
    
    # Filter by length and stopwords
    tokens = [
        t for t in tokens 
        if len(t) >= min_length and t not in STOPWORDS
    ]
    
    # Apply Indonesian stemming if enabled
    if stem:
        tokens = [stem_indonesian(t) for t in tokens]
    
    return tokens


def term_hash(term: str) -> int:
    """
    Hash a term to a vocabulary index using deterministic hashing.
    
    Uses MD5 for consistent hash values across processes and sessions.
    This allows for vocabulary-free sparse vectors.
    
    Args:
        term: The term to hash
        
    Returns:
        Integer index in range [0, VOCAB_SIZE)
    """
    import hashlib
    # Use MD5 for deterministic hashing (consistent across processes)
    hash_bytes = hashlib.md5(term.encode('utf-8')).digest()
    # Take first 4 bytes as integer
    hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
    return hash_int % VOCAB_SIZE


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """
    Compute term frequency (TF) for tokens.
    
    Uses log-normalized TF: 1 + log(count)
    
    Args:
        tokens: List of tokens
        
    Returns:
        Dictionary mapping terms to TF scores
    """
    if not tokens:
        return {}
    
    counts = Counter(tokens)
    tf = {}
    
    for term, count in counts.items():
        # Log-normalized TF
        tf[term] = 1.0 + math.log(count) if count > 0 else 0.0
    
    return tf


def generate_sparse_vector(
    text: str,
    boost_terms: Optional[List[str]] = None,
    boost_factor: float = 2.0
) -> SparseVector:
    """
    Generate a sparse vector from text for BM25-style matching.
    
    Args:
        text: Input text to vectorize
        boost_terms: Optional list of terms to boost (e.g., from title)
        boost_factor: Multiplier for boosted terms
        
    Returns:
        SparseVector with term indices and TF values
    """
    tokens = tokenize(text)
    
    if not tokens:
        # Return empty sparse vector
        return SparseVector(indices=[], values=[])
    
    # Compute TF scores
    tf_scores = compute_tf(tokens)
    
    # Apply boost to specific terms
    if boost_terms:
        boost_tokens = set(tokenize(' '.join(boost_terms)))
        for term in boost_tokens:
            if term in tf_scores:
                tf_scores[term] *= boost_factor
            else:
                # Add boosted term even if not in content
                tf_scores[term] = boost_factor
    
    # Convert to sparse vector format
    indices = []
    values = []
    
    for term, score in tf_scores.items():
        idx = term_hash(term)
        indices.append(idx)
        values.append(score)
    
    # Sort by index for consistent ordering
    sorted_pairs = sorted(zip(indices, values), key=lambda x: x[0])
    
    # Handle duplicate indices (hash collisions) by summing values
    merged = {}
    for idx, val in sorted_pairs:
        if idx in merged:
            merged[idx] += val
        else:
            merged[idx] = val
    
    final_indices = list(merged.keys())
    final_values = list(merged.values())
    
    return SparseVector(indices=final_indices, values=final_values)


def generate_query_sparse_vector(query: str) -> SparseVector:
    """
    Generate a sparse vector optimized for queries.
    
    Queries are typically short, so we use simpler weighting.
    
    Args:
        query: Search query text
        
    Returns:
        SparseVector for the query
    """
    tokens = tokenize(query, min_length=1)  # Allow shorter tokens for queries
    
    if not tokens:
        return SparseVector(indices=[], values=[])
    
    # For queries, use binary weighting (1.0 for presence)
    # This works better for short queries
    term_set = set(tokens)
    
    indices = [term_hash(term) for term in term_set]
    values = [1.0] * len(indices)
    
    # Sort and merge duplicates
    merged = {}
    for idx, val in zip(indices, values):
        merged[idx] = merged.get(idx, 0) + val
    
    return SparseVector(
        indices=list(merged.keys()),
        values=list(merged.values())
    )


# Convenience function for backward compatibility
def text_to_sparse(text: str) -> Dict[str, List]:
    """Convert text to sparse vector dictionary format."""
    sv = generate_sparse_vector(text)
    return sv.to_dict()


def query_to_sparse(query: str) -> Dict[str, List]:
    """Convert query to sparse vector dictionary format."""
    sv = generate_query_sparse_vector(query)
    return sv.to_dict()
