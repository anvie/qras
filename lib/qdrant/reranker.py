"""
LLM-based reranking for search results using Ollama.

This module provides reranking functionality using local LLMs via Ollama
to score and reorder search results based on query relevance.
"""

import logging
import requests
import json
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def clean_markdown_for_rerank(text: str) -> str:
    """
    Clean markdown artifacts from text for LLM reranking.
    Removes formatting noise so LLM can focus on actual content.
    """
    if not text:
        return ""
    
    # Remove code blocks (```...```)
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    
    # Remove inline code (`...`)
    text = re.sub(r'`[^`]+`', ' ', text)
    
    # Remove markdown tables (lines with |)
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        # Skip table separator lines (|---|---|)
        if re.match(r'^[\s|:-]+$', line):
            continue
        # Remove pipe characters from table rows but keep content
        if '|' in line:
            # Extract cell contents, skip empty cells
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if cells:
                clean_lines.append(' '.join(cells))
        else:
            clean_lines.append(line)
    text = '\n'.join(clean_lines)
    
    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_
    
    # Remove links but keep text [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove bullet markers
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
    
    # Remove numbered list markers
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Collapse multiple whitespace/newlines
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


# Default reranking prompt template
RERANK_PROMPT_TEMPLATE = """Kamu adalah penilai relevansi dokumen. Berikan skor 1-10 untuk relevansi dokumen terhadap query.

Query: {query}

Dokumen:
---
{document}
---

Berikan HANYA angka 1-10 (1=tidak relevan, 10=sangat relevan).
Skor:"""


class LLMReranker:
    """Reranker using Ollama LLM for relevance scoring."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen2.5:0.5b",
        timeout: int = 30,
        max_workers: int = 4,
    ):
        """
        Initialize the LLM reranker.

        Args:
            ollama_url: Ollama API URL
            model: Model name to use for reranking
            timeout: Request timeout in seconds
            max_workers: Max concurrent requests for parallel scoring
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_workers = max_workers

    def _score_single(
        self, query: str, document: str, result_index: int
    ) -> tuple[int, float]:
        """
        Score a single document against the query.

        Args:
            query: Search query
            document: Document content to score
            result_index: Original index of the result

        Returns:
            Tuple of (result_index, relevance_score)
        """
        prompt = RERANK_PROMPT_TEMPLATE.format(
            query=query,
            document=document[:1500],  # Truncate long docs
        )

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 10,  # Only need a number
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            response_text = result.get("response", "").strip()

            # Extract score from response
            score = self._extract_score(response_text)
            logger.debug(f"Result {result_index}: score={score} (raw: {response_text})")
            return (result_index, score)

        except Exception as e:
            logger.warning(f"Failed to score result {result_index}: {e}")
            return (result_index, 0.0)

    def _extract_score(self, text: str) -> float:
        """Extract numeric score from LLM response."""
        # Try to find a number 1-10
        match = re.search(r'\b([1-9]|10)\b', text)
        if match:
            return float(match.group(1))
        
        # Fallback: try to parse first number
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            score = float(match.group(1))
            return min(max(score, 1.0), 10.0)  # Clamp to 1-10
        
        return 5.0  # Neutral score if parsing fails

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        parallel: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using LLM scoring.

        Args:
            query: Original search query
            results: Search results to rerank
            top_k: Return only top K results (None = all)
            parallel: Use parallel scoring for speed

        Returns:
            Reranked results with 'llm_score' field added
        """
        if not results:
            return []

        # Prepare documents for scoring (clean markdown artifacts)
        docs_to_score = []
        for i, result in enumerate(results):
            content = result.get("payload", {}).get("content", "")
            title = result.get("payload", {}).get("title", "")
            # Clean markdown for better LLM understanding
            clean_content = clean_markdown_for_rerank(content)
            clean_title = clean_markdown_for_rerank(title)
            doc_text = f"{clean_title}\n{clean_content}" if clean_title else clean_content
            docs_to_score.append((i, doc_text))

        # Score documents
        scores = {}
        
        if parallel and len(docs_to_score) > 1:
            # Parallel scoring
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._score_single, query, doc, idx): idx
                    for idx, doc in docs_to_score
                }
                for future in as_completed(futures):
                    idx, score = future.result()
                    scores[idx] = score
        else:
            # Sequential scoring
            for idx, doc in docs_to_score:
                _, score = self._score_single(query, doc, idx)
                scores[idx] = score

        # Add LLM scores to results
        for i, result in enumerate(results):
            result["llm_score"] = scores.get(i, 0.0)
            # Combined score: blend original + LLM (LLM weighted higher)
            original_score = result.get("score", 0.0)
            result["combined_score"] = (original_score * 0.3) + (scores.get(i, 0.0) / 10.0 * 0.7)

        # Sort by LLM score (or combined score)
        reranked = sorted(results, key=lambda x: x.get("llm_score", 0), reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen2.5:0.5b",
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function for reranking results.

    Args:
        query: Search query
        results: Search results to rerank
        ollama_url: Ollama API URL
        model: Model name
        top_k: Return only top K results

    Returns:
        Reranked results
    """
    reranker = LLMReranker(ollama_url=ollama_url, model=model)
    return reranker.rerank(query, results, top_k=top_k)
