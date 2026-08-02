"""
ChargeFlow AI V2 — RAG Text Chunker
=====================================
Splits raw document texts into overlapping, metadata-rich passages for vector indexing.

Key Features:
  - Preserves section headers in Markdown (#, ##, ###)
  - Chunk size: configurable (default ~500 chars)
  - Overlap size: configurable (default ~100 chars)
  - Deterministic chunk_id generation: `{source_slug}_chunk_{idx:03d}`
  - Metadata preservation: source path, document type, chunk index
"""

import re
from typing import Dict, List, Any


class TextChunker:
    """
    Splits documents into overlapping chunks with source metadata.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _slugify(self, text: str) -> str:
        slug = re.sub(r'[^a-zA-Z0-9]', '_', text.lower())
        return re.sub(r'_+', '_', slug).strip('_')

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a list of raw documents into structured chunk dicts:
            [
                {
                    "chunk_id": "docs_problem_framing_md_chunk_001",
                    "source": "docs/problem_framing.md",
                    "section_title": "Overview",
                    "text": "...",
                    "type": "markdown_doc"
                },
                ...
            ]
        """
        all_chunks = []

        for doc in documents:
            source = doc["source"]
            content = doc["content"]
            doc_type = doc.get("type", "unknown")
            source_slug = self._slugify(source)

            if doc_type == "markdown_doc":
                # Split by markdown headers (#, ##, ###) or paragraphs
                sections = re.split(r'\n(?=#{1,3}\s)', content)
                chunk_counter = 0

                for section in sections:
                    lines = section.strip().split('\n')
                    section_title = "General"
                    if lines and lines[0].startswith('#'):
                        section_title = lines[0].lstrip('#').strip()

                    section_text = section.strip()
                    if not section_text:
                        continue

                    # Sub-chunk section if larger than chunk_size
                    if len(section_text) <= self.chunk_size:
                        chunk_counter += 1
                        all_chunks.append({
                            "chunk_id": f"{source_slug}_chunk_{chunk_counter:03d}",
                            "source": source,
                            "section_title": section_title,
                            "text": section_text,
                            "type": doc_type,
                        })
                    else:
                        start = 0
                        while start < len(section_text):
                            end = start + self.chunk_size
                            chunk_text = section_text[start:end].strip()
                            if chunk_text:
                                chunk_counter += 1
                                all_chunks.append({
                                    "chunk_id": f"{source_slug}_chunk_{chunk_counter:03d}",
                                    "source": source,
                                    "section_title": section_title,
                                    "text": chunk_text,
                                    "type": doc_type,
                                })
                            start += (self.chunk_size - self.chunk_overlap)
            else:
                # Direct fixed-size chunking for JSON / CSV artifacts
                start = 0
                chunk_counter = 0
                while start < len(content):
                    end = start + self.chunk_size
                    chunk_text = content[start:end].strip()
                    if chunk_text:
                        chunk_counter += 1
                        all_chunks.append({
                            "chunk_id": f"{source_slug}_chunk_{chunk_counter:03d}",
                            "source": source,
                            "section_title": f"Artifact Data Part {chunk_counter}",
                            "text": chunk_text,
                            "type": doc_type,
                        })
                    start += (self.chunk_size - self.chunk_overlap)

        return all_chunks
