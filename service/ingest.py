"""
Document ingestion for Neo-Cortex knowledge collection.

Reads markdown files from data/raw_docs/, chunks them by sections,
and stores them in the cortex_knowledge ChromaDB collection.

Usage:
    python -m service.ingest [--clear] [--path PATH]
"""

import hashlib
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from .config import CORTEX_ROOT, COLLECTION_KNOWLEDGE
from .storage.base import MemoryRecord
from .cortex_engine import CortexEngine

logger = logging.getLogger(__name__)

RAW_DOCS_PATH = CORTEX_ROOT / "data" / "raw_docs"

# Directories to skip (backups, caches, non-doc content)
SKIP_DIRS = {"__pycache__", "service", ".git", "web"}

# Chunking settings
MAX_CHUNK_CHARS = 1500
OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 80


def discover_docs(root: Path) -> List[Path]:
    """Find all markdown files to ingest."""
    docs = []
    for path in sorted(root.rglob("*.md")):
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        docs.append(path)
    return docs


def extract_topic(path: Path, root: Path) -> str:
    """Get topic from first directory under root."""
    try:
        return path.relative_to(root).parts[0]
    except (ValueError, IndexError):
        return "general"


def extract_title(content: str, path: Path) -> str:
    """Get title from first H1 heading or filename."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else path.stem.replace("-", " ").title()


def chunk_by_sections(content: str) -> List[Tuple[str, str]]:
    """
    Split markdown into (heading, text) chunks by ## and ### headings.
    Large sections get sub-chunked by paragraphs.
    """
    # Strip frontmatter
    if content.startswith("---"):
        end = content.find("\n---\n", 3)
        if end != -1:
            content = content[end + 5:]

    chunks = []
    current_heading = "Introduction"
    current_lines = []

    for line in content.split("\n"):
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading_match:
            # Flush previous section
            text = "\n".join(current_lines).strip()
            if text and len(text) >= MIN_CHUNK_CHARS:
                for sub in _sub_chunk(text):
                    chunks.append((current_heading, sub))

            current_heading = heading_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    text = "\n".join(current_lines).strip()
    if text and len(text) >= MIN_CHUNK_CHARS:
        for sub in _sub_chunk(text):
            chunks.append((current_heading, sub))

    return chunks


def _sub_chunk(text: str) -> List[str]:
    """Split large text blocks into smaller chunks."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > MAX_CHUNK_CHARS and current:
            chunks.append("\n\n".join(current))
            # Overlap: keep last paragraph
            if current and len(current[-1]) <= OVERLAP_CHARS:
                current = [current[-1]]
                current_len = len(current[-1])
            else:
                current = []
                current_len = 0
        current.append(para)
        current_len += para_len

    if current:
        joined = "\n\n".join(current)
        if len(joined) >= MIN_CHUNK_CHARS or not chunks:
            chunks.append(joined)

    return chunks


def make_chunk_id(source: str, heading: str, index: int, content: str) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{source}:{heading}:{index}:{content[:100]}"
    return f"kb_{hashlib.md5(raw.encode()).hexdigest()[:16]}"


def ingest_docs(root: Path = RAW_DOCS_PATH, clear: bool = False) -> dict:
    """
    Ingest all markdown docs into cortex_knowledge.

    Args:
        root: Path to raw docs directory
        clear: If True, clear existing knowledge before ingesting

    Returns:
        Stats dict with counts
    """
    engine = CortexEngine()
    engine.initialize()

    if clear:
        # Delete all existing knowledge entries
        existing = engine.storage.list_all(COLLECTION_KNOWLEDGE)
        if existing:
            ids = [r.id for r in existing]
            engine.storage.delete(COLLECTION_KNOWLEDGE, ids)
            print(f"Cleared {len(ids)} existing knowledge entries")

    docs = discover_docs(root)
    print(f"Found {len(docs)} markdown files to ingest")

    total_chunks = 0
    file_count = 0
    errors = []

    for doc_path in docs:
        try:
            content = doc_path.read_text(encoding="utf-8", errors="replace")
            topic = extract_topic(doc_path, root)
            title = extract_title(content, doc_path)
            rel_path = str(doc_path.relative_to(root))

            sections = chunk_by_sections(content)
            if not sections:
                continue

            records = []
            for i, (heading, chunk_text) in enumerate(sections):
                chunk_id = make_chunk_id(rel_path, heading, i, chunk_text)

                # Prefix with context for better retrieval
                context_prefix = f"[{topic}/{title}] {heading}\n\n"
                full_content = context_prefix + chunk_text

                record = MemoryRecord(
                    id=chunk_id,
                    content=full_content,
                    agent_id="SYSTEM",
                    visibility="shared",
                    layer="cortex",
                    message_type="fact",
                    tags=[topic, title.lower().replace(" ", "-")],
                    created_at=datetime.now(),
                    attention_weight=1.0,
                )
                records.append(record)

            if records:
                engine.storage.add(COLLECTION_KNOWLEDGE, records)
                total_chunks += len(records)
                file_count += 1
                print(f"  [{file_count}/{len(docs)}] {rel_path}: {len(records)} chunks")

        except Exception as e:
            errors.append((str(doc_path), str(e)))
            print(f"  ERROR: {doc_path.name}: {e}")

    stats = {
        "files_processed": file_count,
        "total_chunks": total_chunks,
        "errors": len(errors),
        "error_details": errors,
    }

    print(f"\nIngestion complete: {file_count} files, {total_chunks} chunks")
    if errors:
        print(f"Errors: {len(errors)}")

    # Verify
    count = engine.storage.count(COLLECTION_KNOWLEDGE)
    print(f"Knowledge collection now has {count} entries")

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clear = "--clear" in sys.argv

    custom_path = None
    if "--path" in sys.argv:
        idx = sys.argv.index("--path")
        if idx + 1 < len(sys.argv):
            custom_path = Path(sys.argv[idx + 1])

    root = custom_path or RAW_DOCS_PATH
    ingest_docs(root=root, clear=clear)
