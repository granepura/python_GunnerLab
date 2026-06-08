"""Build a ChromaDB index from MCCE4 source files for RAG retrieval."""

import ast
import os
import sys
from pathlib import Path


def _require_chromadb():
    try:
        import chromadb
        return chromadb
    except ImportError:
        print("Error: chromadb is not installed.", file=sys.stderr)
        print("Install with:  pip install chromadb sentence-transformers", file=sys.stderr)
        sys.exit(1)


def _get_embedding_function():
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        return SentenceTransformerEmbeddingFunction(model_name="nomic-ai/nomic-embed-text-v1",
                                                    trust_remote_code=True)
    except ImportError:
        print("Error: sentence-transformers is not installed.", file=sys.stderr)
        print("Install with:  pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)


def _chunk_python_ast(filepath: str) -> list:
    """Split a Python file into function/class-level chunks using AST."""
    chunks = []
    try:
        with open(filepath) as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, OSError):
        return _chunk_sliding_window(filepath)

    lines = source.splitlines(keepends=True)
    rel_path = filepath

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 20
            chunk_text = "".join(lines[start:end])
            if len(chunk_text.strip()) < 10:
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append({
                "text": chunk_text[:3000],
                "filepath": rel_path,
                "chunk_type": f"python_{kind}",
                "name": node.name,
            })

    if not chunks:
        return _chunk_sliding_window(filepath)
    return chunks


def _chunk_ftpl(filepath: str) -> list:
    """Split an .ftpl file into residue-block chunks."""
    chunks = []
    try:
        with open(filepath) as f:
            content = f.read()
    except OSError:
        return chunks

    rel_path = filepath
    fname = os.path.basename(filepath)
    res_name = fname.replace(".ftpl", "")

    blocks = content.split("\n\n")
    for i, block in enumerate(blocks):
        block = block.strip()
        if len(block) < 10:
            continue
        chunks.append({
            "text": block[:2000],
            "filepath": rel_path,
            "chunk_type": "ftpl_block",
            "name": f"{res_name}_block{i}",
        })

    if not chunks and content.strip():
        chunks.append({
            "text": content[:3000],
            "filepath": rel_path,
            "chunk_type": "ftpl_file",
            "name": res_name,
        })
    return chunks


def _chunk_sliding_window(filepath: str, window: int = 60, overlap: int = 15) -> list:
    """Generic sliding-window chunker for text files."""
    chunks = []
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return chunks

    rel_path = filepath
    fname = os.path.basename(filepath)
    i = 0
    idx = 0
    while i < len(lines):
        chunk_lines = lines[i:i + window]
        text = "".join(chunk_lines).strip()
        if len(text) > 20:
            chunks.append({
                "text": text[:3000],
                "filepath": rel_path,
                "chunk_type": "text_window",
                "name": f"{fname}_chunk{idx}",
            })
            idx += 1
        i += window - overlap

    return chunks


def collect_source_files(root_dir: str) -> list:
    """Collect all files to index from the MCCE4 source tree."""
    files = []
    root = Path(root_dir)

    py_dirs = ["bin", "MCCE_bin"]
    for d in py_dirs:
        dirpath = root / d
        if dirpath.is_dir():
            for f in dirpath.rglob("*.py"):
                if "__pycache__" not in str(f) and ".mcce_chat_index" not in str(f):
                    files.append(("python", str(f)))

    param_dir = root / "param"
    if param_dir.is_dir():
        for f in param_dir.glob("*.ftpl"):
            files.append(("ftpl", str(f)))

    text_dirs = ["doc", "docs", "runprms", "schedulers"]
    text_exts = {".md", ".txt", ".sh", ".py", ".prm", ".crgms"}
    for d in text_dirs:
        dirpath = root / d
        if dirpath.is_dir():
            for f in dirpath.rglob("*"):
                if f.is_file() and f.suffix in text_exts:
                    files.append(("text", str(f)))

    return files


def build_index(root_dir: str, index_dir: str = None, force: bool = False):
    """Build or rebuild the ChromaDB index."""
    chromadb = _require_chromadb()
    ef = _get_embedding_function()

    if index_dir is None:
        index_dir = os.path.join(root_dir, "bin", ".mcce_chat_index")

    os.makedirs(index_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=index_dir)

    if force:
        try:
            client.delete_collection("mcce4_source")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name="mcce4_source",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not force:
        print(f"Index already exists with {collection.count()} chunks at {index_dir}")
        print("Use --rebuild-index to force rebuild.")
        return

    source_files = collect_source_files(root_dir)
    print(f"Indexing {len(source_files)} files from {root_dir} ...")

    all_chunks = []
    for ftype, fpath in source_files:
        if ftype == "python":
            all_chunks.extend(_chunk_python_ast(fpath))
        elif ftype == "ftpl":
            all_chunks.extend(_chunk_ftpl(fpath))
        else:
            all_chunks.extend(_chunk_sliding_window(fpath))

    if not all_chunks:
        print("No chunks found to index.")
        return

    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        documents = [c["text"] for c in batch]
        metadatas = [{"filepath": c["filepath"], "chunk_type": c["chunk_type"],
                      "name": c["name"]} for c in batch]
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Indexed {len(all_chunks)} chunks into {index_dir}")
