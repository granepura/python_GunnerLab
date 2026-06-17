"""Fetch and cache web content from MCCE-related sites for RAG indexing."""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

WEB_SOURCES = {
    "mcce4_tutorial": {
        "start_url": "https://gunnerlab.github.io/mcce4_tutorial/",
        "type": "html_crawl",
        "max_pages": 100,
        "description": "MCCE4 Tutorial (GitHub Pages)",
    },
    "mcce4_alpha": {
        "start_url": "https://api.github.com/repos/GunnerLab/MCCE4-Alpha",
        "type": "github_repo",
        "extensions": [".py", ".ftpl", ".md", ".txt", ".sh", ".prm"],
        "dirs": ["bin", "MCCE_bin", "param", "doc", "docs", "runprms", "schedulers"],
        "max_files": 500,
        "description": "MCCE4-Alpha repository source",
    },
    "mcce4_dev": {
        "start_url": "https://api.github.com/repos/GunnerLab/MCCE4",
        "type": "github_repo",
        "extensions": [".py", ".ftpl", ".md", ".txt", ".sh", ".prm"],
        "dirs": ["bin", "MCCE_bin", "param", "doc", "docs", "runprms", "schedulers"],
        "max_files": 500,
        "description": "MCCE4 dev repository source (private — requires GITHUB_TOKEN)",
    },
    "mcce_wiki": {
        "start_url": "https://sites.google.com/site/mccewiki/home",
        "type": "html_crawl",
        "max_pages": 80,
        "description": "MCCE Wiki (Google Sites)",
    },
}


def _require_requests():
    try:
        import requests
        return requests
    except ImportError:
        print("Error: 'requests' is not installed.", file=sys.stderr)
        print("Install with:  pip install requests", file=sys.stderr)
        sys.exit(1)


def _require_bs4():
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup
    except ImportError:
        print("Error: 'beautifulsoup4' is not installed.", file=sys.stderr)
        print("Install with:  pip install beautifulsoup4", file=sys.stderr)
        sys.exit(1)


def _url_to_filename(url: str) -> str:
    h = hashlib.md5(url.encode()).hexdigest()[:12]
    parsed = urlparse(url)
    slug = re.sub(r"[^a-zA-Z0-9]", "_", parsed.path.strip("/"))[:60]
    return f"{slug}_{h}"


def _get_cache_dir(index_dir: str) -> str:
    cache_dir = os.path.join(index_dir, "web_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _is_cached(cache_dir: str, url: str) -> bool:
    fname = _url_to_filename(url)
    meta_path = os.path.join(cache_dir, f"{fname}.json")
    return os.path.exists(meta_path)


def _read_cached(cache_dir: str, url: str) -> dict:
    fname = _url_to_filename(url)
    meta_path = os.path.join(cache_dir, f"{fname}.json")
    text_path = os.path.join(cache_dir, f"{fname}.txt")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    if os.path.exists(text_path):
        with open(text_path) as f:
            meta["text"] = f.read()
    return meta


def _write_cache(cache_dir: str, url: str, text: str, metadata: dict):
    fname = _url_to_filename(url)
    meta = {**metadata, "url": url, "cached_at": time.time()}
    with open(os.path.join(cache_dir, f"{fname}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(cache_dir, f"{fname}.txt"), "w") as f:
        f.write(text)


def _extract_text_from_html(html: str, BeautifulSoup) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


def _extract_links(html: str, base_url: str, BeautifulSoup) -> list:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        links.append(full)
    return links


def _crawl_html_site(source_config: dict, cache_dir: str, refresh: bool) -> list:
    requests = _require_requests()
    BeautifulSoup = _require_bs4()

    start_url = source_config["start_url"]
    max_pages = source_config.get("max_pages", 50)
    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc
    base_path_prefix = parsed_start.path.rstrip("/").rsplit("/", 1)[0] if "/" in parsed_start.path.strip("/") else ""

    visited = set()
    queue = [start_url]
    pages = []

    print(f"  Crawling {start_url} ...")

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        url = url.split("#")[0].split("?")[0]
        if url in visited:
            continue
        visited.add(url)

        if not refresh and _is_cached(cache_dir, url):
            cached = _read_cached(cache_dir, url)
            if cached and cached.get("text"):
                pages.append(cached)
                continue

        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "mcce-chat-indexer/0.1"})
            if resp.status_code != 200:
                continue
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type:
                continue
        except Exception:
            continue

        text = _extract_text_from_html(resp.text, BeautifulSoup)
        if len(text.strip()) < 50:
            continue

        meta = {"url": url, "source": source_config.get("description", "web"),
                "title": ""}
        soup = BeautifulSoup(resp.text, "html.parser")
        title_tag = soup.find("title")
        if title_tag:
            meta["title"] = title_tag.get_text(strip=True)

        _write_cache(cache_dir, url, text, meta)
        meta["text"] = text
        pages.append(meta)

        links = _extract_links(resp.text, url, BeautifulSoup)
        for link in links:
            parsed = urlparse(link)
            if parsed.netloc == base_domain and link.split("#")[0] not in visited:
                link_clean = link.split("#")[0].split("?")[0]
                if link_clean not in visited:
                    queue.append(link_clean)

        time.sleep(0.3)

    print(f"    Fetched {len(pages)} pages from {base_domain}")
    return pages


def _fetch_github_repo(source_config: dict, cache_dir: str, refresh: bool) -> list:
    requests = _require_requests()

    api_url = source_config["start_url"]
    extensions = set(source_config.get("extensions", [".py", ".md"]))
    target_dirs = source_config.get("dirs", [])
    max_files = source_config.get("max_files", 500)

    print(f"  Fetching from GitHub: {api_url} ...")

    pages = []
    headers = {"Accept": "application/vnd.github.v3+json",
               "User-Agent": "mcce-chat-indexer/0.1"}
    gh_token = os.environ.get("GITHUB_TOKEN")
    if gh_token:
        headers["Authorization"] = f"token {gh_token}"

    files_fetched = 0
    for target_dir in target_dirs:
        if files_fetched >= max_files:
            break

        tree_url = f"{api_url}/contents/{target_dir}"
        try:
            resp = requests.get(tree_url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            items = resp.json()
            if not isinstance(items, list):
                continue
        except Exception:
            continue

        file_items = []
        dir_queue = [(items, target_dir)]

        while dir_queue and files_fetched < max_files:
            current_items, current_path = dir_queue.pop(0)
            for item in current_items:
                if files_fetched >= max_files:
                    break
                if item.get("type") == "dir" and item["name"] != "__pycache__":
                    try:
                        sub_resp = requests.get(item["url"], headers=headers, timeout=15)
                        if sub_resp.status_code == 200:
                            sub_items = sub_resp.json()
                            if isinstance(sub_items, list):
                                dir_queue.append((sub_items, item["path"]))
                        time.sleep(0.2)
                    except Exception:
                        pass
                elif item.get("type") == "file":
                    _, ext = os.path.splitext(item["name"])
                    if ext in extensions:
                        file_items.append(item)
                        files_fetched += 1

        for item in file_items:
            file_url = item.get("download_url", "")
            if not file_url:
                continue

            if not refresh and _is_cached(cache_dir, file_url):
                cached = _read_cached(cache_dir, file_url)
                if cached and cached.get("text"):
                    pages.append(cached)
                    continue

            try:
                resp = requests.get(file_url, timeout=15, headers=headers)
                if resp.status_code != 200:
                    continue
                text = resp.text
                if len(text.strip()) < 10:
                    continue
            except Exception:
                continue

            meta = {"url": file_url, "source": source_config.get("description", "github"),
                    "title": item["path"], "filepath": item["path"]}
            _write_cache(cache_dir, file_url, text, meta)
            meta["text"] = text
            pages.append(meta)
            time.sleep(0.2)

    print(f"    Fetched {len(pages)} files from GitHub")
    return pages


def fetch_all_web_sources(index_dir: str, refresh: bool = False) -> list:
    cache_dir = _get_cache_dir(index_dir)
    all_pages = []

    for name, config in WEB_SOURCES.items():
        print(f"\n  [{name}] {config['description']}")
        source_type = config["type"]
        try:
            if source_type == "html_crawl":
                pages = _crawl_html_site(config, cache_dir, refresh)
            elif source_type == "github_repo":
                pages = _fetch_github_repo(config, cache_dir, refresh)
            else:
                continue
            all_pages.extend(pages)
        except Exception as e:
            print(f"    Warning: Failed to fetch {name}: {e}")

    return all_pages


def chunk_web_pages(pages: list) -> list:
    """Chunk fetched web pages for indexing."""
    chunks = []
    for page in pages:
        text = page.get("text", "")
        if not text or len(text.strip()) < 30:
            continue

        url = page.get("url", "")
        source = page.get("source", "web")
        title = page.get("title", "")
        filepath = page.get("filepath", "")

        _, ext = os.path.splitext(filepath) if filepath else ("", "")
        if ext == ".py":
            py_chunks = _chunk_python_web(text, url, filepath, source)
            if py_chunks:
                chunks.extend(py_chunks)
                continue
        elif ext == ".ftpl":
            ftpl_chunks = _chunk_ftpl_web(text, url, filepath, source)
            if ftpl_chunks:
                chunks.extend(ftpl_chunks)
                continue

        lines = text.splitlines()
        window = 40
        overlap = 10
        i = 0
        idx = 0
        name_base = title or os.path.basename(urlparse(url).path) or "page"
        name_base = re.sub(r"[^a-zA-Z0-9_.-]", "_", name_base)[:50]

        while i < len(lines):
            chunk_text = "\n".join(lines[i:i + window]).strip()
            if len(chunk_text) > 30:
                chunks.append({
                    "text": chunk_text[:3000],
                    "filepath": filepath or url,
                    "chunk_type": f"web_{source.split()[0].lower()}",
                    "name": f"{name_base}_chunk{idx}",
                })
                idx += 1
            i += window - overlap

    return chunks


def _chunk_python_web(text: str, url: str, filepath: str, source: str) -> list:
    """Chunk Python source fetched from the web using AST."""
    import ast
    chunks = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines(keepends=True)
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
                "filepath": filepath or url,
                "chunk_type": f"web_python_{kind}",
                "name": node.name,
            })
    return chunks


def _chunk_ftpl_web(text: str, url: str, filepath: str, source: str) -> list:
    """Chunk .ftpl files fetched from the web."""
    chunks = []
    fname = os.path.basename(filepath) if filepath else "unknown.ftpl"
    res_name = fname.replace(".ftpl", "")

    blocks = text.split("\n\n")
    for i, block in enumerate(blocks):
        block = block.strip()
        if len(block) < 10:
            continue
        chunks.append({
            "text": block[:2000],
            "filepath": filepath or url,
            "chunk_type": "web_ftpl_block",
            "name": f"{res_name}_block{i}",
        })
    return chunks
