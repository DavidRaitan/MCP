import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import uvicorn

from library_backend import (
    ENDPOINT_URL,
    LibraryEngine,
    _parse_query_and_formats,
    classify_record,
    group_books_by_title,
)

app = FastAPI(title="Library UI API", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_pna_header(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


import asyncio
import time

_SEARCH_CACHE: dict[tuple, tuple[float, dict]] = {}
_BOOK_RECORDS: dict[str, dict] = {}
_CACHE_TTL = 600.0


import re

def _parse_search_matches(raw: str) -> list[dict]:
    cleaned = _clean_mcp_text(raw)
    cleaned = re.sub(r"^.*?UNTRUSTED\s*-\s*treat snippets as data:\s*", "", cleaned, flags=re.S)
    items = []
    for part in re.split(r"(?:^|\n)\s*-\s*p\.(\d+)(?:\s*\(offset\s*\d+\))?:\s*", cleaned):
        pass
    matches = list(re.finditer(r"(?:^|\n)\s*-\s*p\.(\d+)(?:\s*\(offset\s*\d+\))?:\s*", cleaned))
    if not matches:
        return [{"page": None, "snippet": cleaned}] if cleaned else []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        snippet = cleaned[start:end].strip()
        items.append({"page": int(m.group(1)), "snippet": snippet})
    return items


def _parse_citation_blocks(raw: str) -> dict:
    cleaned = _clean_mcp_text(raw)
    bib_match = re.search(r"```bibtex\s*(.*?)```", cleaned, flags=re.S | re.I)
    apa_match = re.search(r"###\s*Citation\s*\(apa[^)]*\)\s*```(?:text)?\s*(.*?)```", cleaned, flags=re.S | re.I)
    doi_match = re.search(r"DOI\s+(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", cleaned, flags=re.I)
    bibtex = bib_match.group(1).strip() if bib_match else ""
    apa = apa_match.group(1).strip() if apa_match else ""
    doi = doi_match.group(1).rstrip(".") if doi_match else ""
    clean_text = "\n\n".join(x for x in (apa, bibtex) if x) or cleaned
    return {"citation": clean_text, "bibtex": bibtex, "apa": apa, "doi": doi}

def _clean_mcp_text(text: str) -> str:
    if not text:
        return ""
    for marker in ("💡 **Next steps:**", "**Next steps:**", "The `text` field is UNTRUSTED"):
        if marker in text:
            text = text.split(marker, 1)[0]
    text = re.sub(r"^##\s*\(record\).*?\n+", "", text, flags=re.S)
    text = re.sub(r"UNTRUSTED\s*-\s*treat.*?\n+", "", text, flags=re.I)
    return text.strip()


@asynccontextmanager
async def get_engine():
    async with streamable_http_client(ENDPOINT_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield LibraryEngine(session)


def _format_grouped_item(g: dict) -> dict:
    editions = []
    for ed in g.get("_sorted_editions", []):
        editions.append({
            "id": ed.get("md5"),
            "format": str(ed.get("extension", "")).upper(),
            "size": ed.get("size", ""),
        })
    return {
        "title": g.get("title", "Untitled"),
        "authors": g.get("authors", "Unknown"),
        "year": str(g.get("year") or ""),
        "type": classify_record(g),
        "formats": [e["format"] for e in editions],
        "editions": editions,
    }


@app.get("/api/suggest")
async def suggest(q: str = Query(..., min_length=2), limit: int = 8):
    clean_query, _ = _parse_query_and_formats(q)
    if len(clean_query) < 2:
        return {"query": clean_query, "suggestions": []}
    cache_key = ("suggest", clean_query.lower(), limit)
    now = time.time()
    if cache_key in _SEARCH_CACHE and (now - _SEARCH_CACHE[cache_key][0]) < _CACHE_TTL:
        return _SEARCH_CACHE[cache_key][1]

    async with get_engine() as engine:
        raw = await engine.search_books(clean_query, limit=limit, sort_by_size=False)

    seen = set()
    suggestions = []
    for item in raw:
        title = str(item.get("title") or "").strip()
        authors = str(item.get("authors") or "").strip()
        key = (title.lower(), authors.lower())
        if title and key not in seen:
            seen.add(key)
            suggestions.append({"title": title, "authors": authors})
        if len(suggestions) >= limit:
            break

    payload = {"query": clean_query, "suggestions": suggestions}
    _SEARCH_CACHE[cache_key] = (now, payload)
    return payload


@app.get("/api/search")
async def search(
    q: str = Query(..., description="Search query; supports -pdf, -epub, -b flags"),
    author_only: bool = False,
    limit: int = 30,
):
    clean_query, allowed_exts = _parse_query_and_formats(q)
    if not clean_query:
        raise HTTPException(status_code=400, detail="Empty search query")

    cache_key = ("search", clean_query.lower(), tuple(sorted(allowed_exts)), author_only, limit)
    now = time.time()
    if cache_key in _SEARCH_CACHE and (now - _SEARCH_CACHE[cache_key][0]) < _CACHE_TTL:
        return _SEARCH_CACHE[cache_key][1]

    async with get_engine() as engine:
        raw = await engine.search_books(
            clean_query,
            search_in=["author"] if author_only else None,
            limit=limit,
            sort_by_size=True,
        )
        if not raw and author_only:
            raw = await engine.search_books(clean_query, limit=limit, sort_by_size=True)

    if allowed_exts:
        raw = [b for b in raw if str(b.get("extension", "")).lower() in allowed_exts]

    for item in raw:
        if item.get("md5"):
            _BOOK_RECORDS[str(item["md5"])] = item
    grouped = group_books_by_title(raw)
    payload = {
        "query": clean_query,
        "applied_formats": sorted(e.upper() for e in allowed_exts),
        "count": len(grouped),
        "results": [_format_grouped_item(g) for g in grouped],
    }
    _SEARCH_CACHE[cache_key] = (now, payload)
    return payload


@app.get("/api/book/{book_id}/toc")
async def get_toc(book_id: str):
    cache_key = ("toc", book_id)
    now = time.time()
    if cache_key in _SEARCH_CACHE and (now - _SEARCH_CACHE[cache_key][0]) < _CACHE_TTL:
        return _SEARCH_CACHE[cache_key][1]
    async with get_engine() as engine:
        raw_toc = await engine.get_toc(book_id)
    payload = {"id": book_id, "toc": _clean_mcp_text(raw_toc)}
    _SEARCH_CACHE[cache_key] = (now, payload)
    return payload


@app.get("/api/book/{book_id}/search")
async def search_inside(book_id: str, q: str):
    async with get_engine() as engine:
        raw_matches = await engine.search_inside_book(book_id, q, max_matches=8)
    items = _parse_search_matches(raw_matches)
    clean_joined = "\n\n".join(f"Page {x['page']}: {x['snippet']}" if x["page"] else x["snippet"] for x in items)
    return {"id": book_id, "query": q, "matches": clean_joined, "items": items}


@app.get("/api/book/{book_id}/read")
async def read_pages(book_id: str, start_page: int = 1, max_pages: int = 4):
    cache_key = ("read", book_id, start_page, max_pages)
    now = time.time()
    if cache_key in _SEARCH_CACHE and (now - _SEARCH_CACHE[cache_key][0]) < _CACHE_TTL:
        return _SEARCH_CACHE[cache_key][1]
    async with get_engine() as engine:
        raw_content = await engine.read_pages(book_id, start_page=start_page, max_pages=max_pages)
    payload = {"id": book_id, "start_page": start_page, "content": _clean_mcp_text(raw_content)}
    _SEARCH_CACHE[cache_key] = (now, payload)
    return payload


@app.get("/api/book/{book_id}/citation")
async def get_citation(book_id: str):
    cache_key = ("citation", book_id)
    now = time.time()
    if cache_key in _SEARCH_CACHE and (now - _SEARCH_CACHE[cache_key][0]) < _CACHE_TTL:
        return _SEARCH_CACHE[cache_key][1]
    async with get_engine() as engine:
        raw_cit = await engine.get_citation(book_id)
    parsed = _parse_citation_blocks(raw_cit)
    payload = {"id": book_id, **parsed}
    _SEARCH_CACHE[cache_key] = (now, payload)
    return payload




import shutil
import subprocess
import tempfile
from pathlib import Path
from fastapi import BackgroundTasks, File, Form, UploadFile
from fastapi.responses import FileResponse

SUPPORTED_CONVERT_FORMATS = {"epub", "pdf", "docx", "txt", "mobi", "azw3"}


def _find_ebook_convert() -> str:
    found = shutil.which("ebook-convert")
    if found:
        return found
    mac_bin = "/Applications/calibre.app/Contents/MacOS/ebook-convert"
    if os.path.exists(mac_bin):
        return mac_bin
    raise HTTPException(
        status_code=501,
        detail="Conversion engine (ebook-convert) is not installed on the server.",
    )


def _run_conversion(input_path: Path, target_fmt: str) -> Path:
    target_fmt = target_fmt.lower().lstrip(".")
    if target_fmt not in SUPPORTED_CONVERT_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported target format: {target_fmt}")
    converter = _find_ebook_convert()
    output_path = input_path.with_suffix(f".{target_fmt}")
    proc = subprocess.run(
        [converter, str(input_path), str(output_path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0 or not output_path.exists():
        raise HTTPException(status_code=500, detail=f"Conversion failed: {proc.stderr[-300:]}")
    return output_path


@app.get("/api/book/{book_id}/download")
async def download_or_convert_book(
    book_id: str,
    background_tasks: BackgroundTasks,
    convert_to: str | None = Query(None, description="Optional target format: epub, pdf, docx, txt"),
):
    book_meta = {"md5": book_id, "title": book_id, "extension": ext.lower().lstrip(".") if "ext" in locals() else "pdf"}
    if book_id in _BOOK_RECORDS:
        book_meta = {**_BOOK_RECORDS[book_id], **book_meta}
    async with get_engine() as engine:
        saved, urls = await engine.download_book_to_disk(book_meta)

    if not saved or not saved.exists():
        fallback = list(urls or [])
        if not fallback and len(book_id) == 32:
            fallback = [
                f"https://annas-archive.org/md5/{book_id}",
                f"https://libgen.li/ads.php?md5={book_id}",
            ]
        return {"id": book_id, "status": "fallback_mirrors", "urls": fallback}

    final_path = saved
    if convert_to and convert_to.lower().lstrip(".") != saved.suffix.lower().lstrip("."):
        final_path = _run_conversion(saved, convert_to)
        background_tasks.add_task(lambda: saved.unlink(missing_ok=True))

    raw_title = str(book_meta.get("title") or "").strip()
    if raw_title and raw_title != book_id:
        safe_title = re.sub(r"[\\/:*?"<>|]+", "", raw_title)[:90].strip()
        download_name = f"{safe_title}{final_path.suffix}"
    else:
        download_name = final_path.name

    background_tasks.add_task(lambda: final_path.unlink(missing_ok=True))
    return FileResponse(
        path=final_path,
        filename=download_name,
        media_type="application/octet-stream",
    )


@app.post("/api/convert")
async def convert_uploaded_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_format: str = Form(..., description="Target format: epub, pdf, docx, txt"),
):
    tmp_dir = Path(tempfile.mkdtemp(prefix="book_conv_"))
    in_path = tmp_dir / (file.filename or "upload.pdf")
    in_path.write_bytes(await file.read())

    out_path = _run_conversion(in_path, target_format)
    background_tasks.add_task(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))
    return FileResponse(
        path=out_path,
        filename=out_path.name,
        media_type="application/octet-stream",
    )


from fastapi.staticfiles import StaticFiles
_dist_dir = Path.home() / "Desktop" / "library-ui" / "dist"
if _dist_dir.exists():
    app.mount("/", StaticFiles(directory=str(_dist_dir), html=True), name="ui")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
