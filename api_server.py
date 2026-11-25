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


@app.get("/api/search")
async def search(
    q: str = Query(..., description="Search query; supports -pdf, -epub, -b flags"),
    author_only: bool = False,
    limit: int = 30,
):
    clean_query, allowed_exts = _parse_query_and_formats(q)
    if not clean_query:
        raise HTTPException(status_code=400, detail="Empty search query")

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

    grouped = group_books_by_title(raw)
    return {
        "query": clean_query,
        "applied_formats": sorted(e.upper() for e in allowed_exts),
        "count": len(grouped),
        "results": [_format_grouped_item(g) for g in grouped],
    }


@app.get("/api/book/{book_id}/toc")
async def get_toc(book_id: str):
    async with get_engine() as engine:
        return {"id": book_id, "toc": await engine.get_toc(book_id)}


@app.get("/api/book/{book_id}/search")
async def search_inside(book_id: str, q: str):
    async with get_engine() as engine:
        return {"id": book_id, "query": q, "matches": await engine.search_inside_book(book_id, q, max_matches=8)}


@app.get("/api/book/{book_id}/read")
async def read_pages(book_id: str, start_page: int = 1, max_pages: int = 4):
    async with get_engine() as engine:
        return {"id": book_id, "start_page": start_page, "content": await engine.read_pages(book_id, start_page=start_page, max_pages=max_pages)}


@app.get("/api/book/{book_id}/citation")
async def get_citation(book_id: str):
    async with get_engine() as engine:
        return {"id": book_id, "citation": await engine.get_citation(book_id)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
