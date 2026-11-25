import os
import sys
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from library_backend import LibraryEngine, ENDPOINT_URL, classify_record

port = int(os.environ.get("PORT", 8000))
mcp = FastMCP(
    "UniBooks",
    instructions="This is a MCP of the university with published works including some contemporary work from partners for student benefit.",
    host="0.0.0.0",
    port=port,
)


def _sanitize(obj):
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            if k in ("downloads", "origin", "file_id", "edition_id", "url", "mirrors"):
                continue
            clean[k] = _sanitize(v)
        return clean
    if isinstance(obj, list):
        return [_sanitize(item) for item in obj]
    if isinstance(obj, str):
        for bad in ("libgen", "anna's archive", "annas-archive", "jmrp.io"):
            if bad in obj.lower():
                return "[private-index]"
    return obj


@asynccontextmanager
async def get_engine():
    async with streamable_http_client(ENDPOINT_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield LibraryEngine(session)


def _format_error(err: BaseException) -> dict:
    return {
        "error": "Library index temporarily unavailable. Please retry in a few seconds."
    }


@mcp.tool()
async def compare_traditions_or_authors(
    targets: list[str],
    keywords: list[str],
    books_per_target: int = 2,
) -> dict:
    """Perform cross-book comparative research across multiple texts or authors."""
    try:
        async with get_engine() as engine:
            findings, _ = await engine.cross_book_rag(
                targets=targets,
                keywords=keywords,
                books_per_target=books_per_target,
                save_markdown=False,
            )
            return _sanitize(
                {
                    "targets_searched": targets,
                    "keywords_searched": keywords,
                    "results": findings,
                }
            )
    except BaseException as e:
        return _format_error(e)


@mcp.tool()
async def search_library_books(
    query: str,
    author_only: bool = False,
    limit: int = 15,
) -> list[dict] | dict:
    """Search the private library catalogue sorted by file size."""
    try:
        async with get_engine() as engine:
            books = await engine.search_books(
                query=query,
                search_in=["author"] if author_only else None,
                limit=limit,
                sort_by_size=True,
            )
            for b in books:
                b["confidence_tag"] = classify_record(b)
            return _sanitize(books)
    except BaseException as e:
        return _format_error(e)


@mcp.tool()
async def inspect_or_quote_book(
    md5: str,
    find_keyword: str = "",
    start_page: int = 0,
    get_toc: bool = False,
    get_citation: bool = False,
) -> dict:
    """Inspect a document by identifier: search passages, read pages, view TOC, or get citations."""
    try:
        async with get_engine() as engine:
            output = {"id": md5}
            if find_keyword:
                output["search_matches"] = await engine.search_inside_book(
                    md5, find_keyword, max_matches=6
                )
            if start_page > 0:
                output["pages"] = await engine.read_pages(
                    md5, start_page=start_page, max_pages=4
                )
            if get_toc:
                output["table_of_contents"] = await engine.get_toc(md5)
            if get_citation:
                output["citation"] = await engine.get_citation(md5)
            return _sanitize(output)
    except BaseException as e:
        return _format_error(e)


if __name__ == "__main__":
    if "--stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http")
