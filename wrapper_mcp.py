import os
import sys
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from library_backend import LibraryEngine, ENDPOINT_URL, classify_record

port = int(os.environ.get("PORT", 8000))
mcp = FastMCP("ComparativeResearchMCP", host="0.0.0.0", port=port)


@asynccontextmanager
async def get_engine():
    async with streamable_http_client(ENDPOINT_URL, timeout=30.0) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield LibraryEngine(session)


def _format_error(err: BaseException) -> dict:
    if isinstance(err, BaseExceptionGroup):
        details = [f"{type(e).__name__}: {e}" for e in err.exceptions]
        return {"error": "Upstream libgen-mcp connection failed", "details": details}
    return {"error": f"{type(err).__name__}: {err}"}


@mcp.tool()
async def compare_traditions_or_authors(
    targets: list[str],
    keywords: list[str],
    books_per_target: int = 2,
) -> dict:
    """
    Perform Cross-Book / Cross-Tradition RAG by scanning multiple primary texts or authors
    in parallel without downloading files locally.
    """
    try:
        async with get_engine() as engine:
            findings, _ = await engine.cross_book_rag(
                targets=targets,
                keywords=keywords,
                books_per_target=books_per_target,
                save_markdown=False,
            )
            return {
                "targets_searched": targets,
                "keywords_searched": keywords,
                "results": findings,
            }
    except BaseException as e:
        return _format_error(e)


@mcp.tool()
async def search_library_books(
    query: str,
    author_only: bool = False,
    limit: int = 15,
) -> list[dict] | dict:
    """
    Search for books sorted heaviest-to-lightest, tagged with confidence labels
    ('[BOOK]' for full books vs '[PAPER]' for short articles/reviews), along with MD5 identifiers.
    """
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
            return books
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
    """
    Inspect a specific book by its MD5 hash: search for exact quotes (find_keyword),
    read pages (start_page > 0), fetch the table of contents (get_toc=True), or
    generate BibTeX/APA citations (get_citation=True).
    """
    try:
        async with get_engine() as engine:
            output = {"md5": md5}
            if find_keyword:
                output["search_matches"] = await engine.search_inside_book(md5, find_keyword, max_matches=6)
            if start_page > 0:
                output["pages"] = await engine.read_pages(md5, start_page=start_page, max_pages=4)
            if get_toc:
                output["table_of_contents"] = await engine.get_toc(md5)
            if get_citation:
                output["citation"] = await engine.get_citation(md5)
            return output
    except BaseException as e:
        return _format_error(e)


@mcp.tool()
async def download_book(md5: str, title: str = "book", authors: str = "Unknown", extension: str = "pdf") -> dict:
    """
    Return resolved mirror download URLs (and save to ~/Downloads when running locally).
    """
    try:
        async with get_engine() as engine:
            book_meta = {"md5": md5, "title": title, "authors": authors, "extension": extension}
            saved_path, urls = await engine.download_book_to_disk(book_meta)
            return {
                "md5": md5,
                "saved_to": str(saved_path.resolve()) if saved_path else None,
                "fallback_urls": urls,
            }
    except BaseException as e:
        return _format_error(e)


if __name__ == "__main__":
    if "--stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http")