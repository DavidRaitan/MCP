import os
import sys
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from library_backend import LibraryEngine, ENDPOINT_URL, classify_record

port = int(os.environ.get("PORT", 8000))
mcp = FastMCP("ComparativeResearchMCP", host="0.0.0.0", port=port)


@mcp.tool()
async def compare_traditions_or_authors(
    targets: list[str],
    keywords: list[str],
    books_per_target: int = 2,
) -> dict:
    """
    Perform Cross-Book / Cross-Tradition RAG by scanning multiple primary texts or authors
    in parallel without downloading files locally.

    HOW TO USE FOR BROAD COMPARISONS:
    When a user asks to compare traditions, eras, or thinkers (e.g., "Compare ancient Greece
    to modern science" or "Compare the Bible and Aristotle on money"), decompose the request into:
    - targets: 3 to 6 representative primary authors, canonical books, or topics
      (e.g., ["Plato", "Aristotle", "Galileo Galilei", "Isaac Newton"]).
    - keywords: 2 to 4 concrete concepts or terms that appear in primary texts
      (e.g., ["experiment", "nature", "cause", "mathematics"]).
    """
    async with streamable_http_client(ENDPOINT_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            engine = LibraryEngine(session)
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


@mcp.tool()
async def search_library_books(
    query: str,
    author_only: bool = False,
    limit: int = 15,
) -> list[dict]:
    """
    Search for books sorted heaviest-to-lightest, tagged with confidence labels
    ('[BOOK]' for full books vs '[PAPER]' for short articles/reviews), along with MD5 identifiers.
    """
    async with streamable_http_client(ENDPOINT_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            engine = LibraryEngine(session)
            books = await engine.search_books(
                query=query,
                search_in=["author"] if author_only else None,
                limit=limit,
                sort_by_size=True,
            )
            for b in books:
                b["confidence_tag"] = classify_record(b)
            return books


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
    async with streamable_http_client(ENDPOINT_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            engine = LibraryEngine(session)
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


@mcp.tool()
async def download_book(md5: str, title: str = "book", authors: str = "Unknown", extension: str = "pdf") -> dict:
    """
    Download a book to ~/Downloads (when running locally) and return resolved mirror URLs.
    Pass the md5, title, authors, and extension from search_library_books.
    """
    async with streamable_http_client(ENDPOINT_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            engine = LibraryEngine(session)
            book_meta = {"md5": md5, "title": title, "authors": authors, "extension": extension}
            saved_path, urls = await engine.download_book_to_disk(book_meta)
            return {
                "md5": md5,
                "saved_to": str(saved_path.resolve()) if saved_path else None,
                "fallback_urls": urls,
            }


if __name__ == "__main__":
    if "--stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http")