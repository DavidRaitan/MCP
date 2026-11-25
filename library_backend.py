import asyncio
import datetime
import os
import re
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ENDPOINT_URL = "https://mcp.jmrp.io/libgen"
DEFAULT_DOWNLOAD_DIR = Path.home() / "Downloads"
DOWNLOAD_DIR = Path(os.getenv("LIBGEN_DOWNLOAD_DIR", str(DEFAULT_DOWNLOAD_DIR))).expanduser()
REPORTS_DIR = Path("./reports")


def _parse_size_mb(size_str: str) -> float:
    """Normalize size strings ('20 MB', '970 kB', '1.2 GB') into float MB."""
    if not size_str:
        return 0.0
    m = re.match(r"([\d.]+)\s*(b|kb|mb|gb)", str(size_str).strip(), re.I)
    if not m:
        return 0.0
    val, unit = float(m.group(1)), m.group(2).lower()
    if unit == "b":
        return val / (1024.0 * 1024.0)
    if unit == "kb":
        return val / 1024.0
    if unit == "gb":
        return val * 1024.0
    return val


def classify_record(book: dict) -> str:
    """Distinguish full books ('[BOOK]') from journal articles/reviews ('[PAPER]')."""
    ext = str(book.get("extension", "")).lower()
    mb = _parse_size_mb(book.get("size", ""))
    title = str(book.get("title", "")).lower()

    if ext in {"epub", "mobi", "azw3", "djvu", "fb2"}:
        return "[BOOK]"
    if re.search(r"\bvol\.\s*\d+|\biss\.\s*\d+|\(\d{4}-[a-z]{3}\b", title):
        return "[PAPER]"
    if mb >= 0.75:
        return "[BOOK]"
    return "[PAPER]"


def rank_readable_books(books: list[dict]) -> list[dict]:
    """Rank books for text extraction: full EPUBs first, then native-text PDFs."""
    def score(b: dict) -> tuple:
        tag = classify_record(b)
        ext = str(b.get("extension", "")).lower()
        mb = _parse_size_mb(b.get("size", ""))
        is_book = 0 if tag == "[BOOK]" else 1
        if ext == "epub":
            tier = 0
        elif ext == "pdf" and 0.4 <= mb <= 30.0:
            tier = 1
        elif ext == "pdf":
            tier = 2
        else:
            tier = 3
        return (is_book, tier, -mb)

    return sorted(books, key=score)


def _fit(text: str, width: int) -> str:
    """Pad or cleanly ellipsize a single-line cell to exact column width."""
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= width:
        return clean.ljust(width)
    return clean[: max(1, width - 1)] + "…"


class LibraryEngine:
    """
    Decoupled Cross-Book RAG & Retrieval Backend.
    Paced for the public mcp.jmrp.io rate limit (~2 req/s) and resilient to mirror errors.
    """

    def __init__(
        self,
        session: ClientSession,
        download_dir: Path = DOWNLOAD_DIR,
        reports_dir: Path = REPORTS_DIR,
    ):
        self.session = session
        self.download_dir = download_dir
        self.reports_dir = reports_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        # Keep concurrency <= 2 to stay within mcp.jmrp.io's ~2 req/sec limiter
        self._sem = asyncio.Semaphore(2)

    async def search_books(
        self,
        query: str,
        search_in: Optional[list[str]] = None,
        limit: int = 40,
        sort_by_size: bool = True,
    ) -> list[dict]:
        args = {
            "query": query,
            "topics": ["nonfiction", "fiction"],
            "results_per_page": 50 if limit <= 50 else 100,
        }
        if search_in:
            args["search_in"] = search_in

        async with self._sem:
            await asyncio.sleep(0.35)
            res = await self.session.call_tool("search", arguments=args)

        books = []
        if res.structured_content and "results" in res.structured_content:
            books = res.structured_content["results"]

        if search_in == ["author"] and len(query.strip()) >= 2:
            q_tokens = [t.lower() for t in query.strip().split() if len(t) >= 2]
            if q_tokens:
                exact = [
                    b for b in books
                    if all(tok in str(b.get("authors", "")).lower() for tok in q_tokens)
                ]
                rest = [b for b in books if b not in exact]
                books = exact + rest

        if sort_by_size:
            books = sorted(
                books,
                key=lambda b: (
                    0 if classify_record(b) == "[BOOK]" else 1,
                    -_parse_size_mb(b.get("size", "")),
                ),
            )

        return books[:limit]

    async def search_inside_book(
        self, md5: str, keyword: str, max_matches: int = 5
    ) -> str:
        async with self._sem:
            await asyncio.sleep(0.55)  # Gentle pacing for ~2 req/s rate limit
            try:
                res = await self.session.call_tool(
                    "read",
                    arguments={"md5": md5, "find": keyword, "max_matches": max_matches},
                )
                return res.content[0].text if res.content else ""
            except Exception as e:
                return f"[Error scanning {md5}: {e}]"

    async def get_toc(self, md5: str) -> str:
        async with self._sem:
            res = await self.session.call_tool(
                "read", arguments={"md5": md5, "outline": True, "max_depth": 2}
            )
            return res.content[0].text if res.content else ""

    async def read_pages(
        self, md5: str, start_page: int = 1, max_pages: int = 4
    ) -> str:
        async with self._sem:
            res = await self.session.call_tool(
                "read",
                arguments={"md5": md5, "start_page": start_page, "max_pages": max_pages},
            )
            return res.content[0].text if res.content else ""

    async def get_citation(self, identifier: str) -> str:
        ident = identifier.strip()
        args = {"enrich": True, "cite_as": ["apa"]}
        if ident.startswith("10.") or "/" in ident:
            args["doi"] = ident
        else:
            args["md5"] = ident
        async with self._sem:
            res = await self.session.call_tool("get_details", arguments=args)
            return res.content[0].text if res.content else ""

    async def cross_book_rag(
        self,
        targets: list[str],
        keywords: list[str],
        books_per_target: int = 2,
        max_matches_per_book: int = 4,
        save_markdown: bool = True,
    ) -> tuple[list[dict], Optional[Path]]:
        candidate_books = []
        seen_md5 = set()

        for target in targets:
            target = target.strip()
            if not target:
                continue
            raw_books = await self.search_books(target, limit=25, sort_by_size=False)
            ranked = rank_readable_books(raw_books)

            picked = 0
            for b in ranked:
                md5 = b.get("md5")
                if md5 and md5 not in seen_md5:
                    seen_md5.add(md5)
                    b["_matched_target"] = target
                    candidate_books.append(b)
                    picked += 1
                    if picked >= books_per_target:
                        break

        async def scan_one(book: dict, kw: str) -> dict:
            text = await self.search_inside_book(
                book["md5"], kw, max_matches=max_matches_per_book
            )
            has_hit = (
                "extractable=false" not in text.lower()
                and "0 matches" not in text.lower()
                and "no matches" not in text.lower()
                and len(text.strip()) > 0
            )
            return {
                "target": book["_matched_target"],
                "title": book.get("title"),
                "authors": book.get("authors"),
                "year": book.get("year"),
                "ext": book.get("extension"),
                "size": book.get("size"),
                "md5": book["md5"],
                "keyword": kw,
                "has_hit": has_hit,
                "excerpts": text,
            }

        tasks = [scan_one(book, kw) for book in candidate_books for kw in keywords]
        findings = await asyncio.gather(*tasks)

        report_path = None
        if save_markdown and findings:
            report_path = self._write_markdown_report(targets, keywords, findings)

        return findings, report_path

    def _write_markdown_report(
        self, targets: list[str], keywords: list[str], findings: list[dict]
    ) -> Path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", "_vs_".join(targets[:3]))[:45].strip("_")
        report_file = self.reports_dir / f"{ts}_{slug}.md"

        lines = [
            f"# Cross-Book Research Report: {', '.join(targets)}",
            f"- **Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"- **Concepts / Keywords:** {', '.join(keywords)}",
            "",
            "---",
            "",
        ]

        for f in findings:
            badge = "MATCHES FOUND" if f["has_hit"] else "NO MATCHES / SCANNED PDF"
            lines.extend([
                f"## [{badge}] {f['target']} — `{f['keyword']}`",
                f"- **Book:** {f['title']} ({f['authors']}, {f['year']})",
                f"- **Format / Size:** {str(f['ext']).upper()} ({f['size']}) | **MD5:** `{f['md5']}`",
                "",
                f["excerpts"].strip(),
                "",
                "---",
                "",
            ])

        report_file.write_text("\n".join(lines), encoding="utf-8")
        return report_file

    async def resolve_download_urls(
        self, md5: str, source: Optional[str] = None
    ) -> list[str]:
        """Extract all clean mirror URLs returned by the MCP download tool."""
        args = {"md5": md5}
        if source:
            args["source"] = source

        async with self._sem:
            await asyncio.sleep(0.4)
            res = await self.session.call_tool("download", arguments=args)

        urls = []
        # 1. Pull from structured JSON 'links' array if present
        if res.structured_content and isinstance(res.structured_content, dict):
            for u in res.structured_content.get("links", []):
                clean_u = str(u).strip("\"'<> ")
                if clean_u.startswith("http") and clean_u not in urls:
                    urls.append(clean_u)

        # 2. Pull from ResourceLink or text blocks without trailing quotes
        for block in res.content:
            if hasattr(block, "uri") and block.uri:
                clean_u = str(block.uri).strip("\"'<> ")
                if clean_u.startswith("http") and clean_u not in urls:
                    urls.append(clean_u)
            if hasattr(block, "text"):
                for m in re.findall(r"https?://[^\s\)\]\">']+", block.text):
                    if m not in urls:
                        urls.append(m)

        return urls

    async def _stream_binary_to_file(
        self, client: httpx.AsyncClient, url: str, dest: Path
    ) -> bool:
        """
        Downloads a binary file directly or follows one level of HTML 'GET' button.
        Skips MySQL 'max_user_connections' overload pages automatically.
        """
        async with client.stream("GET", url) as resp:
            if resp.status_code != 200:
                return False

            ctype = resp.headers.get("content-type", "").lower()

            # Case A: Server immediately streams binary file
            if "text/html" not in ctype:
                written = 0
                with open(dest, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
                        written += len(chunk)
                if written >= 1024:
                    return True
                if dest.exists():
                    dest.unlink()
                return False

            # Case B: HTML landing page — check for MySQL overload or follow GET link
            html = (await resp.aread()).decode("utf-8", errors="ignore")
            if "max_user_connections" in html.lower():
                return False

            get_match = re.search(
                r'<a[^>]+href=["\']([^"\']*(?:get\.php|download)[^"\']*)["\']',
                html,
                re.I,
            )
            if not get_match:
                return False

            direct_url = urljoin(str(resp.url), get_match.group(1))
            async with client.stream("GET", direct_url) as file_resp:
                if file_resp.status_code != 200:
                    return False
                if "text/html" in file_resp.headers.get("content-type", "").lower():
                    return False

                written = 0
                with open(dest, "wb") as f:
                    async for chunk in file_resp.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
                        written += len(chunk)
                if written >= 1024:
                    return True
                if dest.exists():
                    dest.unlink()
                return False

    async def download_book_to_disk(self, book: dict) -> tuple[Optional[Path], list[str]]:
        md5 = book["md5"]
        safe_title = re.sub(r'[\\/*?:"<>|]', "", str(book.get("title", md5)))[:100].strip()
        safe_author = re.sub(r'[\\/*?:"<>|]', "", str(book.get("authors", "Unknown")))[:60].strip()
        ext = str(book.get("extension", "pdf")).lower().lstrip(".")
        dest = self.download_dir / f"{safe_author} - {safe_title}.{ext}"

        candidate_urls = []
        for src in [None, "randombook", "libgen", "annas"]:
            try:
                found = await self.resolve_download_urls(md5, source=src)
                for u in found:
                    if u not in candidate_urls:
                        candidate_urls.append(u)
            except Exception:
                continue

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            )
        }

        async with httpx.AsyncClient(
            follow_redirects=True, timeout=60.0, headers=headers
        ) as client:
            for url in candidate_urls:
                try:
                    ok = await self._stream_binary_to_file(client, url, dest)
                    if ok:
                        return dest, candidate_urls
                except Exception:
                    if dest.exists():
                        dest.unlink()
                    continue

        return None, candidate_urls


# --- Aligned CLI Table Formatter ---

def print_books_table(books: list[dict]):
    """Render a strictly aligned, terminal-width-aware table sorted by size."""
    if not books:
        print("\n  No matching books found.\n")
        return

    term_width = shutil.get_terminal_size((120, 24)).columns
    flex_width = max(40, term_width - 48)
    title_w = int(flex_width * 0.65)
    author_w = flex_width - title_w

    header = (
        f"  {'#':>2} | {'Type':<7} | {'Fmt':<5} | {'Size':>8} | {'Year':<4} | "
        f"{'Title':<{title_w}} | {'Author':<{author_w}}"
    )
    divider = "  " + "-" * (len(header) - 2)

    print("\n" + header)
    print(divider)
    for i, b in enumerate(books, 1):
        tag = classify_record(b)
        ext = str(b.get("extension", "?")).upper()
        size = str(b.get("size", "?"))
        year = str(b.get("year") or "----")
        title = _fit(b.get("title", "Untitled"), title_w)
        author = _fit(b.get("authors", "Unknown"), author_w)
        print(
            f"  {i:>2} | {tag:<7} | {ext:<5} | {size:>8} | {year:<4} | "
            f"{title} | {author}"
        )
    print(divider + "\n")


# --- Interactive CLI Flows ---

async def handle_cross_book_rag(engine: LibraryEngine):
    print("\n--- [C] CROSS-BOOK / CROSS-TRADITION RAG ---")
    print("Compare multiple authors, books, or traditions (e.g. 'Plato, Aristotle, Bible').")
    raw_targets = input("Targets (comma-separated): ").strip()
    if not raw_targets:
        return
    targets = [t.strip() for t in raw_targets.split(",") if t.strip()]

    raw_kws = input("Concepts/keywords to scan for (comma-separated): ").strip()
    if not raw_kws:
        return
    keywords = [k.strip() for k in raw_kws.split(",") if k.strip()]

    per_target = input("Top books per target (default 2): ").strip()
    books_per_target = int(per_target) if per_target.isdigit() else 2

    print(f"\nScanning up to {len(targets) * books_per_target} books across {len(keywords)} keyword(s)...\n")
    findings, report_path = await engine.cross_book_rag(
        targets=targets,
        keywords=keywords,
        books_per_target=books_per_target,
        save_markdown=True,
    )

    for f in findings:
        status = "MATCH" if f["has_hit"] else "NONE "
        print("=" * 88)
        print(f"[{status}] {f['target']} | Keyword: '{f['keyword']}' | {f['title']} ({f['year']}) [{str(f['ext']).upper()}]")
        print("-" * 88)
        print(f["excerpts"][:1200].strip())
        print()

    if report_path:
        print(f"Full untruncated Markdown report saved to: {report_path.resolve()}\n")


async def handle_single_book_search(engine: LibraryEngine):
    print("\n--- [S] SEARCH & INSPECT SINGLE BOOK ---")
    query = input("Search by title, author, or topic: ").strip()
    if not query:
        return

    books = await engine.search_books(query, limit=30, sort_by_size=True)
    print_books_table(books)
    if not books:
        return

    pick = input("Select # to open book tools (or Enter to go back): ").strip()
    if not pick.isdigit() or not (1 <= int(pick) <= len(books)):
        return

    book = books[int(pick) - 1]
    md5 = book["md5"]

    while True:
        print(f"\nActive Book: {book.get('title')} — {book.get('authors')} ({str(book.get('extension', '')).upper()}, {book.get('size')})")
        print("  [F] Find/search keyword inside this book")
        print("  [T] Table of Contents")
        print("  [R] Read specific pages")
        print("  [C] Get BibTeX & APA Citation")
        print("  [D] Download full book")
        print("  [B] Back to main menu")
        act = input("Action: ").strip().upper()

        if act == "F":
            kw = input("Keyword or phrase to find inside book: ").strip()
            if kw:
                print("\nScanning book...")
                print(await engine.search_inside_book(md5, kw, max_matches=8))
        elif act == "T":
            print("\n" + await engine.get_toc(md5))
        elif act == "R":
            p = input("Start page (default 1): ").strip()
            start_page = int(p) if p.isdigit() else 1
            print("\n" + await engine.read_pages(md5, start_page=start_page, max_pages=4))
        elif act == "C":
            print("\n" + await engine.get_citation(md5))
        elif act == "D":
            print(f"\nDownloading '{book.get('title')}'...")
            saved, urls = await engine.download_book_to_disk(book)
            if saved:
                print(f"Saved ({saved.stat().st_size / 1024:.1f} KB) -> {saved.resolve()}")
            else:
                print("Mirrors returned browser/overload pages. Direct fallback links:")
                for u in urls:
                    print(f"  -> {u}")
        elif act == "B":
            break


async def handle_download_by_author(engine: LibraryEngine):
    print("\n--- [D] DOWNLOAD BY AUTHOR ---")
    author = input("Enter author name: ").strip()
    if not author:
        return

    print(f"Searching books by '{author}'...")
    books = await engine.search_books(author, search_in=["author"], limit=40, sort_by_size=True)
    if not books:
        books = await engine.search_books(author, limit=40, sort_by_size=True)

    print_books_table(books)
    if not books:
        return

    while True:
        choice = input("Enter # to download, 'F' to filter format (PDF/EPUB), or Enter to cancel: ").strip()
        if not choice:
            return
        if choice.upper() == "F":
            avail = sorted({str(b.get("extension", "")).upper() for b in books if b.get("extension")})
            fmt = input(f"Filter by format ({', '.join(avail)}): ").strip().lower()
            filtered = [b for b in books if str(b.get("extension", "")).lower() == fmt]
            if filtered:
                books = filtered
                print_books_table(books)
            else:
                print(f"No '{fmt}' files found.\n")
            continue

        if choice.isdigit() and 1 <= int(choice) <= len(books):
            selected = books[int(choice) - 1]
            print(f"\nFetching '{selected.get('title')}' ({selected.get('size')})...")
            saved, urls = await engine.download_book_to_disk(selected)
            if saved:
                print(f"Saved ({saved.stat().st_size / 1024:.1f} KB) -> {saved.resolve()}\n")
            else:
                print("Mirrors returned browser/overload pages. Direct fallback links:")
                for u in urls:
                    print(f"  -> {u}")
                print()
            return


async def run_cli():
    print(f"Connecting to {ENDPOINT_URL}...")
    async with streamable_http_client(ENDPOINT_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            engine = LibraryEngine(session)
            print("Connected.\n")

            while True:
                print("=== RESEARCH & LIBRARY CLI ===")
                print(f"  Download Folder: {engine.download_dir}")
                print("  [C] Cross-Book / Cross-Tradition RAG (Multi-author scan + Markdown report)")
                print("  [S] Single-Book Search & Deep Dive (Search inside, TOC, Read, Cite, Download)")
                print("  [D] Download Books by Author (Sorted by size with [BOOK]/[PAPER] tags)")
                print("  [B] Quick Citation Lookup (by MD5 or DOI)")
                print("  [O] Configure Download Folder")
                print("  [Q] Quit")
                choice = input("Select option: ").strip().upper()

                if choice == "C":
                    await handle_cross_book_rag(engine)
                elif choice == "S":
                    await handle_single_book_search(engine)
                elif choice == "D":
                    await handle_download_by_author(engine)
                elif choice == "B":
                    ident = input("Enter book MD5 or article DOI: ").strip()
                    if ident:
                        print("\n" + await engine.get_citation(ident) + "\n")
                elif choice == "O":
                    new_dir = input(
                        f"Enter new download path (or 'default' for {DEFAULT_DOWNLOAD_DIR}): "
                    ).strip()
                    if new_dir.lower() == "default":
                        engine.download_dir = DEFAULT_DOWNLOAD_DIR
                    elif new_dir:
                        engine.download_dir = Path(new_dir).expanduser()
                    engine.download_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Download folder set to: {engine.download_dir.resolve()}\n")
                elif choice == "Q":
                    break


if __name__ == "__main__":
    asyncio.run(run_cli())