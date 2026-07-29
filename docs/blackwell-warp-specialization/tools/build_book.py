#!/usr/bin/env python3
"""Build the Blackwell warp-specialization article as one offline HTML file."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import mimetypes
import re
import sys
from pathlib import Path
from html.parser import HTMLParser

import markdown


BOOK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = BOOK_DIR / "article.md"
DEFAULT_OUTPUT = BOOK_DIR / "index.html"
CSS_SOURCE = BOOK_DIR / "assets" / "book.css"
JS_SOURCE = BOOK_DIR / "assets" / "book.js"


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _embed_local_images(rendered: str, source_dir: Path) -> str:
    pattern = re.compile(r'(<img\b[^>]*?\bsrc=")([^":?#]+)("[^>]*>)')

    def replace(match: re.Match[str]) -> str:
        relative = html.unescape(match.group(2))
        candidate = (source_dir / relative).resolve()
        try:
            candidate.relative_to(BOOK_DIR.resolve())
        except ValueError as exc:
            raise ValueError(f"image escapes book directory: {relative}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"missing local image: {candidate}")
        return f"{match.group(1)}{_data_uri(candidate)}{match.group(3)}"

    return pattern.sub(replace, rendered)


def _extract_title(rendered: str) -> tuple[str, str]:
    match = re.search(r"<h1[^>]*>(.*?)</h1>", rendered, flags=re.DOTALL)
    if not match:
        raise ValueError("article must contain one level-one heading")
    heading = re.sub(
        r'<a\b[^>]*class="heading-anchor"[^>]*>.*?</a>',
        "",
        match.group(1),
        flags=re.DOTALL | re.IGNORECASE,
    )
    title = re.sub(r"<[^>]+>", "", heading)
    body = rendered[: match.start()] + rendered[match.end() :]
    return html.unescape(title), body


def _wrap_tables(rendered: str) -> str:
    """Give wide Markdown tables their own horizontal scroll container."""
    return re.sub(
        r"(<table\b[^>]*>.*?</table>)",
        r'<div class="table-scroll" role="region" aria-label="可横向滚动的表格" tabindex="0">\1</div>',
        rendered,
        flags=re.DOTALL | re.IGNORECASE,
    )


def build(source: Path) -> str:
    source_text = source.read_text(encoding="utf-8")
    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    renderer = markdown.Markdown(
        extensions=[
            "markdown.extensions.extra",
            "markdown.extensions.sane_lists",
            "markdown.extensions.toc",
            "markdown.extensions.codehilite",
        ],
        extension_configs={
            "markdown.extensions.toc": {
                "permalink": "§",
                "permalink_class": "heading-anchor",
                "toc_depth": "2-3",
            },
            "markdown.extensions.codehilite": {
                "guess_lang": False,
                "linenums": False,
                "css_class": "codehilite",
            },
        },
        output_format="html5",
    )
    rendered = _wrap_tables(renderer.convert(source_text))
    title, body = _extract_title(rendered)
    body = _embed_local_images(body, source.parent)
    toc = renderer.toc
    cover_path = BOOK_DIR / "assets" / "cover.webp"
    cover = ""
    if cover_path.is_file():
        cover = (
            '<img class="hero-art" src="'
            + _data_uri(cover_path)
            + '" alt="抽象呈现异步数据通路与专用 warp 角色的封面图">'
        )

    css = CSS_SOURCE.read_text(encoding="utf-8")
    script = JS_SOURCE.read_text(encoding="utf-8")
    build_inputs_digest = hashlib.sha256(
        "\0".join((source_text, body, cover, css, script)).encode("utf-8")
    ).hexdigest()
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="zh-CN" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark light">
  <meta name="generator" content="docs/blackwell-warp-specialization/tools/build_book.py">
  <meta name="source-sha256" content="{digest}">
  <meta name="build-inputs-sha256" content="{build_inputs_digest}">
  <title>{escaped_title}</title>
  <style>{css}</style>
</head>
<body>
  <div class="reading-progress" aria-hidden="true"><span></span></div>
  <button class="nav-toggle" type="button" aria-label="打开目录" aria-expanded="false">目录</button>
  <aside class="sidebar" aria-label="章节导航">
    <div class="sidebar-head">
      <div class="eyebrow">TRITON · BLACKWELL · SM103</div>
      <a class="book-mark" href="#top">Warp Specialization</a>
    </div>
    <label class="search-label" for="toc-search">筛选章节</label>
    <input id="toc-search" class="toc-search" type="search" placeholder="输入章节或符号…" autocomplete="off">
    <nav id="toc" class="toc">{toc}</nav>
  </aside>
  <main id="top">
    <header class="hero">
      <div class="hero-copy">
        <div class="eyebrow">SOURCE-GROUNDED · COMPILE-ONLY</div>
        <h1>{escaped_title}</h1>
        <p>从前端请求、分区数据流和 ARef 所有权，一直读到 LLVM worker 状态机与 SM103 指令结构。</p>
        <div class="hero-tags"><span>TMA</span><span>TCGen05</span><span>TMEM</span><span>MLIR</span></div>
      </div>
{cover}
    </header>
    <article>{body}</article>
    <footer>由冻结源码与编译态证据生成 · source SHA-256 {digest[:16]}</footer>
  </main>
  <button class="theme-toggle" type="button" aria-label="切换明暗主题">◐</button>
  <script>{script}</script>
</body>
</html>
"""


class _OfflineResourceAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.casefold(): value or "" for name, value in attrs}
        if tag.casefold() in {"iframe", "object", "embed", "audio", "video"}:
            self.errors.append(f"unsupported resource-bearing tag <{tag}>")
        if "src" in values and not values["src"].startswith("data:"):
            self.errors.append(f"non-embedded src on <{tag}>: {values['src']!r}")
        if "srcset" in values:
            self.errors.append(f"srcset is not allowed on <{tag}>")
        if tag.casefold() == "link" and not values.get("href", "").startswith("data:"):
            self.errors.append("non-embedded link resource is not allowed")


def validate(document: str) -> None:
    audit = _OfflineResourceAudit()
    audit.feed(document)
    css_urls = [
        match.group(1).strip().strip("'\"")
        for match in re.finditer(r"url\s*\(\s*([^)]*?)\s*\)", document, re.IGNORECASE)
    ]
    checks = {
        "document language": '<html lang="zh-CN"' in document,
        "embedded style": "<style>" in document and "</style>" in document,
        "embedded script": "<script>" in document and "</script>" in document,
        "table of contents": 'id="toc"' in document,
        "offline resource audit": not audit.errors,
        "no CSS imports": re.search(r"@import\b", document, re.IGNORECASE) is None,
        "no external CSS urls": all(value.startswith("data:") for value in css_urls),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        details = "; ".join(audit.errors)
        raise ValueError("HTML validation failed: " + ", ".join(failed) + (f" ({details})" if details else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate that the existing output is byte-for-byte current",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    document = build(source)
    validate(document)
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != document:
            print(f"stale or missing generated book: {output}", file=sys.stderr)
            return 1
        print(f"validated {output}")
        return 0
    output.write_text(document, encoding="utf-8")
    print(f"wrote {output} ({len(document):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
