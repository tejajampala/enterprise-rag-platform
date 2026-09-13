"""Controlled CFPB corpus acquisition with manifests, limits, and an offline cache."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from consumer_finance_rag.acquisition import SourceRecord

CorpusCategory = Literal[
    "credit cards",
    "credit reporting",
    "mortgages",
    "bank accounts",
    "debt collection",
    "payments/fraud",
]


class CategoryRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: CorpusCategory
    keywords: list[str] = Field(min_length=1)


class CorpusSourceManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    sitemap_url: str
    allowed_url_prefix: str
    source_version: str
    max_documents: int = Field(gt=0, le=5000)
    category_rules: list[CategoryRule] = Field(min_length=1)


class CorpusManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    sources: list[CorpusSourceManifest] = Field(min_length=1)


class ByteFetcher(Protocol):
    def __call__(self, url: str, timeout: float) -> bytes: ...


def load_corpus_manifest(path: str | Path) -> CorpusManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CorpusManifest.model_validate(data)


def http_fetch(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "enterprise-rag-platform/0.2 corpus-evaluation-fetcher"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return bytes(response.read())


def parse_sitemap(content: bytes) -> list[str]:
    root = ET.fromstring(content)
    return [
        element.text.strip()
        for element in root.iter()
        if element.tag.endswith("loc") and element.text and element.text.strip()
    ]


class _ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._title_depth = 0
        self._main_depth = 0
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "title":
            self._title_depth += 1
        if tag == "main":
            self._main_depth += 1
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._title_depth:
            self._title_depth -= 1
        if tag == "main" and self._main_depth:
            self._main_depth -= 1
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        cleaned = re.sub(r"\s+", " ", data).strip()
        if not cleaned or self._ignored_depth:
            return
        if self._title_depth:
            self.title_parts.append(cleaned)
        if self._main_depth:
            self.text_parts.append(cleaned)


def parse_cfpb_page(content: bytes) -> tuple[str, str]:
    parser = _ArticleParser()
    parser.feed(content.decode("utf-8", errors="replace"))
    title = " ".join(parser.title_parts).removesuffix(" | Consumer Financial Protection Bureau")
    text = "\n\n".join(parser.text_parts)
    if not title or not text:
        raise ValueError("CFPB page did not contain both a title and main content")
    return title, text


def _category_for(url: str, rules: Iterable[CategoryRule]) -> CorpusCategory | None:
    lowered = url.lower()
    for rule in rules:
        if any(keyword.lower() in lowered for keyword in rule.keywords):
            return rule.category
    return None


def _cache_path(cache_directory: Path, url: str, suffix: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_directory / f"{digest}{suffix}"


def _fetch_cached(
    url: str,
    *,
    cache_directory: Path,
    suffix: str,
    timeout: float,
    fetcher: ByteFetcher,
    refresh: bool,
) -> bytes:
    path = _cache_path(cache_directory, url, suffix)
    if path.is_file() and not refresh:
        return path.read_bytes()
    content = fetcher(url, timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def acquire_manifest_corpus(
    manifest: CorpusManifest,
    *,
    cache_directory: str | Path,
    timeout: float,
    fetcher: ByteFetcher = http_fetch,
    refresh: bool = False,
) -> list[SourceRecord]:
    """Fetch bounded official pages; unit tests inject a network-free fetcher."""

    cache = Path(cache_directory)
    records: list[SourceRecord] = []
    for source in manifest.sources:
        sitemap = _fetch_cached(
            source.sitemap_url,
            cache_directory=cache,
            suffix=".xml",
            timeout=timeout,
            fetcher=fetcher,
            refresh=refresh,
        )
        urls = parse_sitemap(sitemap)
        nested = [url for url in urls if urlparse(url).path.endswith(".xml")]
        if nested:
            urls = []
            for nested_url in nested[:20]:
                nested_content = _fetch_cached(
                    nested_url,
                    cache_directory=cache,
                    suffix=".xml",
                    timeout=timeout,
                    fetcher=fetcher,
                    refresh=refresh,
                )
                urls.extend(parse_sitemap(nested_content))

        selected: list[tuple[str, CorpusCategory]] = []
        for url in sorted(set(urls)):
            if not url.startswith(source.allowed_url_prefix):
                continue
            category = _category_for(url, source.category_rules)
            if category is not None:
                selected.append((url, category))
            if len(selected) >= source.max_documents:
                break

        for url, category in selected:
            page = _fetch_cached(
                url,
                cache_directory=cache,
                suffix=".html",
                timeout=timeout,
                fetcher=fetcher,
                refresh=refresh,
            )
            try:
                title, text = parse_cfpb_page(page)
            except ValueError:
                continue
            slug = urlparse(url).path.strip("/").replace("/", "-")
            records.append(
                SourceRecord(
                    source_id=f"{source.source_id}-{slug}",
                    title=title,
                    source_url=url,
                    category=category,
                    source_version=source.source_version,
                    text=text,
                )
            )
    return records


def write_source_records(records: list[SourceRecord], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.model_dump(mode="json") for record in records]
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
