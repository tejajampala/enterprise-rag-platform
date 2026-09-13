from __future__ import annotations

from pathlib import Path

from consumer_finance_rag.corpus_expansion import (
    CorpusManifest,
    acquire_manifest_corpus,
    parse_cfpb_page,
    parse_sitemap,
)

SITEMAP_URL = "https://www.consumerfinance.gov/sitemap.xml"
PAGE_URL = "https://www.consumerfinance.gov/ask-cfpb/credit-report-error-en-1/"
SITEMAP = f"""<?xml version='1.0'?>
<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
  <url><loc>{PAGE_URL}</loc></url>
</urlset>""".encode()
PAGE = b"""<html><head><title>Credit report help | Consumer Financial Protection Bureau</title>
</head><body><nav>Skip me</nav><main><h1>Dispute errors</h1><p>Send supporting records.</p>
</main><script>ignore()</script></body></html>"""


def _manifest() -> CorpusManifest:
    return CorpusManifest.model_validate(
        {
            "name": "test",
            "version": "1",
            "sources": [
                {
                    "source_id": "cfpb",
                    "sitemap_url": SITEMAP_URL,
                    "allowed_url_prefix": "https://www.consumerfinance.gov/ask-cfpb/",
                    "source_version": "snapshot-1",
                    "max_documents": 10,
                    "category_rules": [
                        {"category": "credit reporting", "keywords": ["credit-report"]}
                    ],
                }
            ],
        }
    )


def test_manifest_acquisition_parses_official_page_and_uses_cache(tmp_path: Path) -> None:
    responses = {SITEMAP_URL: SITEMAP, PAGE_URL: PAGE}
    calls: list[str] = []

    def fetch(url: str, timeout: float) -> bytes:
        assert timeout == 2.0
        calls.append(url)
        return responses[url]

    first = acquire_manifest_corpus(
        _manifest(), cache_directory=tmp_path, timeout=2.0, fetcher=fetch
    )
    second = acquire_manifest_corpus(
        _manifest(),
        cache_directory=tmp_path,
        timeout=2.0,
        fetcher=lambda url, timeout: (_ for _ in ()).throw(AssertionError("cache miss")),
    )

    assert first == second
    assert calls == [SITEMAP_URL, PAGE_URL]
    assert first[0].category == "credit reporting"
    assert first[0].title == "Credit report help"
    assert "Send supporting records" in first[0].text


def test_corpus_parsers_are_independently_testable() -> None:
    assert parse_sitemap(SITEMAP) == [PAGE_URL]
    assert parse_cfpb_page(PAGE)[0] == "Credit report help"
