# Development corpus

`cfpb_consumer_finance.json` is a small offline fixture paraphrased from the linked public
Consumer Financial Protection Bureau consumer-information pages. It is designed for local
development and tests, not as a substitute for the current source pages or individualized
legal/financial advice. `source_version` records the fixture review date; it is not presented
as the CFPB page publication date.

Acquisition is intentionally separate from normalization. A later source adapter can refresh
the same `SourceRecord` contract without changing chunking or indexing.
