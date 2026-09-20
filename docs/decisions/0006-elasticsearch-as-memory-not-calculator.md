# 0006 — Elasticsearch as evidence memory, not as the statistics engine

## Problem

Elasticsearch is a sponsor technology and a genuinely good fit for part of this system. The
temptation is to push more into it than belongs there — aggregations can compute means and
correlations, so why not run the statistics there too?

## Candidate designs

1. **Elasticsearch computes the statistics** via aggregations.
2. **Elasticsearch stores raw rows**; the engine queries subsets and computes locally.
3. **Elasticsearch stores only findings** (EvidenceObjects); raw data stays in Parquet.
4. **No Elasticsearch** — local vector store.

## Chosen

**(3).**

## Why

- **It is the job Elasticsearch is uniquely good at here.** The engine's hard retrieval
  problem is *"which of our previous findings bear on this new one?"* — a hybrid
  lexical + semantic + structured-filter query. That is exactly what Elasticsearch does
  better than anything we would write.
- **Correctness stops depending on the network.** A correlation must be reproducible and
  exact. Making it depend on a cluster's availability, version and aggregation semantics
  trades a guarantee for a dependency.
- **It is faster locally anyway.** 3.5 M rows of Parquet through Polars/NumPy beats a round
  trip, for the sizes this engine handles.
- **Findings are small and text-rich** — thousands of documents with natural-language
  summaries. That is Elasticsearch's sweet spot, whereas millions of numeric rows are not.

## Rejected

- **(1) Statistics in Elasticsearch** — would make the engine's core claim ("a deterministic
  layer decides what is true") contingent on a remote service, and aggregation semantics for
  things like Spearman, mutual information and cross-fold stability are not there anyway.
- **(2) Raw rows in Elasticsearch** — 3.7 M rows/month with no benefit; Parquet is already
  columnar, compressed and local.
- **(4) No Elasticsearch** — loses cross-run, cross-dataset memory, which is the product's
  distinguishing feature. (The local `MemoryEvidenceStore` implements the *same* retrieval
  semantics as a fallback, so the engine runs without it — but it is a fallback, not the
  design.)

## Retrieval design

Lexical and vector scores are on incomparable scales, so they are merged with **Reciprocal
Rank Fusion**, which uses only ranks:

```
score(d) = Σᵢ wᵢ / (k + rankᵢ(d))      k = 60
```

Three-tier strategy, because cluster capabilities vary:

1. native `rrf` retriever (server-side fusion) — **confirmed in use on Elasticsearch 9.5.4**
2. two queries fused client-side with the *identical* formula
3. BM25 only

The resolved mode is recorded in telemetry, so we can see which ran rather than assume.

`dense_vector` uses `int8_hnsw`: ~4× less memory than float32, with rescoring from
full-precision values to recover recall.

## The indexing-lag problem

Elasticsearch is near-real-time: a document written moments ago is not yet searchable. Within
one analysis the engine writes evidence and then immediately searches for related evidence —
so without intervention it would never see its own findings. **Measured: 0 retrievals.**

Refreshing on every write is expensive. The chosen fix is a **lazy refresh**: a `_dirty` flag
set on write, and one refresh before a search only when dirty. **Measured after: 48
retrievals per run.**

## Security posture

- **API-key auth only.** Never the built-in `elastic` superuser.
- **TLS verification always on.** An intercepted key is worse than a setup error.
- Errors pass through a scrubber that redacts anything key-shaped before it can reach a log
  or an HTTP response.
- `delete_index()` **refuses** any index whose name does not contain `test`.

## Implications

- **Runtime:** retrieval ~75 ms per hybrid query measured against the live cluster.
- **Memory:** int8 quantization keeps the vector footprint small.
- **Availability:** a failed write queues for retry and the analysis continues; the
  degradation is reported, never silent.

## Tests

`tests/integration/test_elasticsearch.py` — 19 tests against the **live cluster**: write/read,
bulk, all five filter types, hybrid semantic retrieval, the unresolved→explained lifecycle,
retrieval-mode assertion, the delete guard, and credential scrubbing. Auto-skips without
credentials.
