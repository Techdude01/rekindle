# Rekindle design contract

## Objective

Predict the top-K products a user is likely to interact with next. A review's presence is an
implicit positive event; star ratings are excluded from targets and features.

## Data safety

- Use a recent 36-month review window.
- Deduplicate user–product events before modeling.
- Derive the iterative 5-core warm set from training interactions only.
- Build every aggregate, feature, and user-history context as of the prediction timestamp.
- Keep items first observed after the training boundary out of the primary candidate catalog.

**Threshold distinction:** the training-only 5-core defines the warm collaborative dataset.
The separate two-prior-interaction rule determines whether an individual warm target event has
enough earlier warm-catalog history to become a training or replay example. It does not replace
or weaken the 5-core criterion.

## Model and evaluation

The user tower combines a learned user representation with a recency-weighted summary of the
last 20 interacted products. Warm retrieval items use ID embeddings only. Metadata has two
separate downstream roles: time-safe similarity/affinity features for the ranker, and a
metadata-only TF-IDF representation for the synthetic cold-item fallback. A
sampled-softmax/contrastive objective uses in-batch and a uniform/popularity negative mixture.
Each retrieval source supplies up to 200 candidates; the deduplicated neural, item-CF, and
popularity union has at most 600 candidates for LightGBM LambdaRank to reduce to 10 products.

Ranker training uses rolling temporal cross-fitting: a retriever trained on an earlier fold
generates candidate lists for the next fold. This prevents a ranking feature from receiving a
retrieval score inflated by having trained on its own target.

Sequential replay freezes model weights before test. It reports retrieval Recall@50/@100/@200,
final NDCG@10, and slice metrics by user-history depth, item popularity, and warm/synthetic
cold status. Exact FAISS is the quality baseline; HNSW is tuned on the measured
recall–latency curve.

The main replay scores every eligible warm test event in chronological order. Products already
seen by that user are excluded from candidate and final lists. Ranker training keeps only
cross-fitted candidate lists where retrieval found the target; missed targets are never injected
and score zero in end-to-end replay.

## Baselines and model selection

Time-decayed popularity is tuned over 30-, 90-, and 180-day half-lives on validation only.
Item–item collaborative filtering uses training-only cosine similarity and a recency-weighted
history of up to 20 products. Retriever training uses at most 20 epochs with patience-five
early stopping on exact validation Recall@100 for a deterministic 1,000-event subset. This
keeps model selection aligned with the real 55,927-product catalog while remaining practical on
local hardware. The 999-negative sampled Recall@100 is calculated once for the winning checkpoint
as a diagnostic, not a model-selection criterion. Ranking cross-fitting uses three chronological
folds and computes the 5-core separately within each earlier fold.

The first neural ablation removes learned user-ID embeddings while retaining the same
recency-weighted item history, objective, and evaluation slice. This tests whether sparse
per-user histories make a durable user representation less reliable than shared sequence signal.

## Operational assumptions

The reference environment is an 18 GB Apple Silicon Mac. DuckDB performs disk-backed
aggregation over Parquet; Polars handles compact derived tables. Full retriever/ranker/index
rebuilds are weekly in the documented operating model, popularity aggregates refresh daily,
and user context updates per interaction. CUDA execution is optional, never required.
