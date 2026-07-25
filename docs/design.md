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

## Model and evaluation

The user tower combines a learned user representation with a recency-weighted summary of the
last 20 interacted products. A sampled-softmax/contrastive objective uses in-batch and a
uniform/popularity negative mixture. FAISS retrieves 200 candidates; LightGBM LambdaRank
returns 10 products.

Ranker training uses rolling temporal cross-fitting: a retriever trained on an earlier fold
generates candidate lists for the next fold. This prevents a ranking feature from receiving a
retrieval score inflated by having trained on its own target.

Sequential replay freezes model weights before test. It reports retrieval Recall@50/@100/@200,
final NDCG@10, and slice metrics by user-history depth, item popularity, and warm/synthetic
cold status. Exact FAISS is the quality baseline; HNSW is tuned on the measured
recall–latency curve.

## Operational assumptions

The reference environment is an 18 GB Apple Silicon Mac. DuckDB performs disk-backed
aggregation over Parquet; Polars handles compact derived tables. Full retriever/ranker/index
rebuilds are weekly in the documented operating model, popularity aggregates refresh daily,
and user context updates per interaction. CUDA execution is optional, never required.
