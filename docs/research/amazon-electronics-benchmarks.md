# Amazon Reviews 2023 Electronics: benchmark context

Research date: 2026-07-27.

## Bottom line

There is **no public result found that exactly matches Rekindle's task**: Amazon
Reviews 2023 Electronics; a 36-month window; 5-core derived from *training only*;
global chronological replay; a warm, full-catalog ranking set; prior-item exclusion;
and an end-to-end retrieval-plus-ranker Top-10 metric.

The closest peer-reviewed reference is AlphaRec (ICLR 2025). Its Amazon Electronics
2023 table reports a best **NDCG@20 of 0.0323** and **Recall@20 of 0.0687** for AlphaRec,
with the strongest non-AlphaRec baseline at NDCG **0.0288**. That is useful evidence
that a score below 0.1 is not inherently poor on this dataset family. It is **not** a
leaderboard comparison to Rekindle because the task, filtering, split, cutoff, and
architecture differ materially.

## What the official dataset provides

The McAuley Lab's official 5-core Electronics release contains about **1.6M users**,
**368.2K items**, and **15.5M reviews** before splitting. Its official absolute-time
split contains **13.1M / 1.2M / 1.2M** train/validation/test reviews. The release
deduplicates repeated user-item reviews and applies its 5-core before splitting.
It also offers a leave-last-out split. [Official 5-core processing and statistics](https://amazon-reviews-2023.github.io/data_processing/5core.html)

The official processing repository explicitly documents both leave-last-out and global
timestamp splitting. Its timestamp split uses one shared 80/10/10-style global cutoff
over the full Amazon Reviews 2023 corpus. [Official processing scripts and split rationale](https://github.com/hyp1231/AmazonReviews2023/blob/main/benchmark_scripts/README.md)

This is adjacent to, but not identical with, Rekindle: Rekindle limits the data to a
recent 36-month window and computes its 5-core from training data only. That avoids
using future test membership to decide who belongs in the warm collaborative corpus,
but produces a much smaller evaluation catalog (55,927 warm items in the current run).

## Published result directly on Amazon Electronics 2023

[AlphaRec: *Language Representations Can be What Recommenders Need* (ICLR 2025)](https://arxiv.org/html/2407.05441v4)
includes a dedicated Amazon Electronics 2023 experiment. The authors say they retain
only post-2022 interactions and run the same collaborative-filtering experiment used
elsewhere in the paper. Their table reports:

| Model | Recall@20 | NDCG@20 | HR@20 |
| --- | ---: | ---: | ---: |
| MF | 0.0130 | 0.0089 | 0.0136 |
| MultVAE | 0.0227 | 0.0158 | 0.0237 |
| LightGCN | 0.0237 | 0.0161 | 0.0248 |
| SGL | 0.0519 | 0.0250 | 0.0551 |
| BC Loss | 0.0548 | 0.0265 | 0.0585 |
| XSimGCL | 0.0534 | 0.0261 | 0.0569 |
| KAR | 0.0611 | 0.0283 | 0.0661 |
| RLMRec | 0.0633 | 0.0288 | 0.0674 |
| AlphaRec | **0.0687** | **0.0323** | **0.0732** |

### AlphaRec protocol caveats

- The paper's Electronics appendix says it uses interactions after 2022 and refers
  back to its main experimental protocol. [Electronics experiment and table](https://arxiv.org/html/2407.05441v4)
- That protocol splits each user's history at a 4:3:3 ratio, removes users with fewer
  than 20 interactions, and removes validation/test items absent from training.
  [Dataset protocol](https://arxiv.org/html/2407.05441v4)
- It uses all-ranking: all items except a user's training positives are ranked. The
  training selection metric is Recall@20. [Implementation details](https://arxiv.org/html/2407.05441v4)
- The public code's default evaluation cutoff is 20. [Repository evaluation default](https://raw.githubusercontent.com/LehengTHU/AlphaRec/master/parse.py)

Consequently, AlphaRec's 0.0323 is a credible **directional reference**, but not a
number Rekindle may claim to beat or trail. In particular, it evaluates a single-stage
collaborative-filtering model at a different cutoff and has a much stricter user-history
filter but a different temporal protocol.

## Useful temporal benchmark context (not Electronics)

[BLaIR / Amazon Reviews 2023](https://arxiv.org/html/2403.03952v2), from the dataset
authors, is useful for understanding the low-score regime under time-aware evaluation.
For sequential recommendation it uses an unfiltered global timestamp split and reports
NDCG@10 values such as **0.0113-0.0138** on Amazon Video Games and **0.0177-0.0241**
on All Beauty, depending on the text encoder. [Detailed sequential results](https://arxiv.org/html/2403.03952v2)

It is not an Electronics benchmark, and it does not evaluate Rekindle's two-stage
candidate union, so these values must not be used as a direct score comparison. They do
show why claims that Amazon Reviews 2023 needs NDCG above 0.1 to be useful are not
well supported.

## Comparison to Rekindle

| Dimension | AlphaRec Electronics 2023 | Rekindle |
| --- | --- | --- |
| Task | Single-stage collaborative filtering | Two-stage next-review replay |
| Time handling | Post-2022; per-user 4:3:3 split | 36-month global chronological 80/10/10 split |
| Warmth rule | Users with fewer than 20 interactions removed | Iterative 5-core from training only; replay additionally needs two prior events |
| Candidate universe | All-ranking, training positives excluded | 55,927-item warm catalog; seen products excluded |
| Output cutoff | Repository default is Top-20 | Top-10 |
| Ranking misses | Not a separate retrieval stage | Candidate-union misses receive score zero end-to-end |

Rekindle currently has **0.160 candidate-union Recall@200** on a 1,000-event validation
sample and **0.0711 conditional NDCG@10** for the ranker on fold-3 candidate-hit
queries. Neither is an end-to-end NDCG@10. The final replay metric must score every
eligible test event, assigning zero when the target never enters the union.

## How to judge the final result

The valid primary comparison is internal and protocol-matched:

1. Evaluate the frozen two-stage system, popularity, item-item CF, and retrieval-only
   on the **same** chronological test replay.
2. Report both candidate-union Recall@200 and end-to-end NDCG@10 / HitRate@10.
3. State the catalog size, eligible event count, seen-item filtering, and whether the
   metric is full-catalog or sampled.

Good README wording after the final run is therefore: “On a strict chronological,
training-only-5-core replay of Amazon Reviews 2023 Electronics, Rekindle improved
end-to-end NDCG@10 by X% over the strongest time-safe baseline.” Include the absolute
scores and do not call the result state of the art.

## Excluded claims

This research intentionally excludes résumé/LinkedIn claims and papers that merely say
“Amazon Electronics” without identifying the 2023 release, their split, candidate set,
or cutoff. A reported 0.3-0.6 NDCG can be legitimate under a sampled-negative or
different target protocol, but it is not evidence for this full-catalog chronological
replay task without those details.
