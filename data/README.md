# Local data layout

Rekindle expects the Amazon Reviews 2023 Electronics review and metadata files to be placed
locally. They are never committed to this repository.

```text
data/
  raw/
    Electronics.jsonl.gz
    meta_Electronics.jsonl.gz
  interim/        # DuckDB working tables; ignored
  processed/      # Parquet outputs; ignored
```

The preparation command will record source identifiers, input checksums, time-window
boundaries, and row counts in a generated data manifest. The manifest supports
reproducibility without redistributing the dataset.
