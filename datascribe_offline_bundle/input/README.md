# Input data

Place **CSV** (or **Excel**) files here for offline runs.

- Default sample: `sample_titanic.csv` — use `--target Survived` for classification-style EDA.
- To use your own file:

```bash
python datascribe_offline_bundle/run_offline.py --input path/to/your.csv --target YourTargetColumn
```

Paths can be absolute or relative; if a relative path is not found from the current working directory, the script also resolves paths relative to the bundle folder when helpful.
