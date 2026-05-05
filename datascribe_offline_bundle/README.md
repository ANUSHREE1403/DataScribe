# DataScribe Offline Bundle (Simple Guide)

This folder is a **ready-to-use mini package** inside your DataScribe project.

It helps you:
- give an input file,
- run analysis,
- get all outputs in one place:
  - cleaned dataset,
  - charts,
  - HTML report,
  - PDF report,
  - ML results (if training is enabled).

You do **not** need to run the website for this.

## Folder contents

- `input/sample_titanic.csv`  
  Sample data you can test with.

- `input/README.md`  
  Short help for putting your own file.

- `run_offline.py`  
  Main script that runs everything.

- `output/`  
  All generated files come here.

## Before you run

Open terminal in the main project folder (`DataScribe`) and run:

```bash
pip install -r requirements.txt
```

## Quick run (with sample file)

```bash
cd path/to/DataScribe
python datascribe_offline_bundle/run_offline.py --target Survived
```

## Run directly from inside offline bundle (easy)

If user opens only the offline bundle folder, they can run it like a normal app entry file:

```bash
cd path/to/DataScribe/datascribe_offline_bundle
python run.py --target Survived --train
```

This `run.py` is the **offline bundle launcher**.

## Run with ML training also

```bash
python datascribe_offline_bundle/run_offline.py --target Survived --train
```

## What files are created in `output/`

- `cleaned_dataset.csv`  
  Cleaned version of your data.

- `report.html`  
  Full HTML report with:
  - dataset summary,
  - quality checks,
  - stats,
  - insights,
  - visualizations,
  - ML section (if `--train` used),
  - raw JSON sections for full details.

- `report.pdf`  
  Full PDF report with:
  - summary pages,
  - analysis info pages,
  - ML info pages (if enabled),
  - one page per chart image.

- `charts/`  
  All chart PNG images.

- `charts_dashboard.html`  
  Simple page to view chart images quickly.

- `ml_results.json`  
  ML metrics and model details (only if `--train` is used).

- `manifest.json`  
  Run metadata (input name, shapes, output paths, timestamps).

## Useful options (simple)

- `--input <path>`  
  Use your own CSV/Excel file.

- `--target <column_name>`  
  Target column name (needed for target analysis / ML).

- `--train`  
  Run ML training and include ML results.

- `--no-charts`  
  Skip chart creation (faster run).

- `--out-dir <path>`  
  Save outputs in a different folder.

- `--max-rows` and `--max-cols`  
  Limit data size if file is too big.

## Example with your own file

```bash
python datascribe_offline_bundle/run_offline.py --input "C:/mydata/sales.csv" --target Sales --train
```

If running from inside offline bundle:

```bash
python run.py --input "C:/mydata/sales.csv" --target Sales --train
```

## Important note

Web mode on localhost is also available (`python run.py`) with login/signup, upload, visualizations, ML, and downloads.
Use this offline bundle when you want everything generated in one folder quickly without opening the web app.

For full app usage details, see the root `README.md`.
