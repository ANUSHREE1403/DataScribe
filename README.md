# DataScribe (Simple Guide)

DataScribe helps you analyze a dataset quickly and get easy-to-read outputs like:
- data quality checks,
- summaries,
- charts,
- reports.

It supports:
- login/signup,
- dataset upload from browser,
- automatic EDA,
- visualizations,
- optional ML training,
- downloadable report files.

## What this project has

There are 2 main ways to use DataScribe:

1. **Web app mode** (login + upload file in browser)  
   Run with: `python run.py`

2. **Offline bundle mode** (all outputs in one folder, no web UI needed)  
   Run with: `python datascribe_offline_bundle/run_offline.py ...`

If you want complete files in one place (CSV + charts + HTML + PDF + ML JSON), use the offline bundle.

## Quick Start (Beginner)

### 1) Open terminal in project folder

```bash
cd path/to/DataScribe
```

### 2) Install requirements

```bash
pip install -r requirements.txt
```

### 3) Start the web app

```bash
python run.py
```

Now open: `http://localhost:8000`

## How to use the web app

1. Sign up / log in.
2. Upload CSV or Excel file.
3. (Optional) set target column.
4. Run analysis.
5. View results and download available report files.

Download buttons available on results page:
- PDF report
- Excel report
- HTML report
- R code
- processed dataset CSV

## Offline bundle (recommended for full output package)

This folder gives you all outputs in one clean place:

`datascribe_offline_bundle/`

Run this:

```bash
python datascribe_offline_bundle/run_offline.py --target Survived --train
```

You will get files inside:

`datascribe_offline_bundle/output/`

Main output files:
- `cleaned_dataset.csv`
- `report.html`
- `report.pdf`
- `charts/` (PNG chart files)
- `charts_dashboard.html`
- `ml_results.json` (if `--train`)
- `manifest.json`

For detailed offline steps, read:
`datascribe_offline_bundle/README.md`

For CD/DVD submission instructions, read:
`BURN_TO_CD_INSTRUCTIONS.txt`

## Input file formats

- `.csv`
- `.xlsx`
- `.xls`

Example datasets are in `tests/`.

## Common commands

### Web app

```bash
python run.py
```

### Web app with local URL

After starting, open:

```text
http://127.0.0.1:8000
```

### Offline full report with ML

```bash
python datascribe_offline_bundle/run_offline.py --target Survived --train
```

### Offline with your own file

```bash
python datascribe_offline_bundle/run_offline.py --input "C:/mydata/file.csv" --target YourTarget --train
```

### Run tests

```bash
pytest tests/
```

## Main folders (quick reference)

- `core/` → EDA + chart generation logic
- `web/` → FastAPI app, auth, templates
- `utils/` → config + helper functions
- `tests/` → sample datasets
- `datascribe_offline_bundle/` → offline all-in-one runner
- `uploads/` → uploaded dataset snapshots
- `reports/` → generated download files

## Optional `.env` settings

You can create a `.env` file in project root.  
If you skip it, defaults still work locally.

Example:

```env
APP_NAME=DataScribe
DEBUG=False
SECRET_KEY=change-this-in-production
UPLOAD_DIR=uploads
REPORTS_DIR=reports
```

## Deployment (short)

- For simple cloud deploy (Render/Railway):
  - Build command: `pip install -r requirements.txt`
  - Start command: `python run.py`
  - Set `SECRET_KEY` in environment variables.

## Need help?

- GitHub Issues: [DataScribe Issues](https://github.com/ANUSHREE1403/DataScribe/issues)
- GitHub Discussions: [DataScribe Discussions](https://github.com/ANUSHREE1403/DataScribe/discussions)
- Email: `workanushree14@gmail.com`

## License

MIT License. See `LICENSE`.
