from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
import os
import uuid
import json
from typing import Optional
from datetime import datetime

from utils.config import settings
from web.auth import (
    AnalysisJob,
    authenticate_user,
    create_analysis_job,
    create_user,
    get_current_user,
    get_db,
)

# Heavy libraries loaded lazily on first analysis
pd = None
run_eda = None
CORE_AVAILABLE = None


def _ensure_heavy_imports():
    global pd, run_eda, CORE_AVAILABLE
    if pd is not None:
        return

    import pandas as _pd
    pd = _pd

    try:
        from core.eda_engine import run_eda as _run_eda
        run_eda = _run_eda
        CORE_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Core EDA engine not available: {e}")
        CORE_AVAILABLE = False


app = FastAPI(
    title=settings.app_name,
    description=settings.app_subtitle,
    version=settings.app_version,
)

app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)

templates = Jinja2Templates(directory="web/templates")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.reports_dir, exist_ok=True)

jobs = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(file: UploadFile):
    _ensure_heavy_imports()
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.file)
        elif file.filename.endswith(".parquet"):
            df = pd.read_parquet(file.file)
        else:
            raise ValueError("Unsupported file format")

        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading dataset: {e}")


def generate_html_report(job: dict) -> str:
    analysis_results = job.get("analysis_results", {})
    os.makedirs(settings.reports_dir, exist_ok=True)

    overview = analysis_results.get("overview", {})
    quality = analysis_results.get("data_quality", {})
    cols_info = overview.get("columns", {})

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>DataScribe Report</title>
<style>
body{{font-family:Arial,sans-serif;margin:40px;color:#2c3e50}}
.header{{text-align:center;color:#3498db}}
.section{{margin:20px 0;padding:20px;border:1px solid #ddd;border-radius:8px}}
.metric{{display:inline-block;margin:10px;padding:15px;background:#f8f9fa;border-radius:5px}}
.metric-value{{font-size:24px;font-weight:bold;color:#3498db}}
.metric-label{{color:#666}}
</style></head><body>
<div class="header"><h1>DataScribe Analysis Report</h1>
<p>Generated: {job.get('created_at','')}</p></div>
<div class="section"><h2>Dataset Overview</h2>
<div class="metric"><div class="metric-value">{overview.get('shape',[0,0])[0]}</div><div class="metric-label">Rows</div></div>
<div class="metric"><div class="metric-value">{overview.get('shape',[0,0])[1]}</div><div class="metric-label">Columns</div></div>
<div class="metric"><div class="metric-value">{overview.get('memory_usage',0):.2f} MB</div><div class="metric-label">Memory</div></div>
</div>
<div class="section"><h2>Data Quality</h2>
<div class="metric"><div class="metric-value">{quality.get('data_quality_score',0):.1f}%</div><div class="metric-label">Quality Score</div></div>
<div class="metric"><div class="metric-value">{quality.get('missing_values',{{}}).get('total_missing',0)}</div><div class="metric-label">Missing Values</div></div>
<div class="metric"><div class="metric-value">{quality.get('duplicates',{{}}).get('count',0)}</div><div class="metric-label">Duplicates</div></div>
</div>
<div class="section"><h2>Columns</h2>
<div class="metric"><div class="metric-value">{cols_info.get('numerical',0)}</div><div class="metric-label">Numerical</div></div>
<div class="metric"><div class="metric-value">{cols_info.get('categorical',0)}</div><div class="metric-label">Categorical</div></div>
</div>
<div class="section"><h2>Insights</h2><ul>"""

    insights = analysis_results.get("insights", {})
    for itype, ilist in insights.items():
        if ilist:
            html += f"<h3>{itype.replace('_',' ').title()}</h3><ul>"
            for item in ilist:
                html += f"<li>{item}</li>"
            html += "</ul>"

    html += "</ul></div></body></html>"
    return html


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db=Depends(get_db)):
    try:
        user = get_current_user(request, db)
    except Exception:
        user = None
    error = request.query_params.get("error", "")
    return templates.TemplateResponse("index.html", {"request": request, "user": user, "error": error})


@app.get("/signup", response_class=HTMLResponse)
async def signup_get(request: Request, db=Depends(get_db)):
    user = get_current_user(request, db)
    if user:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("signup.html", {"request": request, "message": None})


@app.post("/signup", response_class=HTMLResponse)
async def signup_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    full_name: Optional[str] = Form(None),
    db=Depends(get_db),
):
    user = get_current_user(request, db)
    if user:
        return RedirectResponse(url="/", status_code=303)
    try:
        user = create_user(db, email=email.strip().lower(), password=password, full_name=full_name)
        request.session["user_id"] = user.id
        return RedirectResponse(url="/", status_code=303)
    except ValueError as e:
        return templates.TemplateResponse("signup.html", {"request": request, "message": str(e)})


@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request, db=Depends(get_db)):
    user = get_current_user(request, db)
    if user:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "message": None})


@app.post("/login", response_class=HTMLResponse)
async def login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db=Depends(get_db),
):
    user = authenticate_user(db, email=email.strip().lower(), password=password)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "message": "Invalid email or password. On free hosting, accounts may reset after redeploys — try signing up again.",
            },
        )
    request.session["user_id"] = user.id
    return RedirectResponse(url="/", status_code=303)


@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=303)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@app.get("/analyze")
async def analyze_get():
    return RedirectResponse(url="/", status_code=303)


@app.post("/analyze")
async def analyze_dataset(
    request: Request,
    db=Depends(get_db),
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    include_plots: bool = Form(False),
    train_model: bool = Form(False),
    model_choice: str = Form("auto"),
    include_python_code: bool = Form(False),
):
    _ensure_heavy_imports()

    try:
        user = get_current_user(request, db)
    except Exception:
        return RedirectResponse(url="/?error=Database+unavailable.+Try+again.", status_code=303)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not any(file.filename.endswith(ext) for ext in settings.allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Unsupported format. Allowed: {', '.join(settings.allowed_extensions)}")

    # File size guard
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    max_size = 10 * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(status_code=400, detail=f"File too large ({file_size/1024/1024:.1f}MB). Max 10MB.")

    job_id = str(uuid.uuid4())

    try:
        df = load_dataset(file)

        max_rows = int(os.environ.get("MAX_ANALYSIS_ROWS", "3000"))
        max_cols = int(os.environ.get("MAX_ANALYSIS_COLS", "20"))
        original_shape = df.shape

        if df.shape[0] > max_rows:
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        if df.shape[1] > max_cols:
            if target_column and target_column in df.columns:
                cols = [c for c in df.columns if c != target_column][: max_cols - 1] + [target_column]
                df = df[cols]
            else:
                df = df.iloc[:, :max_cols]

        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        import gc
        gc.collect()

        dataset_path = os.path.join(settings.upload_dir, f"{job_id}.csv")
        df.to_csv(dataset_path, index=False)
        original_filename = file.filename

        if not CORE_AVAILABLE:
            raise RuntimeError("Core analysis engine not available")

        # Run EDA
        print(f"Running EDA for job {job_id}, shape {df.shape} (original {original_shape})")
        try:
            analysis_results = run_eda(df, target_column)
        except Exception as eda_err:
            print(f"EDA error: {eda_err}")
            analysis_results = {
                "overview": {"shape": list(df.shape), "columns": list(df.columns[:10])},
                "data_quality": {"data_quality_score": 0},
                "statistics": {},
                "univariate": {},
                "bivariate": {},
                "multivariate": {},
                "insights": {"warnings": [f"Analysis error: {eda_err}"]},
            }

        del df
        gc.collect()

        jobs[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "dataset_path": dataset_path,
            "analysis_results": analysis_results,
            "visualization_urls": {},
            "target_column": target_column,
            "filename": original_filename,
            "created_at": datetime.now().isoformat(),
            "ml": None,
        }

        try:
            create_analysis_job(
                db,
                user_id=user.id,
                job_id=job_id,
                dataset_name=original_filename,
                target_column=target_column,
                model_choice=model_choice,
                accuracy=None,
            )
        except Exception as db_err:
            print(f"DB save warning: {db_err}")

        return RedirectResponse(url=f"/results/{job_id}", status_code=303)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Analysis failed for {job_id}: {e}")
        from urllib.parse import quote
        return RedirectResponse(url="/?error=" + quote("Analysis failed. Try a smaller file."), status_code=303)


# ---------------------------------------------------------------------------
# Results & Downloads
# ---------------------------------------------------------------------------

@app.get("/results/{job_id}")
async def get_results(request: Request, job_id: str, db=Depends(get_db)):
    user = get_current_user(request, db)
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] == "failed":
        return templates.TemplateResponse("error.html", {"request": request, "error": job.get("error", "Unknown error")})

    return templates.TemplateResponse(
        "results.html",
        {"request": request, "job": job, "analysis_results": job["analysis_results"], "user": user},
    )


@app.get("/download/{job_id}/dataset")
async def download_dataset(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    dataset_path = jobs[job_id].get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    return FileResponse(dataset_path, media_type="text/csv", filename=f"DataScribe_{job_id}.csv")


@app.get("/download/{job_id}/report/html")
async def download_html_report(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    html_content = generate_html_report(job)
    report_path = os.path.join(settings.reports_dir, f"report_{job_id}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return FileResponse(report_path, media_type="text/html", filename=f"DataScribe_Report_{job_id}.html")


# ---------------------------------------------------------------------------
# Health & History
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "healthy", "app_name": settings.app_name, "version": settings.app_version}


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request, db=Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    analyses = (
        db.query(AnalysisJob)
        .filter(AnalysisJob.user_id == user.id)
        .order_by(AnalysisJob.created_at.desc())
        .all()
    )
    return templates.TemplateResponse("history.html", {"request": request, "user": user, "analyses": analyses})
