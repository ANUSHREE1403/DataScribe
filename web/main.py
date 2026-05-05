from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
import os
import uuid
import json
import re
from typing import Optional, Dict, Any
from datetime import datetime

from utils.config import settings
from utils.logging_utils import get_logger
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
generate_visualizations = None
CORE_AVAILABLE = None
VIS_AVAILABLE = None


def _ensure_heavy_imports():
    global pd, run_eda, generate_visualizations, CORE_AVAILABLE, VIS_AVAILABLE
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

    try:
        from core.visualization_engine import generate_visualizations as _generate_visualizations
        generate_visualizations = _generate_visualizations
        VIS_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Visualization engine not available: {e}")
        VIS_AVAILABLE = False


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
job_logs: Dict[str, list[str]] = {}
logger = get_logger("datascribe.web", os.path.join(settings.reports_dir, "datascribe_web.log"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(file: UploadFile):
    try:
        _ensure_heavy_imports()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Core engine failed to load: {e}")
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(file.file)
            except ImportError:
                raise ValueError("Excel support not available. Please upload a CSV file instead.")
        else:
            raise ValueError("Unsupported file format. Please upload a CSV file.")

        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
        return df
    except ValueError:
        raise
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

    logs = job.get("processing_logs", []) or []
    if logs:
        html += "<div class='section'><h2>Processing Logs</h2><pre style='white-space:pre-wrap;background:#f8f9fa;padding:12px;border-radius:8px;'>"
        html += "\n".join([str(x) for x in logs[-200:]])
        html += "</pre></div>"
    html += "</ul></div></body></html>"
    return html


def _log(job_id: str, message: str) -> None:
    line = f"[DataScribe][{job_id}] {message}"
    logger.info(line)
    if job_id not in job_logs:
        job_logs[job_id] = []
    job_logs[job_id].append(line)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:80] or "dataset"


def _json_safe(obj: Any) -> Any:
    """Convert nested objects into JSON-safe structures (stringify non-primitive dict keys)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _write_pdf_report(job_id: str, job: Dict[str, Any]) -> str:
    """Create a downloadable PDF report with summary + generated charts."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF export unavailable: {e}")

    report_path = os.path.join(settings.reports_dir, f"report_{job_id}.pdf")
    analysis = job.get("analysis_results", {}) or {}
    overview = analysis.get("overview", {}) or {}
    quality = analysis.get("data_quality", {}) or {}
    text = "\n".join(
        [
            "DataScribe Report",
            f"Job ID: {job_id}",
            f"Created at: {job.get('created_at', '')}",
            "",
            f"Shape: {overview.get('shape')}",
            f"Memory MB: {overview.get('memory_usage')}",
            f"Quality score: {quality.get('data_quality_score')}",
            f"Total missing: {(quality.get('missing_values') or {}).get('total_missing')}",
            f"Duplicates: {(quality.get('duplicates') or {}).get('count')}",
        ]
    )

    with PdfPages(report_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", family="monospace", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Add generated plot images to PDF.
        # On constrained hosts (e.g. Render free), keep this lightweight.
        include_plots = True
        max_plots = 999
        if os.environ.get("PORT"):
            max_plots = 2

        job_static_dir = os.path.join("static", "jobs", job_id)
        if include_plots and os.path.isdir(job_static_dir):
            added = 0
            for name in sorted(os.listdir(job_static_dir)):
                if not name.lower().endswith(".png"):
                    continue
                img_path = os.path.join(job_static_dir, name)
                try:
                    img = plt.imread(img_path)
                except Exception:
                    continue
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis("off")
                ax.set_title(name, fontsize=12)
                ax.imshow(img)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                added += 1
                if added >= max_plots:
                    break

    return report_path


def _write_excel_report(job_id: str, job: Dict[str, Any]) -> tuple[str, str, str]:
    """Create a downloadable Excel report with key tables; fallback to CSV if openpyxl missing."""
    if pd is None:
        _ensure_heavy_imports()
    report_path = os.path.join(settings.reports_dir, f"report_{job_id}.xlsx")
    analysis = job.get("analysis_results", {}) or {}
    dataset_path = job.get("dataset_path")

    overview = analysis.get("overview", {}) or {}
    quality = analysis.get("data_quality", {}) or {}
    insights = analysis.get("insights", {}) or {}
    summary_df = pd.DataFrame(
        [
            {"metric": "rows", "value": (overview.get("shape") or [None, None])[0]},
            {"metric": "columns", "value": (overview.get("shape") or [None, None])[1]},
            {"metric": "memory_mb", "value": overview.get("memory_usage")},
            {"metric": "quality_score", "value": quality.get("data_quality_score")},
            {"metric": "total_missing", "value": (quality.get("missing_values") or {}).get("total_missing")},
            {"metric": "duplicates", "value": (quality.get("duplicates") or {}).get("count")},
        ]
    )

    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)

            stats = analysis.get("statistics", {}) or {}
            if "numerical" in stats and isinstance(stats["numerical"], dict):
                try:
                    numeric_df = pd.DataFrame(stats["numerical"])
                    numeric_df.to_excel(writer, sheet_name="numeric_stats", index=True)
                except Exception:
                    pass

            insight_rows = []
            for key, vals in insights.items():
                if isinstance(vals, list):
                    for v in vals:
                        insight_rows.append({"category": key, "insight": str(v)})
            if insight_rows:
                pd.DataFrame(insight_rows).to_excel(writer, sheet_name="insights", index=False)

            if dataset_path and os.path.exists(dataset_path):
                try:
                    preview = pd.read_csv(dataset_path).head(5000)
                    preview.to_excel(writer, sheet_name="dataset_preview", index=False)
                except Exception:
                    pass

            ml = job.get("ml")
            if isinstance(ml, dict):
                ml_rows = [{"key": k, "value": str(v)} for k, v in ml.items()]
                pd.DataFrame(ml_rows).to_excel(writer, sheet_name="ml", index=False)

        return (
            report_path,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            f"DataScribe_Report_{_safe_name(str(job.get('filename', job_id)))}.xlsx",
        )
    except Exception as e:
        # Fallback: always provide downloadable CSV if Excel dependency fails.
        _log(job_id, f"Excel writer unavailable ({e}); falling back to CSV summary.")
        fallback_path = os.path.join(settings.reports_dir, f"report_{job_id}_excel_fallback.csv")
        summary_df.to_csv(fallback_path, index=False)
        return (
            fallback_path,
            "text/csv",
            f"DataScribe_Report_{_safe_name(str(job.get('filename', job_id)))}_fallback.csv",
        )


def _write_r_code(job_id: str, job: Dict[str, Any]) -> str:
    dataset_path = job.get("dataset_path", "")
    target = job.get("target_column")
    r_path = os.path.join(settings.reports_dir, f"analysis_{job_id}.R")
    lines = [
        "# Auto-generated by DataScribe",
        f'df <- read.csv("{dataset_path.replace("\\\\", "/")}")',
        "cat('Rows:', nrow(df), 'Cols:', ncol(df), '\\n')",
        "summary(df)",
        "",
    ]
    if target:
        lines += [
            f'target_col <- "{target}"',
            "if (target_col %in% names(df)) {",
            "  print(table(df[[target_col]], useNA = 'ifany'))",
            "}",
        ]
    with open(r_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return r_path


def _generate_job_visualizations(df, analysis_results: Dict[str, Any], target_column: Optional[str], job_id: str) -> Dict[str, str]:
    if not VIS_AVAILABLE or generate_visualizations is None:
        _log(job_id, "Visualization engine unavailable, skipping plot generation.")
        return {}

    job_static_dir = os.path.join("static", "jobs", job_id)
    os.makedirs(job_static_dir, exist_ok=True)
    _log(job_id, f"Generating visualizations in: {job_static_dir}")

    cwd = os.getcwd()
    try:
        os.chdir(job_static_dir)
        raw_plot_map = generate_visualizations(df, analysis_results, target_column) or {}
    finally:
        os.chdir(cwd)

    plot_urls: Dict[str, str] = {}
    for plot_type, plot_file in raw_plot_map.items():
        filename = os.path.basename(str(plot_file))
        abs_path = os.path.join(job_static_dir, filename)
        if os.path.exists(abs_path):
            plot_urls[str(plot_type)] = f"/static/jobs/{job_id}/{filename}"

    _log(job_id, f"Visualization generation complete. {len(plot_urls)} plots available.")
    return plot_urls


def _generate_basic_visualizations(df, target_column: Optional[str], job_id: str) -> Dict[str, str]:
    """
    Render-friendly fallback visualizations (1-2 lightweight charts) for any dataset.
    Generates:
      - missingness bar chart
      - histogram for first numeric column (if present)
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as _pd
    except Exception as e:
        _log(job_id, f"Basic visualization fallback unavailable: {e}")
        return {}

    job_static_dir = os.path.join("static", "jobs", job_id)
    os.makedirs(job_static_dir, exist_ok=True)
    urls: Dict[str, str] = {}

    # 1) Missing values chart (works for any dataset)
    try:
        missing = df.isnull().sum().sort_values(ascending=False)
        if missing.sum() > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            missing.head(20).plot(kind="bar", ax=ax, color="#3498db")
            ax.set_title("Missing values by column")
            ax.set_ylabel("Count")
            ax.set_xlabel("Columns")
            fig.tight_layout()
            out = os.path.join(job_static_dir, "basic_missing_values.png")
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            urls["basic_missing_values"] = f"/static/jobs/{job_id}/basic_missing_values.png"
    except Exception as e:
        _log(job_id, f"Failed missing-values fallback chart: {e}")

    # 2) First numeric histogram
    try:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            col = num_cols[0]
            ser = _pd.to_numeric(df[col], errors="coerce").dropna()
            if not ser.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(ser, bins=30, color="#2ecc71", alpha=0.85)
                ax.set_title(f"Distribution: {col}")
                ax.set_xlabel(str(col))
                ax.set_ylabel("Frequency")
                fig.tight_layout()
                out = os.path.join(job_static_dir, "basic_numeric_distribution.png")
                fig.savefig(out, bbox_inches="tight")
                plt.close(fig)
                urls["basic_numeric_distribution"] = f"/static/jobs/{job_id}/basic_numeric_distribution.png"
    except Exception as e:
        _log(job_id, f"Failed numeric fallback chart: {e}")

    if urls:
        _log(job_id, f"Basic visualization fallback generated {len(urls)} plots.")
    return urls


def _train_baseline_model(df, target_column: str, job_id: str) -> Dict[str, Any]:
    """Train a robust baseline model and return UI-friendly metrics payload."""
    result: Dict[str, Any] = {"enabled": True, "target": target_column}
    if target_column not in df.columns:
        preview = ", ".join([str(c) for c in list(df.columns)[:12]])
        result["error"] = (
            f"Target column '{target_column}' not found. "
            f"Available columns include: {preview}"
        )
        return result

    try:
        import numpy as np
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )
        from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except Exception as e:
        result["error"] = f"ML dependencies not available: {e}"
        return result

    data = df.dropna(subset=[target_column]).copy()
    if len(data) < 20:
        result["error"] = "Need at least 20 non-null target rows for ML."
        return result

    y = data[target_column]
    X = data.drop(columns=[target_column])
    if X.shape[1] == 0:
        result["error"] = "No feature columns available after removing target."
        return result

    # Classification if object dtype or small class cardinality; else regression.
    task = "classification" if str(y.dtype) in ("object", "category", "bool") or y.nunique() < 20 else "regression"
    result["task"] = task

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    _log(job_id, f"ML preprocessing: {len(num_cols)} numeric, {len(cat_cols)} categorical columns.")

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)]), cat_cols),
        ],
        remainder="drop",
    )

    if task == "classification":
        labels, uniques = pd.factorize(y)
        class_balance = {str(uniques[i]): int((labels == i).sum()) for i in range(len(uniques))}
        if len(set(labels)) < 2:
            result["error"] = "Target has only one class; classification needs at least two."
            return result

        model = RandomForestClassifier(n_estimators=160, random_state=42, n_jobs=-1, class_weight="balanced")
        pipe = Pipeline([("prep", prep), ("model", model)])

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        _log(job_id, f"ML train/test split: {len(X_train)} train, {len(X_test)} test.")

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1] if len(set(labels)) == 2 else None

        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        acc = float(accuracy_score(y_test, preds))
        result.update(
            {
                "model_name": "Random Forest",
                "accuracy": acc,
                "precision": float(report["weighted avg"]["precision"]),
                "recall": float(report["weighted avg"]["recall"]),
                "f1": float(report["weighted avg"]["f1-score"]),
                "roc_auc": float(roc_auc_score(y_test, probs)) if probs is not None else None,
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "class_balance": class_balance,
                "preprocessing": {
                    "numeric_features": len(num_cols),
                    "categorical_features": len(cat_cols),
                    "final_feature_count": int(pipe.named_steps["prep"].transform(X_train[:5]).shape[1]),
                },
                "params": pipe.named_steps["model"].get_params(),
            }
        )

        # Cross-validation accuracy
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipe, X, labels, cv=cv, scoring="accuracy")
            result["cv_mean"] = float(cv_scores.mean())
            result["cv_std"] = float(cv_scores.std())
        except Exception as e:
            _log(job_id, f"Cross-validation skipped: {e}")

        # Confusion matrix image
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay

            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", colorbar=True)
            ax.set_title("Confusion Matrix")
            cm_path = os.path.join("static", "jobs", job_id, "confusion_matrix_ml.png")
            os.makedirs(os.path.dirname(cm_path), exist_ok=True)
            fig.savefig(cm_path, bbox_inches="tight")
            plt.close(fig)
            result["confusion_matrix_url"] = f"/static/jobs/{job_id}/confusion_matrix_ml.png"
        except Exception as e:
            _log(job_id, f"Could not save confusion matrix image: {e}")

        # Feature importances
        try:
            feature_names = pipe.named_steps["prep"].get_feature_names_out()
            importances = pipe.named_steps["model"].feature_importances_
            top_pairs = sorted(
                zip([str(n) for n in feature_names], [float(v) for v in importances]),
                key=lambda x: x[1],
                reverse=True,
            )[:12]
            result["top_features"] = top_pairs
        except Exception as e:
            _log(job_id, f"Feature importance unavailable: {e}")

        _log(job_id, f"ML training complete. Accuracy={acc:.4f}")
        return result

    # Regression fallback
    y_num = pd.to_numeric(y, errors="coerce")
    mask = y_num.notna()
    X, y_num = X.loc[mask], y_num.loc[mask]
    if len(y_num) < 20:
        result["error"] = "Not enough numeric target rows for regression."
        return result

    model = RandomForestRegressor(n_estimators=160, random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", prep), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    # Version-safe RMSE: avoid relying on `squared` kwarg availability.
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    result.update(
        {
            "model_name": "Random Forest Regressor",
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "r2": float(r2_score(y_test, preds)),
            "rmse": rmse,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "class_balance": None,
            "preprocessing": {
                "numeric_features": len(num_cols),
                "categorical_features": len(cat_cols),
                "final_feature_count": int(pipe.named_steps["prep"].transform(X_train[:5]).shape[1]),
            },
            "params": pipe.named_steps["model"].get_params(),
            "top_features": None,
        }
    )
    _log(job_id, f"Regression training complete. RMSE={rmse:.4f}")
    return result


def _choose_target_column(df, requested_target: Optional[str], train_model: bool, job_id: str) -> Optional[str]:
    """
    Resolve target column robustly:
    - use requested target if valid
    - if ML requested and missing/invalid, try safe auto-detection and log it
    """
    if requested_target and requested_target in df.columns:
        return requested_target

    if requested_target and requested_target not in df.columns:
        _log(job_id, f"Requested target '{requested_target}' not present in dataset.")

    if not train_model:
        return requested_target

    # Common names first
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for candidate in ("target", "label", "class", "survived", "outcome", "y"):
        if candidate in lower_map:
            chosen = lower_map[candidate]
            _log(job_id, f"Auto-selected target column by name match: {chosen}")
            return chosen

    # Heuristic fallback: first low-cardinality column
    for c in df.columns:
        try:
            nunique = df[c].nunique(dropna=True)
            if 2 <= nunique <= 20:
                _log(job_id, f"Auto-selected target by cardinality heuristic: {c} (unique={nunique})")
                return c
        except Exception:
            continue

    _log(job_id, "Could not auto-detect a target column for ML.")
    return requested_target


def _resolve_target_aliases(df, requested_target: Optional[str], job_id: str) -> Optional[str]:
    """
    Resolve user-entered target names against real columns and common aliases.
    Also creates derived 'Sales' when possible for the sales sample dataset.
    """
    if not requested_target:
        return requested_target

    req = requested_target.strip()
    if not req:
        return None

    # Case-insensitive direct match
    for c in df.columns:
        if str(c).strip().lower() == req.lower():
            if c != requested_target:
                _log(job_id, f"Mapped requested target '{requested_target}' to actual column '{c}'.")
            return str(c)

    col_map = {str(c).strip().lower(): str(c) for c in df.columns}

    # Common financial aliases
    if req.lower() == "sales":
        # Create derived Sales if quantity + unit price exist
        q_col = col_map.get("quantity ordered")
        p_col = col_map.get("price each")
        if q_col and p_col:
            try:
                df["Sales"] = pd.to_numeric(df[q_col], errors="coerce").fillna(0) * pd.to_numeric(
                    df[p_col], errors="coerce"
                ).fillna(0)
                _log(job_id, "Created derived target column 'Sales' = Quantity Ordered * Price Each.")
                return "Sales"
            except Exception as e:
                _log(job_id, f"Could not derive 'Sales' column: {e}")

        for fallback in ("net amount", "gross amount", "quantity ordered"):
            if fallback in col_map:
                mapped = col_map[fallback]
                _log(job_id, f"Mapped requested target 'Sales' to '{mapped}'.")
                return mapped

    if req.lower() in ("profit", "revenue", "amount"):
        for fallback in ("net amount", "gross amount"):
            if fallback in col_map:
                mapped = col_map[fallback]
                _log(job_id, f"Mapped requested target '{requested_target}' to '{mapped}'.")
                return mapped

    return requested_target


def _job_meta_path(job_id: str) -> str:
    return os.path.join(settings.reports_dir, f"job_{job_id}.json")


def _save_job_artifact(job_id: str, job: Dict[str, Any], db=None) -> None:
    try:
        os.makedirs(settings.reports_dir, exist_ok=True)
        with open(_job_meta_path(job_id), "w", encoding="utf-8") as f:
            json.dump(_json_safe(job), f, indent=2, default=str)
    except Exception as e:
        _log(job_id, f"Could not save job artifact: {e}")
    # Optional DB persistence (survives process memory reset when DB is persistent)
    if db is not None:
        try:
            row = db.query(AnalysisJob).filter(AnalysisJob.job_id == job_id).first()
            if row:
                row.notes = json.dumps(_json_safe(job), default=str)
                db.commit()
        except Exception as e:
            _log(job_id, f"Could not save job artifact to DB notes: {e}")


def _load_job_artifact(job_id: str, db=None) -> Optional[Dict[str, Any]]:
    path = _job_meta_path(job_id)
    if not os.path.exists(path):
        data = None
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            data = None
    if db is not None:
        try:
            row = db.query(AnalysisJob).filter(AnalysisJob.job_id == job_id).first()
            if row and row.notes:
                from_db = json.loads(row.notes)
                if isinstance(from_db, dict):
                    return from_db
        except Exception:
            pass
    return None


def _get_job_or_404(job_id: str, db=None) -> Dict[str, Any]:
    job = jobs.get(job_id)
    if job:
        return job
    disk_job = _load_job_artifact(job_id, db=db)
    if disk_job:
        jobs[job_id] = disk_job
        return disk_job
    raise HTTPException(status_code=404, detail="Job not found")


def _download_unavailable_response(file_label: str, job_id: Optional[str] = None, reason: Optional[str] = None) -> HTMLResponse:
    """Friendly fallback page shown when downloads fail on constrained environments."""
    safe_job = _safe_name(job_id) if job_id else "unknown"
    safe_reason = str(reason or "temporary server constraint")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DataScribe - Download Unavailable</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
  <style>
    *{{box-sizing:border-box}}
    body{{font-family:'Inter',Arial,sans-serif;margin:0;color:#2c3e50;background:#f7f9fc}}
    .header{{background:#fff;box-shadow:0 2px 20px rgba(0,0,0,.08);padding:14px 0}}
    .header-inner{{max-width:980px;margin:0 auto;padding:0 20px;display:flex;align-items:center;justify-content:space-between}}
    .brand{{font-weight:800;font-size:1.6rem;background:linear-gradient(135deg,#2c3e50,#3498db);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
    .home-link{{text-decoration:none;color:#3498db;font-weight:700}}
    .wrap{{max-width:980px;margin:30px auto;padding:0 20px}}
    .card{{background:#fff;border:1px solid rgba(52,152,219,.2);border-radius:18px;box-shadow:0 12px 30px rgba(0,0,0,.08);overflow:hidden}}
    .banner{{padding:22px 24px;background:linear-gradient(135deg,#ffe9e9,#ffd9d9);border-bottom:1px solid #f2b5b5}}
    .banner h1{{margin:0;color:#c0392b;font-size:1.9rem;line-height:1.25}}
    .body{{padding:24px}}
    p{{line-height:1.75;margin:0 0 12px}}
    .meta{{margin-top:16px;padding:14px;border-radius:12px;background:#f8f9fa;border:1px solid #e5e7eb;font-size:.95rem}}
    .meta strong{{color:#2c3e50}}
    .actions{{margin-top:18px;display:flex;flex-wrap:wrap;gap:10px}}
    .btn{{display:inline-flex;align-items:center;justify-content:center;padding:11px 16px;border-radius:10px;text-decoration:none;font-weight:700;transition:.2s}}
    .btn.email{{background:#e74c3c;color:#fff}}
    .btn.home{{background:#3498db;color:#fff}}
    .btn:hover{{transform:translateY(-1px);box-shadow:0 8px 18px rgba(0,0,0,.12)}}
    .muted{{color:#7f8c8d;font-size:.85rem;margin-top:10px}}
  </style>
</head>
<body>
  <header class="header">
    <div class="header-inner">
      <div class="brand">DataScribe</div>
      <a class="home-link" href="/">Back to Home</a>
    </div>
  </header>
  <div class="wrap">
    <div class="card">
      <div class="banner">
        <h1>We are really sorry - this download is temporarily unavailable</h1>
      </div>
      <div class="body">
        <p>
          You are very close to your goal, and we truly understand how frustrating this feels.
          Right now, because we are running on Render free tier, large file generation/download can fail due to memory or timeout limits.
        </p>
        <p>
          Please bear with us for now. If you email us, we will be more than happy to help you quickly and share the required file.
          Thank you so much for your patience and support.
        </p>
        <div class="meta">
          <div><strong>Requested file:</strong> {file_label}</div>
          <div><strong>Job ID:</strong> {safe_job}</div>
        </div>
        <div class="actions">
          <a class="btn email" href="mailto:workanushree14@gmail.com?subject=DataScribe%20Download%20Help%20-%20{safe_job}&body=Hi%20DataScribe%20team,%0A%0AI%20could%20not%20download%20the%20file%20from%20job%20ID%20{safe_job}.%20Please%20help.%0A%0AThank%20you.">Email Support</a>
          <a class="btn home" href="/">Back to Home</a>
        </div>
        <div class="muted">Internal note: {safe_reason}</div>
      </div>
    </div>
  </div>
</body>
</html>"""
    return HTMLResponse(content=html, status_code=200)


def _render_template(request: Request, name: str, context: Dict[str, Any], status_code: int = 200):
    """
    Compatibility wrapper for Starlette/FastAPI TemplateResponse signatures.
    Supports both:
      - TemplateResponse(name, context, status_code=...)
      - TemplateResponse(request=request, name=..., context=..., status_code=...)
    """
    try:
        return templates.TemplateResponse(
            request=request,
            name=name,
            context=context,
            status_code=status_code,
        )
    except TypeError:
        return templates.TemplateResponse(name, context, status_code=status_code)


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
    return _render_template(request, "index.html", {"request": request, "user": user, "error": error})


@app.get("/signup", response_class=HTMLResponse)
async def signup_get(request: Request, db=Depends(get_db)):
    user = get_current_user(request, db)
    if user:
        return RedirectResponse(url="/", status_code=303)
    return _render_template(request, "signup.html", {"request": request, "message": None})


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
        return _render_template(request, "signup.html", {"request": request, "message": str(e)})


@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request, db=Depends(get_db)):
    user = get_current_user(request, db)
    if user:
        return RedirectResponse(url="/", status_code=303)
    return _render_template(request, "login.html", {"request": request, "message": None})


@app.post("/login", response_class=HTMLResponse)
async def login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db=Depends(get_db),
):
    user = authenticate_user(db, email=email.strip().lower(), password=password)
    if not user:
        return _render_template(
            request,
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
    include_plots: bool = Form(True),
    train_model: bool = Form(False),
    model_choice: str = Form("auto"),
    include_python_code: bool = Form(False),
):
    try:
        _ensure_heavy_imports()
    except Exception as e:
        return RedirectResponse(url="/?error=" + str(e).replace(" ", "+"), status_code=303)

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
    _log(job_id, f"New analysis request received. include_plots={include_plots}, train_model={train_model}, model_choice={model_choice}")

    try:
        _log(job_id, f"Loading file: {file.filename}")
        df = load_dataset(file)
        _log(job_id, f"Loaded dataset shape: {df.shape}")

        target_column = _resolve_target_aliases(df, target_column, job_id)
        target_column = _choose_target_column(df, target_column, train_model, job_id)
        _log(job_id, f"Target column in use: {target_column}")

        max_rows = int(os.environ.get("MAX_ANALYSIS_ROWS", "3000"))
        max_cols = int(os.environ.get("MAX_ANALYSIS_COLS", "20"))
        original_shape = df.shape

        if df.shape[0] > max_rows:
            _log(job_id, f"Row count {df.shape[0]} > {max_rows}; sampling for analysis.")
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        if df.shape[1] > max_cols:
            _log(job_id, f"Column count {df.shape[1]} > {max_cols}; trimming for analysis.")
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
        _log(job_id, f"Saved processed dataset snapshot: {dataset_path}")

        if not CORE_AVAILABLE:
            raise RuntimeError("Core analysis engine not available")

        # Run EDA
        _log(job_id, f"Running EDA. current_shape={df.shape}, original_shape={original_shape}")
        try:
            analysis_results = run_eda(df, target_column)
        except Exception as eda_err:
            _log(job_id, f"EDA error: {eda_err}")
            analysis_results = {
                "overview": {"shape": list(df.shape), "columns": list(df.columns[:10])},
                "data_quality": {"data_quality_score": 0},
                "statistics": {},
                "univariate": {},
                "bivariate": {},
                "multivariate": {},
                "insights": {"warnings": [f"Analysis error: {eda_err}"]},
            }

        visualization_urls: Dict[str, str] = {}
        if include_plots:
            try:
                visualization_urls = _generate_job_visualizations(df, analysis_results, target_column, job_id)
                if not visualization_urls:
                    _log(job_id, "Primary visualization engine returned no plots; using basic fallback plots.")
                    visualization_urls = _generate_basic_visualizations(df, target_column, job_id)
            except Exception as vis_err:
                _log(job_id, f"Visualization generation failed: {vis_err}")
                visualization_urls = _generate_basic_visualizations(df, target_column, job_id)
        else:
            _log(job_id, "Visualization generation skipped (include_plots disabled).")

        ml_result = None
        if train_model and target_column:
            try:
                _log(job_id, "Starting ML training step.")
                ml_result = _train_baseline_model(df, target_column, job_id)
            except Exception as ml_err:
                _log(job_id, f"ML training failed: {ml_err}")
                ml_result = {"enabled": True, "target": target_column, "error": str(ml_err)}
        elif train_model and not target_column:
            ml_result = {"enabled": True, "target": None, "error": "Select a target column to run ML training."}
            _log(job_id, "ML requested but no target column was provided.")
        else:
            _log(job_id, "ML training skipped by user choice.")

        jobs[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "dataset_path": dataset_path,
            "analysis_results": analysis_results,
            "visualization_urls": visualization_urls,
            "target_column": target_column,
            "filename": original_filename,
            "created_at": datetime.now().isoformat(),
            "ml": ml_result,
            "processing_logs": job_logs.get(job_id, []),
        }
        _log(job_id, f"Job complete. visuals={len(visualization_urls)} ml={'yes' if ml_result else 'no'}")
        jobs[job_id]["processing_logs"] = job_logs.get(job_id, [])
        _save_job_artifact(job_id, jobs[job_id], db=db)

        ml_accuracy = None
        if isinstance(ml_result, dict):
            ml_accuracy = ml_result.get("accuracy")

        try:
            create_analysis_job(
                db,
                user_id=user.id,
                job_id=job_id,
                dataset_name=original_filename,
                target_column=target_column,
                model_choice=model_choice,
                accuracy=float(ml_accuracy) if ml_accuracy is not None else None,
            )
        except Exception as db_err:
            _log(job_id, f"DB save warning: {db_err}")

        del df
        gc.collect()

        return RedirectResponse(url=f"/results/{job_id}", status_code=303)

    except HTTPException:
        raise
    except Exception as e:
        _log(job_id, f"Analysis failed: {e}")
        from urllib.parse import quote
        return RedirectResponse(url="/?error=" + quote("Analysis failed. Try a smaller file."), status_code=303)


# ---------------------------------------------------------------------------
# Results & Downloads
# ---------------------------------------------------------------------------

@app.get("/results/{job_id}")
async def get_results(request: Request, job_id: str, db=Depends(get_db)):
    user = get_current_user(request, db)
    job = _get_job_or_404(job_id, db=db)
    if job["status"] == "failed":
        return _render_template(request, "error.html", {"request": request, "error": job.get("error", "Unknown error")})

    return _render_template(
        request,
        "results.html",
        {"request": request, "job": job, "analysis_results": job["analysis_results"], "user": user},
    )


@app.get("/download/{job_id}/dataset")
async def download_dataset(job_id: str, db=Depends(get_db)):
    try:
        job = _get_job_or_404(job_id, db=db)
        dataset_path = job.get("dataset_path")
        if not dataset_path or not os.path.exists(dataset_path):
            raise FileNotFoundError("dataset file not found")
        return FileResponse(dataset_path, media_type="text/csv", filename=f"DataScribe_{job_id}.csv")
    except Exception as e:
        _log(job_id, f"Dataset download failed: {e}")
        return _download_unavailable_response("Dataset CSV", job_id=job_id, reason=e)


@app.get("/download/{job_id}/report/html")
async def download_html_report(job_id: str, db=Depends(get_db)):
    try:
        job = _get_job_or_404(job_id, db=db)
        html_content = generate_html_report(job)
        report_path = os.path.join(settings.reports_dir, f"report_{job_id}.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return FileResponse(report_path, media_type="text/html", filename=f"DataScribe_Report_{job_id}.html")
    except Exception as e:
        _log(job_id, f"HTML download failed: {e}")
        return _download_unavailable_response("HTML report", job_id=job_id, reason=e)


@app.get("/download/{job_id}/report/pdf")
async def download_pdf_report(job_id: str, db=Depends(get_db)):
    try:
        job = _get_job_or_404(job_id, db=db)
        _log(job_id, "Preparing PDF download.")
        dataset_label = _safe_name(str(job.get("filename", job_id)))
        report_path = _write_pdf_report(job_id, job)
        return FileResponse(report_path, media_type="application/pdf", filename=f"DataScribe_Report_{dataset_label}.pdf")
    except Exception as e:
        _log(job_id, f"PDF download failed: {e}")
        return _download_unavailable_response("PDF report", job_id=job_id, reason=e)


@app.get("/download/{job_id}/report/excel")
async def download_excel_report(job_id: str, db=Depends(get_db)):
    try:
        job = _get_job_or_404(job_id, db=db)
        _log(job_id, "Preparing Excel download.")
        report_path, media_type, download_name = _write_excel_report(job_id, job)
        return FileResponse(
            report_path,
            media_type=media_type,
            filename=download_name,
        )
    except Exception as e:
        _log(job_id, f"Excel download failed: {e}")
        return _download_unavailable_response("Excel report", job_id=job_id, reason=e)


@app.get("/download/{job_id}/code/r")
async def download_r_code(job_id: str, db=Depends(get_db)):
    try:
        job = _get_job_or_404(job_id, db=db)
        _log(job_id, "Preparing R code download.")
        r_path = _write_r_code(job_id, job)
        dataset_label = _safe_name(str(job.get("filename", job_id)))
        return FileResponse(r_path, media_type="text/plain", filename=f"DataScribe_Analysis_{dataset_label}.R")
    except Exception as e:
        _log(job_id, f"R code download failed: {e}")
        return _download_unavailable_response("R code", job_id=job_id, reason=e)


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
    rows = (
        db.query(AnalysisJob)
        .filter(AnalysisJob.user_id == user.id)
        .order_by(AnalysisJob.created_at.desc())
        .all()
    )

    analyses: List[Dict[str, Any]] = []
    for row in rows:
        ml = None
        if row.notes:
            try:
                payload = json.loads(row.notes)
                if isinstance(payload, dict):
                    ml = payload.get("ml")
            except Exception:
                ml = None

        ml_task = None
        if isinstance(ml, dict):
            ml_task = ml.get("task")

        analyses.append(
            {
                "job_id": row.job_id,
                "created_at": row.created_at,
                "dataset_name": row.dataset_name,
                "target_column": row.target_column,
                "model_choice": row.model_choice,
                "ml_task": ml_task,
                "accuracy": float(ml.get("accuracy")) if isinstance(ml, dict) and ml.get("accuracy") is not None else row.accuracy,
                "rmse": float(ml.get("rmse")) if isinstance(ml, dict) and ml.get("rmse") is not None else None,
                "r2": float(ml.get("r2")) if isinstance(ml, dict) and ml.get("r2") is not None else None,
            }
        )

    return _render_template(request, "history.html", {"request": request, "user": user, "analyses": analyses})
