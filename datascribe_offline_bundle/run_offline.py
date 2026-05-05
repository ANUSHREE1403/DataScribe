#!/usr/bin/env python3
"""
Run DataScribe EDA offline: cleaned CSV, summary HTML/PDF, optional PNG charts + baseline ML.

Same core EDA as the web app (`run_eda`). Charts use `core.visualization_engine` (the web UI
does not wire this yet — see bundle README).

Run from repo root:
  python datascribe_offline_bundle/run_offline.py --target Survived
  python datascribe_offline_bundle/run_offline.py --target Survived --train
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from utils.logging_utils import get_logger

logger = get_logger("datascribe.offline", os.path.join(str(Path(__file__).resolve().parents[1]), "reports", "datascribe_offline.log"))
RUN_LOGS: list[str] = []


def _log(message: str) -> None:
    logger.info(message)
    RUN_LOGS.append(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bundle_dir() -> Path:
    return Path(__file__).resolve().parent


def ensure_repo_on_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_raw_csv(path: Path):
    import pandas as pd

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported format: {suffix} (use .csv or .xlsx)")


def clean_like_datascribe(df, max_rows: int | None, max_cols: int | None, target_column: str | None):
    """
    Mirror web load_dataset + analyze trimming (web/main.py), then add duplicate removal
    for a clearly 'cleaned' export.
    """
    import pandas as pd

    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    df = df.drop_duplicates().reset_index(drop=True)

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    if max_cols is not None and df.shape[1] > max_cols:
        if target_column and target_column in df.columns:
            cols = [c for c in df.columns if c != target_column][: max_cols - 1] + [target_column]
            df = df[cols]
        else:
            df = df.iloc[:, :max_cols]

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _json_safe(obj: object) -> object:
    """Make objects JSON-serializable (notably: non-primitive dict keys)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    return obj


def _safe_dumps(obj: object) -> str:
    return json.dumps(_json_safe(obj), indent=2, default=str)


def build_html_report(
    job: dict,
    charts_rel_paths: list[str] | None = None,
    ml_result: dict | None = None,
    processing_logs: list[str] | None = None,
) -> str:
    """Offline HTML report: analysis + charts + ML (when available)."""
    analysis_results = job.get("analysis_results", {})
    overview = analysis_results.get("overview", {})
    quality = analysis_results.get("data_quality", {})
    cols_info = overview.get("columns", {})
    missing_vals = quality.get("missing_values") or {}
    dup_vals = quality.get("duplicates") or {}
    total_missing = missing_vals.get("total_missing", 0)
    dup_count = dup_vals.get("count", 0)
    stats = analysis_results.get("statistics") or {}
    target_analysis = analysis_results.get("target_analysis") or {}

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>DataScribe Report</title>
<style>
body{{font-family:Arial,sans-serif;margin:40px;color:#2c3e50}}
.header{{text-align:center;color:#3498db}}
.section{{margin:20px 0;padding:20px;border:1px solid #ddd;border-radius:8px}}
.metric{{display:inline-block;margin:10px;padding:15px;background:#f8f9fa;border-radius:5px}}
.metric-value{{font-size:24px;font-weight:bold;color:#3498db}}
.metric-label{{color:#666}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
.card{{border:1px solid #e5e7eb;border-radius:12px;padding:12px;background:#fff}}
.thumb{{width:100%;height:auto;border:1px solid #eee;border-radius:10px}}
.muted{{color:#6b7280}}
a.btn{{display:inline-block;padding:10px 14px;border-radius:10px;background:#3498db;color:#fff;text-decoration:none;margin-right:10px}}
a.btn.secondary{{background:#6b7280}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{border:1px solid #e5e7eb;padding:8px;text-align:left;font-size:13px}}
 th{{background:#f8f9fa}}
 code{{background:#f3f4f6;padding:2px 6px;border-radius:6px}}
 pre{{background:#0b1020;color:#e5e7eb;padding:12px;border-radius:10px;overflow:auto;font-size:12px}}
 details{{margin-top:12px}}
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
<div class="metric"><div class="metric-value">{total_missing}</div><div class="metric-label">Missing Values</div></div>
<div class="metric"><div class="metric-value">{dup_count}</div><div class="metric-label">Duplicates</div></div>
</div>
<div class="section"><h2>Columns</h2>
<div class="metric"><div class="metric-value">{cols_info.get('numerical',0)}</div><div class="metric-label">Numerical</div></div>
<div class="metric"><div class="metric-value">{cols_info.get('categorical',0)}</div><div class="metric-label">Categorical</div></div>
</div>
<div class="section"><h2>Data Quality Details</h2>"""

    cols_with_missing = missing_vals.get("columns_with_missing") or []
    if cols_with_missing:
        html += "<p><strong>Columns with missing values:</strong></p><ul>"
        for c in cols_with_missing[:50]:
            html += f"<li><code>{_html_escape(str(c))}</code></li>"
        if len(cols_with_missing) > 50:
            html += f"<li class='muted'>… and {len(cols_with_missing)-50} more</li>"
        html += "</ul>"
    const_cols = quality.get("constant_columns") or []
    if const_cols:
        html += "<p><strong>Constant columns:</strong></p><ul>"
        for c in const_cols[:50]:
            html += f"<li><code>{_html_escape(str(c))}</code></li>"
        if len(const_cols) > 50:
            html += f"<li class='muted'>… and {len(const_cols)-50} more</li>"
        html += "</ul>"
    html += "</div>"

    # Statistics (numerical describe)
    if isinstance(stats, dict) and stats.get("numerical") and isinstance(stats["numerical"], dict):
        num_stats = stats["numerical"]
        cols = [c for c in num_stats.keys() if isinstance(num_stats.get(c), dict)]
        if cols:
            html += "<div class='section'><h2>Numerical Statistics (describe)</h2>"
            html += "<div class='muted'>Showing first 15 numeric columns.</div>"
            html += "<table><thead><tr><th>Column</th><th>count</th><th>mean</th><th>std</th><th>min</th><th>25%</th><th>50%</th><th>75%</th><th>max</th></tr></thead><tbody>"
            for c in cols[:15]:
                d = num_stats.get(c) or {}
                def _fmt(x):
                    try:
                        return f"{float(x):.4g}"
                    except Exception:
                        return _html_escape(str(x))
                html += (
                    f"<tr><td><code>{_html_escape(str(c))}</code></td>"
                    f"<td>{_fmt(d.get('count',''))}</td>"
                    f"<td>{_fmt(d.get('mean',''))}</td>"
                    f"<td>{_fmt(d.get('std',''))}</td>"
                    f"<td>{_fmt(d.get('min',''))}</td>"
                    f"<td>{_fmt(d.get('25%',''))}</td>"
                    f"<td>{_fmt(d.get('50%',''))}</td>"
                    f"<td>{_fmt(d.get('75%',''))}</td>"
                    f"<td>{_fmt(d.get('max',''))}</td></tr>"
                )
            html += "</tbody></table>"
            html += "<details><summary>Raw statistics JSON</summary>"
            html += f"<pre>{_html_escape(_safe_dumps(stats))}</pre></details>"
            html += "</div>"

    # Target analysis (if present)
    if isinstance(target_analysis, dict) and target_analysis:
        html += "<div class='section'><h2>Target Analysis</h2>"
        html += f"<details open><summary>Target analysis JSON</summary><pre>{_html_escape(_safe_dumps(target_analysis))}</pre></details>"
        html += "</div>"

    # Insights
    html += "<div class='section'><h2>Insights</h2>"
    html += "<div class='muted'>These are generated recommendations/warnings from the EDA engine.</div>"
    html += "<ul>"

    insights = analysis_results.get("insights", {})
    for itype, ilist in insights.items():
        if ilist:
            html += f"<h3>{itype.replace('_',' ').title()}</h3><ul>"
            for item in ilist:
                html += f"<li>{_html_escape(str(item))}</li>"
            html += "</ul>"

    # Optional ML block (baseline offline)
    if ml_result and ml_result.get("enabled"):
        html += "<div class='section'><h2>Machine Learning Results</h2>"
        if ml_result.get("error"):
            html += f"<p style='color:#c0392b'><strong>Error:</strong> {ml_result['error']}</p>"
        else:
            html += "<div class='grid'>"
            html += f"<div class='card'><div class='muted'>Task</div><div><strong>{ml_result.get('task','')}</strong></div></div>"
            html += f"<div class='card'><div class='muted'>Model</div><div><strong>{ml_result.get('model_name','')}</strong></div></div>"
            html += f"<div class='card'><div class='muted'>Target</div><div><strong>{ml_result.get('target','')}</strong></div></div>"
            if ml_result.get("task") == "classification":
                acc = ml_result.get("accuracy")
                html += f"<div class='card'><div class='muted'>Accuracy</div><div><strong>{(acc*100):.2f}%</strong></div></div>" if acc is not None else ""
                f1 = ml_result.get("f1")
                html += f"<div class='card'><div class='muted'>F1 (weighted)</div><div><strong>{(f1*100):.2f}%</strong></div></div>" if f1 is not None else ""
            elif ml_result.get("task") == "regression":
                r2 = ml_result.get("r2")
                rmse = ml_result.get("rmse")
                html += f"<div class='card'><div class='muted'>R²</div><div><strong>{r2:.4f}</strong></div></div>" if r2 is not None else ""
                html += f"<div class='card'><div class='muted'>RMSE</div><div><strong>{rmse:.4f}</strong></div></div>" if rmse is not None else ""
            html += "</div></div>"
            cm_file = ml_result.get("confusion_matrix_file")
            if cm_file:
                # Keep it relative if it lives in charts/
                cm_rel = "charts/confusion_matrix_ml.png"
                html += "<div class='section'><h3>Confusion Matrix</h3>"
                html += f"<img class='thumb' style='max-width:650px' src='{cm_rel}' alt='confusion matrix'>"
                html += "</div>"
        html += "<details><summary>Raw ML JSON</summary>"
        html += f"<pre>{_html_escape(_safe_dumps(ml_result))}</pre></details>"
        html += "</div>"

    # Optional charts gallery
    if charts_rel_paths:
        html += "<div class='section'><h2>Visualizations</h2>"
        html += "<p class='muted'>Images are saved under <code>output/charts/</code>. If this page is moved, keep the <code>charts/</code> folder next to it.</p>"
        html += "<div style='margin: 10px 0'>"
        html += "<a class='btn' href='charts_dashboard.html'>Open chart dashboard</a>"
        html += "<a class='btn secondary' href='ml_results.json'>Open ML results (json)</a>"
        html += "<a class='btn secondary' href='manifest.json'>Open manifest (json)</a>"
        html += "<a class='btn secondary' href='cleaned_dataset.csv'>Download cleaned dataset (csv)</a>"
        html += "</div>"
        html += "<div class='grid'>"
        for rel in charts_rel_paths:
            name = rel.split("/")[-1]
            html += f"<div class='card'><div class='muted'>{name}</div><img class='thumb' src='{rel}' alt='{name}'></div>"
        html += "</div></div>"

    # Raw analysis JSON at the end (so you truly have everything in one HTML)
    html += "<div class='section'><h2>Full Analysis JSON</h2>"
    html += "<details><summary>Open full `analysis_results` payload</summary>"
    html += f"<pre>{_html_escape(_safe_dumps(analysis_results))}</pre></details></div>"

    if processing_logs:
        html += "<div class='section'><h2>Processing Logs</h2>"
        html += f"<pre>{_html_escape(chr(10).join(processing_logs[-300:]))}</pre></div>"

    html += "</ul></div></body></html>"
    return html


def _dict_to_lines(d: object, max_lines: int = 1800) -> list[str]:
    try:
        txt = _safe_dumps(d)
    except Exception:
        txt = str(d)
    lines = txt.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... (truncated; {len(lines)-max_lines} more lines)"]
    return lines


def write_report_pdf(
    path: Path,
    job: dict,
    input_name: str,
    charts_dir: Path | None = None,
    ml_result: dict | None = None,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        return False

    analysis = job.get("analysis_results", {})
    overview = analysis.get("overview", {})
    quality = analysis.get("data_quality", {})
    lines = [
        "DataScribe — offline report",
        f"Input: {input_name}",
        f"Generated (UTC): {job.get('created_at', '')}",
        "",
        f"Shape: {overview.get('shape')}",
        f"Quality score: {quality.get('data_quality_score', 0):.1f}",
        f"Total missing cells: {quality.get('missing_values', {}).get('total_missing', 0)}",
        f"Duplicate rows (pre-clean export): see manifest / cleaning steps in README",
    ]
    text = "\n".join(lines)

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        # Summary page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11, va="top", family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Analysis payload page(s)
        analysis_lines = ["Full analysis payload (JSON):", ""] + _dict_to_lines(analysis, max_lines=900)
        for i in range(0, len(analysis_lines), 70):
            chunk = "\n".join(analysis_lines[i : i + 70])
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            ax.text(0.05, 0.95, chunk, transform=ax.transAxes, fontsize=7.5, va="top", family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ML payload page(s)
        if ml_result is not None:
            ml_lines = ["ML results payload (JSON):", ""] + _dict_to_lines(ml_result, max_lines=400)
            for i in range(0, len(ml_lines), 70):
                chunk = "\n".join(ml_lines[i : i + 70])
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off")
                ax.text(0.05, 0.95, chunk, transform=ax.transAxes, fontsize=8, va="top", family="monospace")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # Chart pages (one per PNG)
        if charts_dir and charts_dir.exists():
            pngs = sorted([p for p in charts_dir.iterdir() if p.suffix.lower() == ".png"])
            for p in pngs:
                try:
                    img = plt.imread(p)
                except Exception:
                    continue
                # Landscape tends to fit wide charts better
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis("off")
                ax.set_title(p.name, fontsize=12)
                ax.imshow(img)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    return True


def write_charts_dashboard(charts_dir: Path, created: str) -> Path:
    """List PNGs in charts_dir into dashboard.html (same idea as local_generate_charts.py)."""
    png_files = sorted(p for p in charts_dir.iterdir() if p.suffix.lower() == ".png")
    dashboard = charts_dir.parent / "charts_dashboard.html"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>DataScribe — Charts</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#2c3e50}",
        "img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px;margin:12px 0}",
        ".card{border:1px solid #e5e7eb;padding:16px;border-radius:12px;margin-bottom:18px}</style>",
        "</head><body>",
        "<h1>DataScribe — Visualization Dashboard</h1>",
        f"<p style='color:#6b7280'>Generated: {created}</p>",
        "<p>Open this file from disk; charts load from the <code>charts/</code> folder next to it.</p>",
        "<h2>Charts</h2>",
    ]
    if not png_files:
        parts.append("<p>No PNG charts were generated.</p>")
    else:
        for p in png_files:
            rel = f"charts/{p.name}"
            parts.append(f"<div class='card'><h3>{p.name}</h3><img src='{rel}' alt='{p.name}'/></div>")
    parts.append("</body></html>")
    dashboard.write_text("".join(parts), encoding="utf-8")
    return dashboard


def run_chart_generation(cleaned, analysis_results, target_column, charts_dir: Path) -> dict:
    """Generate PNGs into charts_dir; visualization engine writes filenames there."""
    ensure_repo_on_path()
    from core.visualization_engine import generate_visualizations

    charts_dir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(charts_dir)
    try:
        return generate_visualizations(cleaned, analysis_results, target_column)
    finally:
        os.chdir(cwd)


def run_baseline_ml(cleaned, target_col: str, charts_dir: Path) -> dict:
    """
    sklearn pipeline matching what the results UI expects at a high level (metrics + optional confusion matrix).
    Classification and regression supported via infer_task_type.
    """
    import numpy as np
    import pandas as pd

    ensure_repo_on_path()
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from utils.utils import infer_task_type

    out: dict = {"enabled": True, "target": target_col}
    if target_col not in cleaned.columns:
        out["error"] = f"Target column {target_col!r} not found."
        return out

    df = cleaned.dropna(subset=[target_col]).copy()
    if len(df) < 10:
        out["error"] = "Not enough rows with non-null target after dropping NA (need at least 10)."
        return out

    task = infer_task_type(df, target_col)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    if X.shape[1] == 0:
        out["error"] = "No feature columns left after removing target."
        return out

    def _make_preprocessor(X_feat: pd.DataFrame) -> ColumnTransformer:
        num = X_feat.select_dtypes(include=[np.number]).columns.tolist()
        cat = [c for c in X_feat.columns if c not in num]
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        numeric_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        categorical_tf = Pipeline(
            [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)]
        )
        return ColumnTransformer(
            [("num", numeric_tf, num), ("cat", categorical_tf, cat)],
            remainder="drop",
        )

    if task == "classification":
        y_enc = pd.factorize(y)[0]
        if len(np.unique(y_enc)) < 2:
            out["error"] = "Target has only one class; cannot train."
            return out
        prep = _make_preprocessor(X)
        clf = Pipeline(
            [
                ("prep", prep),
                ("model", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
            ]
        )
        strat = y_enc if len(np.unique(y_enc)) < 20 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=strat
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        out["model_name"] = "Random Forest"
        out["task"] = "classification"
        out["accuracy"] = float(accuracy_score(y_test, pred))
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        out["precision"] = float(report["weighted avg"]["precision"])
        out["recall"] = float(report["weighted avg"]["recall"])
        out["f1"] = float(report["weighted avg"]["f1-score"])
        out["train_size"] = int(len(X_train))
        out["test_size"] = int(len(X_test))
        _num = X.select_dtypes(include=[np.number]).columns.tolist()
        _cat = [c for c in X.columns if c not in _num]
        out["preprocessing"] = {
            "numeric_features": len(_num),
            "categorical_features": len(_cat),
            "final_feature_count": None,
        }
        cm = confusion_matrix(y_test, pred)
        charts_dir.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay

            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", colorbar=True)
            ax.set_title("Confusion matrix (test set)")
            cm_path = charts_dir / "confusion_matrix_ml.png"
            fig.savefig(cm_path, bbox_inches="tight")
            plt.close(fig)
            out["confusion_matrix_file"] = str(cm_path)
        except Exception as e:
            out["confusion_matrix_note"] = f"Could not plot confusion matrix: {e}"
    else:
        y_num = pd.to_numeric(y, errors="coerce")
        mask = y_num.notna()
        X, y_num = X.loc[mask], y_num.loc[mask]
        if len(y_num) < 10:
            out["error"] = "Not enough numeric target rows for regression."
            return out
        prep = _make_preprocessor(X)
        reg = Pipeline(
            [
                ("prep", prep),
                ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.2, random_state=42)
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        out["model_name"] = "Random Forest"
        out["task"] = "regression"
        out["r2"] = float(r2_score(y_test, pred))
        try:
            from sklearn.metrics import root_mean_squared_error

            out["rmse"] = float(root_mean_squared_error(y_test, pred))
        except ImportError:
            out["rmse"] = float(mean_squared_error(y_test, pred, squared=False))
        out["train_size"] = int(len(X_train))
        out["test_size"] = int(len(X_test))
        out["accuracy"] = None
        out["precision"] = None
        out["recall"] = None
        out["f1"] = None

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="DataScribe offline bundle: cleaned CSV + HTML + PDF + manifest.")
    parser.add_argument(
        "--input",
        default=None,
        help="CSV (or Excel) path; default: bundle input/sample_titanic.csv",
    )
    parser.add_argument("--target", default=None, help="Optional target column for supervised-style EDA.")
    parser.add_argument("--out-dir", default=None, help="Output directory; default: bundle output/")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows (same idea as web); omit for no cap.")
    parser.add_argument("--max-cols", type=int, default=None, help="Cap columns; omit for no cap.")
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip PNG charts (core visualization engine).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a baseline Random Forest (needs --target; uses scikit-learn).",
    )
    args = parser.parse_args()
    _log(f"Offline run started. train={args.train}, target={args.target}, no_charts={args.no_charts}")

    bundle = _bundle_dir()
    default_in = bundle / "input" / "sample_titanic.csv"
    in_path = Path(args.input) if args.input else default_in
    if not in_path.is_absolute():
        in_path = (bundle / in_path).resolve() if not in_path.exists() else in_path.resolve()

    out_dir = Path(args.out_dir) if args.out_dir else bundle / "output"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_repo_on_path()
    from core.eda_engine import run_eda

    _log(f"Loading input file: {in_path}")
    raw = load_raw_csv(in_path)
    raw_shape = tuple(raw.shape)
    _log(f"Loaded dataset shape: {raw_shape}")
    cleaned = clean_like_datascribe(
        raw,
        max_rows=args.max_rows,
        max_cols=args.max_cols,
        target_column=args.target,
    )
    _log(f"Cleaned dataset shape: {tuple(cleaned.shape)}")
    cleaned_path = out_dir / "cleaned_dataset.csv"
    cleaned.to_csv(cleaned_path, index=False)
    _log(f"Wrote cleaned dataset: {cleaned_path}")

    analysis_results = run_eda(cleaned, args.target)
    _log("EDA analysis complete.")
    created = datetime.now(timezone.utc).isoformat()
    job = {
        "job_id": "offline-bundle",
        "analysis_results": analysis_results,
        "created_at": created,
    }

    charts_dir = out_dir / "charts"
    plot_map: dict = {}
    dashboard_path: Path | None = None
    if not args.no_charts:
        try:
            plot_map = run_chart_generation(cleaned, analysis_results, args.target, charts_dir)
            dashboard_path = write_charts_dashboard(charts_dir, created)
            _log(f"Generated charts: {len(plot_map)} entries")
        except Exception as e:
            plot_map = {"error": str(e)}
            _log(f"Chart generation failed: {e}")

    pdf_path = out_dir / "report.pdf"
    ml_result: dict | None = None
    if args.train:
        if not args.target:
            ml_result = {"enabled": False, "error": "--train requires --target COLUMN"}
            _log("ML requested without target column.")
        else:
            ml_result = run_baseline_ml(cleaned, args.target, charts_dir)
            _log(f"ML step completed. error={ml_result.get('error') if isinstance(ml_result, dict) else None}")

    pdf_ok = write_report_pdf(
        pdf_path,
        job,
        input_name=in_path.name,
        charts_dir=None if (args.no_charts or "error" in plot_map) else charts_dir,
        ml_result=ml_result,
    )

    charts_rel_paths: list[str] = []
    if not args.no_charts and "error" not in plot_map and charts_dir.exists():
        charts_rel_paths = [f"charts/{p.name}" for p in sorted(charts_dir.iterdir()) if p.suffix.lower() == ".png"]

    html_path = out_dir / "report.html"
    html_path.write_text(
        build_html_report(job, charts_rel_paths=charts_rel_paths, ml_result=ml_result, processing_logs=RUN_LOGS),
        encoding="utf-8",
    )

    manifest = {
        "datascribe_repo_root": str(_repo_root()),
        "input_file": str(in_path),
        "input_basename": in_path.name,
        "target_column": args.target,
        "raw_shape_rows_cols": list(raw_shape),
        "cleaned_shape_rows_cols": list(cleaned.shape),
        "outputs": {
            "cleaned_dataset_csv": str(cleaned_path),
            "report_html": str(html_path),
            "report_pdf": str(pdf_path) if pdf_ok else None,
            "charts_directory": str(charts_dir) if not args.no_charts else None,
            "charts_dashboard_html": str(dashboard_path) if dashboard_path else None,
            "visualization_mapping": plot_map if not args.no_charts else None,
            "ml_results_json": str(out_dir / "ml_results.json") if args.train and ml_result is not None else None,
        },
        "cleaning_steps": [
            "drop rows that are all empty",
            "drop columns that are all empty",
            "drop duplicate rows",
            "optional max_rows / max_cols trimming if flags set",
            "numeric downcasting (int64/float64) like the web app",
        ],
        "generated_at_utc": created,
        "pdf_written": pdf_ok,
        "charts_generated": not args.no_charts and "error" not in plot_map,
        "ml_training": ml_result,
        "processing_logs": RUN_LOGS,
    }
    if not pdf_ok:
        manifest["pdf_note"] = "matplotlib not installed; install requirements or `pip install matplotlib` for PDF."

    if args.train and ml_result is not None:
        (out_dir / "ml_results.json").write_text(
            json.dumps(ml_result, indent=2, default=str), encoding="utf-8"
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "processing_logs.json").write_text(json.dumps(RUN_LOGS, indent=2), encoding="utf-8")

    print("DataScribe offline run complete.")
    print(f"  Cleaned CSV: {cleaned_path}")
    print(f"  HTML report: {html_path}")
    if pdf_ok:
        print(f"  PDF report:  {pdf_path}")
    else:
        print("  PDF skipped (matplotlib unavailable).")
    if not args.no_charts:
        if "error" in plot_map:
            print(f"  Charts:      skipped — {plot_map['error']}")
        else:
            print(f"  Charts:      {charts_dir} ({len(plot_map)} entries)")
            if dashboard_path:
                print(f"  Dashboard:   {dashboard_path}")
    if args.train and ml_result:
        if ml_result.get("error"):
            print(f"  ML train:    error — {ml_result['error']}")
        else:
            print(f"  ML results:  {out_dir / 'ml_results.json'}")
    print(f"  Manifest:    {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
