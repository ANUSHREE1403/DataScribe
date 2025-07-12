from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import shutil
import os
import uuid
from utils import infer_task_type, save_model
from eda import run_eda
from preprocess import preprocess_data
from train_models import train_and_return_models
from evaluate import evaluate_models, print_model_comparison
import joblib
import threading
import json
from typing import List

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Ensure templates directory exists
if not os.path.exists("templates"):
    os.makedirs("templates")

JOBS_DIR = "jobs"
os.makedirs(JOBS_DIR, exist_ok=True)

def save_job(job_id, data):
    with open(os.path.join(JOBS_DIR, f"{job_id}.json"), "w") as f:
        json.dump(data, f)

def load_job(job_id):
    try:
        with open(os.path.join(JOBS_DIR, f"{job_id}.json"), "r") as f:
            return json.load(f)
    except Exception:
        return None

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/train", response_class=HTMLResponse)
def train(request: Request,
          file: UploadFile = File(...),
          target: str = Form(...),
          task: str = Form(None),
          test_size: float = Form(0.2),
          models: List[str] = Form(...)):
    job_id = str(uuid.uuid4())
    job_data = {"status": "running", "result": None}
    save_job(job_id, job_data)
    file_id = job_id
    file_path = f"uploaded_{file_id}.csv"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    def background_train():
        try:
            df = pd.read_csv(file_path)
            run_eda(df, target)
            X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, target, test_size)
            task_type = task or infer_task_type(df, target)
            trained_models = train_and_return_models(X_train, y_train, task_type, selected_models=models)
            results = evaluate_models(trained_models, X_test, y_test, task_type)
            if task_type == 'classification':
                best_model_name = max(results, key=lambda k: results[k]['F1'])
            else:
                best_model_name = max(results, key=lambda k: results[k]['R2'])
            model_paths = {}
            for model_name, model in trained_models.items():
                model_path = f"model_{file_id}_{model_name}.pkl"
                save_model(model, model_path)
                model_paths[model_name] = model_path
            joblib.dump(preprocessor, f"preprocessor_{file_id}.pkl")
            df_results = pd.DataFrame(results).T
            metrics_html = df_results.to_html(classes="table table-bordered", float_format="{:.4f}".format)
            job_data = {
                "status": "done",
                "result": {
                    "metrics_table": metrics_html,
                    "model_paths": model_paths,
                    "best_model_name": best_model_name,
                    "results": results
                }
            }
            save_job(job_id, job_data)
        except Exception as e:
            job_data = {"status": "error", "result": str(e)}
            save_job(job_id, job_data)
    threading.Thread(target=background_train, daemon=True).start()
    return RedirectResponse(url=f"/status/{job_id}", status_code=303)

@app.get("/status/{job_id}", response_class=HTMLResponse)
def check_status(request: Request, job_id: str):
    job = load_job(job_id)
    if not job:
        return HTMLResponse("<h3>Invalid job ID.</h3>", status_code=404)
    if job["status"] == "running":
        return HTMLResponse("""
        <html><body>
        <h3>Training in progress...</h3>
        <script>
        setTimeout(function(){ window.location.reload(); }, 3000);
        </script>
        </body></html>
        """)
    elif job["status"] == "error":
        return HTMLResponse(f"<h3>Error: {job['result']}</h3>")
    else:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "metrics_table": job["result"]["metrics_table"],
            "model_paths": job["result"]["model_paths"],
            "best_model_name": job["result"]["best_model_name"],
            "results": job["result"]["results"]
        })

@app.get("/download/{job_id}/{model_name}")
def download_model(job_id: str, model_name: str):
    job = load_job(job_id)
    if not job or "model_paths" not in job["result"] or model_name not in job["result"]["model_paths"]:
        return HTMLResponse("<h3>Invalid job or model name.</h3>", status_code=404)
    model_path = job["result"]["model_paths"][model_name]
    return FileResponse(model_path, media_type='application/octet-stream', filename=model_path) 