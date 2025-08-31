from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import uuid
import json
from typing import Optional, List
import shutil
from datetime import datetime

# Import DataScribe components
try:
    from core.eda_engine import run_eda
    from core.visualization_engine import generate_visualizations
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Core components not available: {e}")
    CORE_AVAILABLE = False

from utils.config import settings

# Report generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import openpyxl
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

app = FastAPI(
    title=settings.app_name,
    description=settings.app_subtitle,
    version=settings.app_version
)

# Setup templates and static files
templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.reports_dir, exist_ok=True)
os.makedirs("static", exist_ok=True)

# Store analysis jobs
jobs = {}

def load_dataset(file: UploadFile):
    """Load dataset from uploaded file"""
    try:
        import pandas as pd
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.file)
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(file.file)
        else:
            raise ValueError("Unsupported file format")
        
        # Basic data cleaning
        df = df.dropna(how='all')  # Remove completely empty rows
        df = df.dropna(axis=1, how='all')  # Remove completely empty columns
        
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading dataset: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with upload form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    include_plots: bool = Form(True)
):
    """Analyze uploaded dataset"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    if not any(file.filename.endswith(ext) for ext in settings.allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Allowed: {', '.join(settings.allowed_extensions)}")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    try:
        # Load dataset
        df = load_dataset(file)
        
        # Store dataset
        dataset_path = os.path.join(settings.upload_dir, f"{job_id}.csv")
        df.to_csv(dataset_path, index=False)
        
        # Store original filename
        original_filename = file.filename
        
        # Check if core components are available
        if not CORE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Core analysis components not available. Please check dependencies.")
        
        # Run EDA analysis
        print(f"Starting EDA analysis for job {job_id}")
        analysis_results = run_eda(df, target_column)
        print(f"EDA analysis completed. Results keys: {list(analysis_results.keys())}")
        
        # Generate visualizations if requested
        plot_files = {}
        if include_plots and CORE_AVAILABLE:
            print(f"Generating visualizations for job {job_id}")
            # Ensure static directory exists
            os.makedirs("static", exist_ok=True)
            
            plot_files = generate_visualizations(df, analysis_results, target_column)
            print(f"Visualizations generated: {list(plot_files.keys())}")
            print(f"Plot files content: {plot_files}")
            
            # Move plots to static directory and update URLs
            for plot_type, plot_file in plot_files.items():
                print(f"Processing {plot_type}: {plot_file}")
                if plot_file and plot_file != "no_target_data.png":
                    # Check if the file exists in current directory
                    if os.path.exists(plot_file):
                        new_path = os.path.join("static", f"{job_id}_{plot_type}.png")
                        shutil.move(plot_file, new_path)
                        plot_files[plot_type] = f"/static/{job_id}_{plot_type}.png"
                        print(f"Moved {plot_type} plot to {new_path}")
                    else:
                        print(f"Warning: Plot file {plot_file} not found for {plot_type}")
                        plot_files[plot_type] = None
                else:
                    print(f"Skipping {plot_type} plot (no data or placeholder)")
                    plot_files[plot_type] = None
        elif include_plots:
            print("Visualizations requested but core components not available")
            plot_files = {}
        
        # Store job results with both file paths and URLs for visualizations
        visualization_paths = {}
        for plot_type, plot_file in plot_files.items():
            if plot_file and plot_file.startswith("/static/"):
                # Extract the actual file path from the URL
                file_name = plot_file.split("/")[-1]
                actual_path = os.path.join("static", file_name)
                visualization_paths[plot_type] = actual_path  # Store actual file path
                print(f"Stored visualization path for {plot_type}: {actual_path}")
            elif plot_file is None:
                print(f"Skipping {plot_type} - no visualization generated")
            else:
                print(f"Warning: Unexpected plot_file value for {plot_type}: {plot_file}")
        
        jobs[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "dataset_path": dataset_path,
            "analysis_results": analysis_results,
            "visualizations": visualization_paths,  # Store file paths for PDF generation
            "visualization_urls": plot_files,  # Store URLs for web display
            "target_column": target_column,
            "include_plots": include_plots,
            "filename": original_filename,
            "created_at": datetime.now().isoformat()
        }
        
        # Debug: Print what we stored
        print(f"Job {job_id} stored with:")
        print(f"  - Visualizations: {visualization_paths}")
        print(f"  - Visualization URLs: {plot_files}")
        print(f"  - Include plots: {include_plots}")
        print(f"  - Plot files type: {type(plot_files)}")
        print(f"  - Plot files keys: {list(plot_files.keys()) if plot_files else 'None'}")
        
        # Redirect to results page
        return RedirectResponse(url=f"/results/{job_id}", status_code=303)
        
    except Exception as e:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "created_at": datetime.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def generate_python_code(job: dict, job_id: str) -> str:
    """Generate Python code for the analysis"""
    try:
        analysis_results = job.get("analysis_results", {})
        code_path = os.path.join(settings.reports_dir, f"analysis_code_{job_id}.py")
        
        # Get current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = job.get("filename", "Unknown")
        
        code_content = f'''# DataScribe Analysis Code
# Generated on: {current_time}
# Dataset: {filename}

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set style for better plots
plt.style.use('default')

# Try to use seaborn if available
try:
    import seaborn as sns
    sns.set_palette("husl")
    print("Using seaborn for enhanced plots")
except ImportError:
    print("Using matplotlib default style")

# Load your dataset
# df = pd.read_csv('your_dataset.csv')  # Replace with your file path
# df = pd.read_excel('your_dataset.xlsx')  # For Excel files
# df = pd.read_parquet('your_dataset.parquet')  # For Parquet files

# IMPORTANT: Uncomment and modify one of the lines above to load your actual dataset
# Example: df = pd.read_csv('your_dataset.csv')

# For demonstration purposes, create a sample dataset
# Remove this section when you load your actual data
print("Creating sample dataset for demonstration...")
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({{
    'feature_1': np.random.normal(0, 1, n_samples),
    'feature_2': np.random.normal(0, 1, n_samples),
    'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
    'target': np.random.choice([0, 1], n_samples)
}})
print("Sample dataset created successfully!")
print()

# Basic dataset info
print("Dataset Shape:", df.shape)
print("Memory Usage:", df.memory_usage(deep=True).sum() / 1024**2, "MB")
print("\\nData Types:")
print(df.dtypes)

# Data Quality Assessment
print("\\n=== Data Quality Assessment ===")
print("Missing Values:")
print(df.isnull().sum())

print("\\nDuplicate Rows:", df.duplicated().sum())

# Statistical Summary
print("\\n=== Statistical Summary ===")
print(df.describe())

# Column Analysis
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\\nNumerical Columns: {len(numerical_cols)}")
print(f"Categorical Columns: {len(categorical_cols)}")

# Visualizations
if len(numerical_cols) > 0:
    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numerical_cols].corr()
    
    # Try to use seaborn for heatmap, fallback to matplotlib
    try:
        import seaborn as sns
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    except ImportError:
        # Fallback to matplotlib
        plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        
        # Add correlation values as text
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')
    
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Distribution plots for numerical columns
    for col in numerical_cols[:5]:  # Limit to first 5 columns
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
        plt.subplot(2, 1, 2)
        plt.boxplot(df[col].dropna())
        plt.title(f'Box Plot of {col}')
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

if len(categorical_cols) > 0:
    # Bar plots for categorical columns
    for col in categorical_cols[:3]:  # Limit to first 3 columns
        plt.figure(figsize=(10, 6))
        value_counts = df[col].value_counts()
        value_counts.plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Data Cleaning Recommendations
print("\\n=== Data Cleaning Recommendations ===")
for col in df.columns:
    missing_pct = df[col].isnull().sum() / len(df) * 100
    if missing_pct > 0:
        print(f"{col}: {missing_pct:.1f}% missing values")
    
    if df[col].dtype == 'object':
        unique_vals = df[col].nunique()
        if unique_vals == 1:
            print(f"{col}: Constant column (only one unique value)")
        elif unique_vals == len(df):
            print(f"{col}: High cardinality (all values unique)")

print("\\n=== Analysis Complete ===")
print("This code provides a basic EDA framework.")
print("Customize it based on your specific dataset and analysis goals.")
'''
        
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
        
        return None
        
    except Exception as e:
        print(f"Python code generation disabled")
        return None

def generate_r_code(job: dict, job_id: str) -> str:
    """Generate R code for the analysis"""
    try:
        analysis_results = job.get("analysis_results", {})
        code_path = os.path.join(settings.reports_dir, f"analysis_code_{job_id}.R")
        
        code_content = f'''# DataScribe Analysis Code (R)
# Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Dataset: {job.get("filename", "Unknown")}

# Load required libraries
library(ggplot2)
library(dplyr)
library(corrplot)
library(gridExtra)

# Load your dataset
# df <- read.csv('your_dataset.csv')  # Replace with your file path
# df <- read_excel('your_dataset.xlsx')  # For Excel files (requires readxl package)

# IMPORTANT: Uncomment and modify one of the lines above to load your actual dataset
# Example: df <- read.csv('your_dataset.csv')

# For demonstration purposes, create a sample dataset
# Remove this section when you load your actual data
cat("Creating sample dataset for demonstration...\\n")
set.seed(42)
n_samples <- 1000
df <- data.frame(
    feature_1 = rnorm(n_samples, 0, 1),
    feature_2 = rnorm(n_samples, 0, 1),
    feature_3 = sample(c('A', 'B', 'C'), n_samples, replace = TRUE),
    target = sample(c(0, 1), n_samples, replace = TRUE)
)
cat("Sample dataset created successfully!\\n\\n")

# Basic dataset info
cat("Dataset Shape:", nrow(df), "rows x", ncol(df), "columns\\n")
cat("Memory Usage:", object.size(df) / 1024^2, "MB\\n")
cat("\\nData Types:\\n")
str(df)

# Data Quality Assessment
cat("\\n=== Data Quality Assessment ===\\n")
cat("Missing Values:\\n")
print(colSums(is.na(df)))

cat("\\nDuplicate Rows:", sum(duplicated(df)), "\\n")

# Statistical Summary
cat("\\n=== Statistical Summary ===\\n")
print(summary(df))

# Column Analysis
numerical_cols <- sapply(df, is.numeric)
categorical_cols <- sapply(df, is.character) | sapply(df, is.factor)

cat("\\nNumerical Columns:", sum(numerical_cols), "\\n")
cat("Categorical Columns:", sum(categorical_cols), "\\n")

# Visualizations
if(sum(numerical_cols) > 0) {{
    # Correlation Matrix
    cor_matrix <- cor(df[, numerical_cols], use="complete.obs")
    corrplot(cor_matrix, method="color", type="upper", 
             order="hclust", tl.cex=0.7, tl.col="black")
    
    # Distribution plots for numerical columns
    num_cols <- names(df)[numerical_cols]
    for(col in head(num_cols, 5)) {{  # Limit to first 5 columns
        p1 <- ggplot(df, aes_string(x=col)) +
               geom_histogram(fill="skyblue", alpha=0.7, bins=30) +
               labs(title=paste("Distribution of", col)) +
               theme_minimal()
        
        p2 <- ggplot(df, aes_string(y=col)) +
               geom_boxplot(fill="lightgreen", alpha=0.7) +
               labs(title=paste("Box Plot of", col)) +
               theme_minimal()
        
        grid.arrange(p1, p2, ncol=2)
    }}
}}

if(sum(categorical_cols) > 0) {{
    # Bar plots for categorical columns
    cat_cols <- names(df)[categorical_cols]
    for(col in head(cat_cols, 3)) {{  # Limit to first 3 columns
        p <- ggplot(df, aes_string(x=col)) +
             geom_bar(fill="steelblue", alpha=0.7) +
             labs(title=paste("Distribution of", col)) +
             theme_minimal() +
             theme(axis.text.x = element_text(angle = 45, hjust = 1))
        print(p)
    }}
}}

# Data Cleaning Recommendations
cat("\\n=== Data Cleaning Recommendations ===\\n")
for(col in names(df)) {{
    missing_pct <- sum(is.na(df[[col]])) / nrow(df) * 100
    if(missing_pct > 0) {{
        cat(col, ":", round(missing_pct, 1), "% missing values\\n")
    }}
    
    if(is.character(df[[col]]) || is.factor(df[[col]])) {{
        unique_vals <- length(unique(df[[col]]))
        if(unique_vals == 1) {{
            cat(col, ": Constant column (only one unique value)\\n")
        }} else if(unique_vals == nrow(df)) {{
            cat(col, ": High cardinality (all values unique)\\n")
        }}
    }}
}}

cat("\\n=== Analysis Complete ===\\n")
cat("This code provides a basic EDA framework in R.\\n")
cat("Customize it based on your specific dataset and analysis goals.\\n")
'''
        
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
        
        return code_path
        
    except Exception as e:
        print(f"Error generating R code: {e}")
        return None

def generate_html_report(job: dict) -> str:
    """Generate HTML report content"""
    analysis_results = job.get("analysis_results", {})
    
    # Ensure reports directory exists
    os.makedirs(settings.reports_dir, exist_ok=True)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DataScribe Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; color: #667eea; }}
            .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
            .metric-label {{ color: #666; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 DataScribe Analysis Report</h1>
            <p>Generated on: {job.get('created_at', 'Unknown')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Dataset Overview</h2>
            <div class="metric">
                <div class="metric-value">{analysis_results.get('overview', {}).get('shape', [0, 0])[0]}</div>
                <div class="metric-label">Rows</div>
            </div>
            <div class="metric">
                <div class="metric-value">{analysis_results.get('overview', {}).get('shape', [0, 0])[1]}</div>
                <div class="metric-label">Columns</div>
            </div>
            <div class="metric">
                <div class="metric-value">{analysis_results.get('overview', {}).get('memory_usage', 0):.2f} MB</div>
                <div class="metric-label">Memory Usage</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔍 Data Quality Assessment</h2>
            <div class="metric">
                <div class="metric-value">{analysis_results.get('data_quality', {}).get('data_quality_score', 0):.1f}%</div>
                <div class="metric-label">Quality Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{analysis_results.get('data_quality', {}).get('missing_values', {}).get('total_missing', 0)}</div>
                <div class="metric-label">Missing Values</div>
            </div>
            <div class="metric">
                <div class="metric-value">{analysis_results.get('data_quality', {}).get('duplicates', {}).get('count', 0)}</div>
                <div class="metric-label">Duplicate Rows</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Column Analysis</h2>
            <div class="metric">
                <div class="metric-value">{analysis_results.get('overview', {}).get('columns', {}).get('numerical', 0)}</div>
                <div class="metric-label">Numerical Columns</div>
            </div>
            <div class="metric">
                <div class="metric-value">{analysis_results.get('overview', {}).get('columns', {}).get('categorical', 0)}</div>
                <div class="metric-label">Categorical Columns</div>
            </div>
            <div class="metric">
                <div class="metric-value">{analysis_results.get('overview', {}).get('columns', {}).get('datetime', 0)}</div>
                <div class="metric-label">Datetime Columns</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Visualizations</h2>
    """
    
    # Add visualizations if available
    visualization_urls = job.get("visualization_urls", {})
    print(f"HTML Report - Visualization URLs: {visualization_urls}")
    print(f"HTML Report - Job keys: {list(job.keys())}")
    
    if visualization_urls:
        for viz_name, viz_url in visualization_urls.items():
            print(f"HTML Report - Processing {viz_name}: {viz_url}")
            if viz_url:
                # Convert relative paths to static URLs
                if not viz_url.startswith("/static/"):
                    viz_url = f"/static/{viz_url}"
                
                html_content += f"""
                <div style="margin: 20px 0; text-align: center;">
                    <h3>{viz_name.replace('_', ' ').title()}</h3>
                    <img src="{viz_url}" alt="{viz_name} visualization" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px;">
                </div>
                """
    else:
        html_content += "<p>No visualizations were generated for this analysis.</p>"
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>🤖 AI Insights</h2>
            <ul>
    """
    
    insights = analysis_results.get('insights', {})
    for insight_type, insight_list in insights.items():
        if insight_list:
            html_content += f"<h3>{insight_type.replace('_', ' ').title()}</h3><ul>"
            for insight in insight_list:
                html_content += f"<li>{insight}</li>"
            html_content += "</ul>"
    
    html_content += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html_content

def generate_excel_report(job: dict, job_id: str) -> str:
    """Generate Excel report"""
    if not OPENPYXL_AVAILABLE:
        return None
    
    try:
        analysis_results = job.get("analysis_results", {})
        report_path = os.path.join(settings.reports_dir, f"report_{job_id}.xlsx")
        
        wb = Workbook()
        
        # Overview sheet
        ws1 = wb.active
        ws1.title = "Dataset Overview"
        ws1['A1'] = "DataScribe Analysis Report"
        ws1['A1'].font = Font(bold=True, size=16, color="667EEA")
        ws1['A3'] = "Dataset Information"
        ws1['A3'].font = Font(bold=True, size=14)
        
        # Dataset metrics
        overview = analysis_results.get('overview', {})
        metrics_data = [
            ["Metric", "Value"],
            ["Rows", overview.get('shape', [0, 0])[0]],
            ["Columns", overview.get('shape', [0, 0])[1]],
            ["Memory Usage (MB)", f"{overview.get('memory_usage', 0):.2f}"],
            ["Numerical Columns", overview.get('columns', {}).get('numerical', 0)],
            ["Categorical Columns", overview.get('columns', {}).get('categorical', 0)],
            ["Datetime Columns", overview.get('columns', {}).get('datetime', 0)]
        ]
        
        for row in metrics_data:
            ws1.append(row)
        
        # Data Quality sheet
        ws2 = wb.create_sheet("Data Quality")
        ws2['A1'] = "Data Quality Assessment"
        ws2['A1'].font = Font(bold=True, size=16, color="667EEA")
        
        quality = analysis_results.get('data_quality', {})
        quality_data = [
            ["Metric", "Value"],
            ["Quality Score", f"{quality.get('data_quality_score', 0):.1f}%"],
            ["Missing Values", quality.get('missing_values', {}).get('total_missing', 0)],
            ["Duplicate Rows", quality.get('duplicates', {}).get('count', 0)],
            ["Constant Columns", len(quality.get('constant_columns', []))]
        ]
        
        for row in quality_data:
            ws2.append(row)
        
        # Insights sheet
        ws3 = wb.create_sheet("AI Insights")
        ws3['A1'] = "AI Insights & Recommendations"
        ws3['A1'].font = Font(bold=True, size=16, color="667EEA")
        
        insights = analysis_results.get('insights', {})
        row_num = 3
        for insight_type, insight_list in insights.items():
            if insight_list:
                ws3[f'A{row_num}'] = insight_type.replace('_', ' ').title()
                ws3[f'A{row_num}'].font = Font(bold=True, size=12)
                row_num += 1
                for insight in insight_list:
                    ws3[f'A{row_num}'] = f"• {insight}"
                    row_num += 1
                row_num += 1
        
        wb.save(report_path)
        return report_path
        
    except Exception as e:
        print(f"Error generating Excel report: {e}")
        return None

def generate_pdf_report(job: dict, job_id: str) -> str:
    """Generate PDF report"""
    if not REPORTLAB_AVAILABLE:
        print("ReportLab not available for PDF generation")
        return None
    
    try:
        print(f"Starting PDF generation for job {job_id}")
        analysis_results = job.get("analysis_results", {})
        visualizations = job.get("visualizations", {})
        report_path = os.path.join(settings.reports_dir, f"report_{job_id}.pdf")
        
        print(f"Report path: {report_path}")
        print(f"Analysis results keys: {list(analysis_results.keys())}")
        print(f"Visualizations keys: {list(visualizations.keys())}")
        
        doc = SimpleDocTemplate(report_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title Page
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#667EEA')
        )
        story.append(Paragraph("🚀 DataScribe Analysis Report", title_style))
        story.append(Spacer(1, 30))
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Generated on: {timestamp}", styles['Normal']))
        story.append(Spacer(1, 40))
        
        # Report Summary
        story.append(Paragraph("📋 Report Summary", styles['Heading3']))
        story.append(Paragraph("This report provides a comprehensive analysis of your dataset including data quality assessment, statistical summaries, and AI-powered insights.", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Table of Contents
        story.append(Paragraph("📋 Table of Contents", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        toc_items = [
            "1. Dataset Overview",
            "2. Data Quality Assessment", 
            "3. Statistical Summary",
            "4. Visualizations",
            "5. AI Insights & Recommendations"
        ]
        
        for item in toc_items:
            story.append(Paragraph(item, styles['Normal']))
            story.append(Spacer(1, 8))
        
        story.append(PageBreak())
        
        # Dataset Overview
        story.append(Paragraph("📊 Dataset Overview", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        overview = analysis_results.get('overview', {})
        
        # Create overview metrics with better formatting
        overview_data = [
            ['Dataset Metric', 'Value', 'Description'],
            ['Total Rows', f"{overview.get('shape', [0, 0])[0]:,}", 'Number of data records'],
            ['Total Columns', str(overview.get('shape', [0, 0])[1]), 'Number of features/variables'],
            ['Memory Usage', f"{overview.get('memory_usage', 0):.2f} MB", 'Storage space required'],
            ['Numerical Features', str(overview.get('columns', {}).get('numerical', 0)), 'Continuous/numeric variables'],
            ['Categorical Features', str(overview.get('columns', {}).get('categorical', 0)), 'Text/category variables'],
            ['Datetime Features', str(overview.get('columns', {}).get('datetime', 0)), 'Date/time variables']
        ]
        
        overview_table = Table(overview_data)
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17A2B8')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DEE2E6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8F9FA')])
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 15))
        
        # Add dataset insights
        story.append(Paragraph("💡 Dataset Insights", styles['Heading3']))
        total_rows = overview.get('shape', [0, 0])[0]
        total_cols = overview.get('shape', [0, 0])[1]
        
        if total_rows > 10000:
            story.append(Paragraph("• Large dataset with substantial data for robust analysis", styles['Normal']))
        elif total_rows > 1000:
            story.append(Paragraph("• Medium-sized dataset suitable for most analytical tasks", styles['Normal']))
        else:
            story.append(Paragraph("• Compact dataset - consider data augmentation for complex models", styles['Normal']))
        
        if total_cols > 50:
            story.append(Paragraph("• High-dimensional dataset - feature selection may be beneficial", styles['Normal']))
        elif total_cols > 10:
            story.append(Paragraph("• Balanced feature set for comprehensive analysis", styles['Normal']))
        else:
            story.append(Paragraph("• Low-dimensional dataset - consider feature engineering", styles['Normal']))
        
        story.append(Spacer(1, 20))
        story.append(PageBreak())
        
        # Data Quality Assessment
        story.append(Paragraph("🔍 Data Quality Assessment", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        quality = analysis_results.get('data_quality', {})
        
        # Create quality score indicator
        quality_score = quality.get('data_quality_score', 0)
        quality_color = colors.HexColor('#28A745') if quality_score >= 80 else colors.HexColor('#FFC107') if quality_score >= 60 else colors.HexColor('#DC3545')
        quality_status = "Excellent" if quality_score >= 80 else "Good" if quality_score >= 60 else "Needs Improvement"
        
        # Quality overview table
        quality_overview = [
            ['Quality Metric', 'Value', 'Status'],
            ['Overall Quality Score', f"{quality_score:.1f}%", quality_status],
            ['Missing Values', str(quality.get('missing_values', {}).get('total_missing', 0)), '✓' if quality.get('missing_values', {}).get('total_missing', 0) == 0 else '⚠'],
            ['Duplicate Rows', str(quality.get('duplicates', {}).get('count', 0)), '⚠' if quality.get('duplicates', {}).get('count', 0) > 0 else '✓'],
            ['Constant Columns', str(len(quality.get('constant_columns', []))), '⚠' if len(quality.get('constant_columns', [])) > 0 else '✓']
        ]
        
        quality_table = Table(quality_overview)
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#495057')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DEE2E6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8F9FA')])
        ]))
        story.append(quality_table)
        story.append(Spacer(1, 15))
        
        # Add quality insights
        story.append(Paragraph("💡 Quality Insights", styles['Heading3']))
        if quality_score >= 80:
            story.append(Paragraph("• Your dataset has excellent data quality!", styles['Normal']))
        elif quality_score >= 60:
            story.append(Paragraph("• Your dataset has good data quality with some areas for improvement.", styles['Normal']))
        else:
            story.append(Paragraph("• Your dataset needs attention to improve data quality.", styles['Normal']))
        
        # Add specific recommendations
        if quality.get('missing_values', {}).get('total_missing', 0) > 0:
            story.append(Paragraph(f"• Consider handling {quality.get('missing_values', {}).get('total_missing', 0)} missing values", styles['Normal']))
        if quality.get('duplicates', {}).get('count', 0) > 0:
            story.append(Paragraph(f"• {quality.get('duplicates', {}).get('count', 0)} duplicate rows detected - review for data integrity", styles['Normal']))
        if len(quality.get('constant_columns', [])) > 0:
            story.append(Paragraph(f"• {len(quality.get('constant_columns', []))} constant columns detected - consider removal", styles['Normal']))
        
        story.append(Spacer(1, 20))
        story.append(PageBreak())
        
        # Statistical Summary
        story.append(Paragraph("📈 Statistical Summary", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        stats = analysis_results.get('statistics', {})
        if stats:
            # Create a comprehensive summary table
            summary_headers = ['Column', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
            summary_data = [summary_headers]
            
            for col_name, col_stats in stats.items():
                if isinstance(col_stats, dict) and col_stats:
                    row = [col_name]
                    # Add key statistics in order
                    for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                        value = col_stats.get(stat, 'N/A')
                        if isinstance(value, (int, float)):
                            if stat in ['mean', 'std']:
                                row.append(f"{value:.3f}")
                            elif stat in ['25%', '50%', '75%']:
                                row.append(f"{value:.2f}")
                            else:
                                row.append(str(value))
                        else:
                            row.append(str(value))
                    summary_data.append(row)
            
            if len(summary_data) > 1:  # Only add table if we have data
                summary_table = Table(summary_data)
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F7F7F7')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F0F0F0')])
                ]))
                story.append(summary_table)
                story.append(Spacer(1, 15))
                
                # Add summary statistics
                story.append(Paragraph("📊 Summary Statistics", styles['Heading3']))
                story.append(Paragraph(f"• Total numerical columns analyzed: {len(summary_data) - 1}", styles['Normal']))
                story.append(Paragraph(f"• Data points per column: {summary_data[1][1] if len(summary_data) > 1 else 'N/A'}", styles['Normal']))
                story.append(Spacer(1, 10))
        
        story.append(Spacer(1, 20))
        story.append(PageBreak())
        
        # Visualizations Section
        story.append(Paragraph("📊 Visualizations", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Visualizations Overview
        if visualizations:
            story.append(Paragraph("📊 Generated Visualizations", styles['Heading3']))
            story.append(Spacer(1, 8))
            
            # Create a structured list of visualizations
            viz_list = []
            for viz_name in visualizations.keys():
                viz_list.append(viz_name.replace('_', ' ').title())
            
            # Display visualizations in a more organized way
            for i, viz_name in enumerate(viz_list, 1):
                story.append(Paragraph(f"{i}. {viz_name}", styles['Normal']))
            
            story.append(Spacer(1, 15))
            story.append(Paragraph("Below are the detailed visualizations for each analysis type:", styles['Normal']))
            story.append(Spacer(1, 10))
        else:
            story.append(Paragraph("No visualizations were generated for this analysis.", styles['Normal']))
            story.append(Paragraph("This may be due to insufficient data or analysis configuration.", styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Add visualization images if they exist
        if visualizations:
            for i, (viz_name, viz_path) in enumerate(visualizations.items()):
                if viz_path and os.path.exists(viz_path):
                    try:
                        # Add visualization title with better formatting
                        story.append(Paragraph(f"📊 {viz_name.replace('_', ' ').title()}", styles['Heading3']))
                        story.append(Spacer(1, 8))
                        
                        # Add the image
                        img = Image(viz_path)
                        img.drawHeight = 4 * inch
                        img.drawWidth = 6 * inch
                        story.append(img)
                        story.append(Spacer(1, 15))
                        
                        # Add page break between visualizations for better organization
                        if i < len(visualizations) - 1:  # Don't add page break after the last visualization
                            story.append(PageBreak())
                            
                    except Exception as e:
                        print(f"Error adding visualization {viz_name}: {e}")
                        story.append(Paragraph(f"Visualization: {viz_name} (could not display)", styles['Normal']))
                        story.append(Spacer(1, 10))
                else:
                    print(f"Visualization path not found or invalid: {viz_name} -> {viz_path}")
                    story.append(Paragraph(f"Visualization: {viz_name} (file not found)", styles['Normal']))
                    story.append(Spacer(1, 10))
        
        story.append(PageBreak())
        
        # AI Insights & Recommendations
        story.append(Paragraph("🤖 AI Insights & Recommendations", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        insights = analysis_results.get('insights', {})
        if insights:
            for insight_type, insight_list in insights.items():
                if insight_list:
                    # Create a styled insight section
                    story.append(Paragraph(f"📌 {insight_type.replace('_', ' ').title()}", styles['Heading3']))
                    story.append(Spacer(1, 6))
                    
                    # Add insights with better formatting
                    for i, insight in enumerate(insight_list, 1):
                        story.append(Paragraph(f"{i}. {insight}", styles['Normal']))
                    
                    story.append(Spacer(1, 10))
        else:
            story.append(Paragraph("No specific insights available for this dataset.", styles['Normal']))
            story.append(Paragraph("Consider running additional analysis or providing a target variable for more detailed recommendations.", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Build PDF
        print("Building PDF document...")
        doc.build(story)
        print(f"PDF generated successfully at: {report_path}")
        return report_path
        
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.get("/results/{job_id}")
async def get_results(request: Request, job_id: str):
    """Get analysis results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    print(f"Results requested for job {job_id}")
    print(f"Job status: {job['status']}")
    print(f"Job keys: {list(job.keys())}")
    if 'analysis_results' in job:
        print(f"Analysis results keys: {list(job['analysis_results'].keys())}")
    
    if job["status"] == "failed":
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": job.get("error", "Unknown error")
        })
    
    # Debug: Print what's being passed to template
    print(f"Template data - Job keys: {list(job.keys())}")
    print(f"Template data - Visualization URLs: {job.get('visualization_urls', 'NOT_FOUND')}")
    print(f"Template data - Visualizations: {job.get('visualizations', 'NOT_FOUND')}")
    
    return templates.TemplateResponse("results.html", {
        "request": request,
        "job": job,
        "analysis_results": job["analysis_results"]
    })



@app.get("/download/{job_id}/dataset")
async def download_dataset(job_id: str):
    """Download the analyzed dataset"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    dataset_path = job.get("dataset_path")
    
    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    return FileResponse(
        dataset_path,
        media_type='text/csv',
        filename=f"dataset_{job_id}.csv"
    )

@app.get("/download/{job_id}/report/html")
async def download_html_report(job_id: str):
    """Download HTML analysis report"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Generate HTML report
    html_content = generate_html_report(job)
    
    # Save to temporary file
    report_path = os.path.join(settings.reports_dir, f"report_{job_id}.html")
    os.makedirs(settings.reports_dir, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return FileResponse(
        report_path,
        media_type='text/html',
        filename=f"DataScribe_Report_{job_id}.html"
    )

@app.get("/download/{job_id}/report/excel")
async def download_excel_report(job_id: str):
    """Download Excel analysis report"""
    if not OPENPYXL_AVAILABLE:
        raise HTTPException(
            status_code=500, 
            detail="Excel generation not available. Please install openpyxl: pip install openpyxl"
        )
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Generate Excel report
    excel_path = generate_excel_report(job, job_id)
    
    if not excel_path or not os.path.exists(excel_path):
        raise HTTPException(status_code=500, detail="Failed to generate Excel report")
    
    return FileResponse(
        excel_path,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        filename=f"DataScribe_Report_{job_id}.xlsx"
    )

@app.get("/download/{job_id}/report/pdf")
async def download_pdf_report(job_id: str):
    """Download PDF analysis report"""
    print(f"PDF download requested for job: {job_id}")
    
    if not REPORTLAB_AVAILABLE:
        print("ReportLab not available")
        raise HTTPException(
            status_code=500, 
            detail="PDF generation not available. Please install reportlab: pip install reportlab"
        )
    
    if job_id not in jobs:
        print(f"Job {job_id} not found in jobs")
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    print(f"Job found: {job.get('status', 'unknown')}")
    print(f"Job keys: {list(job.keys())}")
    
    # Generate PDF report
    print("Calling generate_pdf_report...")
    pdf_path = generate_pdf_report(job, job_id)
    print(f"PDF path returned: {pdf_path}")
    
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"PDF file not found at: {pdf_path}")
        raise HTTPException(status_code=500, detail="Failed to generate PDF report")
    
    print(f"PDF file exists, returning FileResponse")
    return FileResponse(
        pdf_path,
        media_type='application/pdf',
        filename=f"DataScribe_Report_{job_id}.pdf"
    )

# Python code download temporarily disabled
# @app.get("/download/{job_id}/code/python")
# async def download_python_code(job_id: str):
#     """Download Python analysis code - TEMPORARILY DISABLED"""
#     raise HTTPException(status_code=503, detail="Python code generation temporarily disabled")

@app.get("/download/{job_id}/code/r")
async def download_r_code(job_id: str):
    """Download R analysis code"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Generate R code
    code_path = generate_r_code(job, job_id)
    
    if not code_path or not os.path.exists(code_path):
        raise HTTPException(status_code=500, detail="Failed to generate R code")
    
    return FileResponse(
        code_path,
        media_type='text/plain',
        filename=f"DataScribe_R_Code_{job_id}.R"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "active_jobs": len([j for j in jobs.values() if j["status"] == "completed"]),
        "report_generation": {
            "pdf": REPORTLAB_AVAILABLE,
            "excel": OPENPYXL_AVAILABLE,
            "html": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    ) 