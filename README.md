# DataScribe

**Democratizing Data Analysis: Automated EDA with Human-Readable Insights**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Overview

DataScribe is an AI-powered Exploratory Data Analysis (EDA) platform that automates the entire data analysis workflow. It transforms raw datasets into comprehensive, human-readable insights with beautiful visualizations, machine learning capabilities, and actionable recommendations.

## ✨ Features

### Core Analysis Features
- **📊 Automated EDA**: Comprehensive data analysis with configurable sections
- **🔍 Data Quality Assessment**: Missing values, duplicates, outliers detection with quality scoring
- **📈 Smart Visualizations**: Univariate, bivariate, and multivariate analysis plots
- **📋 Report Generation**: HTML, PDF, and Excel export capabilities
- **🎯 Target Analysis**: Optional target variable analysis for ML tasks
- **💡 AI Insights**: Automated recommendations and data quality scoring

### Machine Learning Capabilities
- **🤖 ML Model Training**: Train multiple ML models with auto-selection
- **📊 Model Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- **📈 Confusion Matrix**: Visual model performance assessment
- **🔄 Cross-Validation**: K-fold validation with mean ± std scores
- **🎯 Feature Importance**: Top features and coefficients analysis
- **📋 Model Comparison**: Auto-select best performing model from:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - SVM (Support Vector Machine)
  - KNN (K-Nearest Neighbors)

### User Experience
- **🎨 Modern UI/UX**: Professional, responsive design with animations
- **📱 Mobile-Friendly**: Optimized for all device sizes
- **⚡ Fast Processing**: Optimized engine for quick analysis
- **📊 Interactive Results**: Rich visualizations and detailed metrics
- **💾 Multiple Formats**: Export in PDF, Excel, HTML, and R code

### Authentication & History (Sem 8)
- **🔐 Login & Sign Up**: User accounts with email and password (bcrypt)
- **🔒 Protected Analysis**: Only logged-in users can run analyses
- **📜 My Analyses**: Per-user history of past analyses (dataset, target, model, accuracy)
- **🗄️ Database**: SQLite for local development; PostgreSQL for deployment (optional)

## 🏗️ Project Structure

```
datascribe/
├── 📁 core/                    # Core EDA engine and analysis
├── 📁 web/                     # Web application and API
├── 📁 reports/                 # Report generation and templates
├── 📁 utils/                   # Utility functions and helpers
├── 📁 tests/                   # Test suite and examples
├── 📁 docs/                    # Documentation and guides
├── 📁 examples/                # Sample datasets and demos
└── 📁 deployment/              # Deployment configurations
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ANUSHREE1403/DataScribe.git
   cd DataScribe
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python run.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8000`

## 📖 Usage

### Web Interface
1. **Sign up** or **log in** (required to run analyses)
2. Upload your dataset (CSV, Excel, Parquet)
3. Configure analysis options:
   - Target column (optional, for supervised ML)
   - Visualization preferences
   - ML model training (optional)
   - Model selection (Auto, Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, KNN)
4. Run the analysis
5. View comprehensive results including:
   - Data quality assessment
   - Statistical summaries
   - Interactive visualizations
   - ML model performance metrics
   - Confusion matrix
   - Feature importance
6. Download reports in multiple formats or open **My Analyses** to see past runs

### API Usage
The `/analyze` endpoint requires a logged-in session. Use a session (cookies) after logging in via `/login`.

```python
import requests

# Create session and log in first
session = requests.Session()
session.post('http://localhost:8000/login', data={'email': 'your@email.com', 'password': 'yourpassword'})

# Upload and analyze dataset with ML training
files = {'file': open('dataset.csv', 'rb')}
data = {
    'target_column': 'target',
    'include_plots': True,
    'include_reports': True,
    'train_model': True,
    'model_choice': 'auto'  # or specific model like 'rf', 'xgboost', etc.
}

response = session.post('http://localhost:8000/analyze', files=files, data=data, allow_redirects=True)
# On success you are redirected to /results/{job_id}; extract job_id from the final URL if needed
job_id = response.url.split('/results/')[-1].split('?')[0] if '/results/' in response.url else None

# Get analysis results (use same session)
results = session.get(f'http://localhost:8000/results/{job_id}')

# Download reports (use same session)
pdf_report = session.get(f'http://localhost:8000/download/{job_id}/report/pdf')
excel_report = session.get(f'http://localhost:8000/download/{job_id}/report/excel')
html_report = session.get(f'http://localhost:8000/download/{job_id}/report/html')
r_code = session.get(f'http://localhost:8000/download/{job_id}/code/r')
```

## 🔧 Configuration

Create a `.env` file in the root directory (optional; defaults work for local use):

```env
# App Configuration
APP_NAME=DataScribe
DEBUG=False

# Authentication (required for production; use a long random string)
SECRET_KEY=your-secret-key-change-in-production

# Database
# Local: uses SQLite (datascribe.db) by default
# Production: set DATABASE_URL to your PostgreSQL connection string
# DATABASE_URL=postgresql://user:password@host:5432/dbname

# File Storage
UPLOAD_DIR=uploads
REPORTS_DIR=reports
MAX_FILE_SIZE=100MB

# EDA Settings
MAX_ROWS_FOR_ANALYSIS=100000
CORRELATION_THRESHOLD=0.7
```

## 📊 Supported Data Formats

- **CSV** (.csv)
- **Excel** (.xlsx, .xls)
- **Parquet** (.parquet)

Example datasets and suggested target columns (Titanic, Ecommerce, Sales) are listed in [tests/dataset_targets.md](tests/dataset_targets.md).

## 🎨 Visualization Features

- **Dataset Overview**: Data types, missing values, column distributions
- **Data Quality**: Missing values heatmap, quality scoring
- **Univariate Analysis**: Histograms, distributions, outlier detection
- **Bivariate Analysis**: Correlation matrices, feature relationships
- **Multivariate Analysis**: PCA, feature importance
- **Target Analysis**: Classification/regression target insights
- **ML Results**: Confusion matrix, feature importance plots, model performance charts

## 📈 Report Types

### HTML Reports
- Interactive, responsive design
- Embedded visualizations
- Professional styling
- Mobile-friendly

### PDF Reports
- Print-ready format
- High-quality graphics
- Professional appearance
- Easy sharing

### Excel Reports
- Multiple worksheets
- Structured data
- Formatted tables
- Business-friendly

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

## 📚 Documentation

- [User Guide](docs/user-guide.md)
- [API Reference](docs/api-reference.md)
- [Developer Guide](docs/developer-guide.md)
- [Examples](examples/)

## 🚀 Deployment

### Local Development
```bash
python run.py
```
Uses SQLite (`datascribe.db`) by default. No database setup required.

### Render / Railway (Web Service)
1. Connect your GitHub repo to Render or Railway.
2. Set **Build Command**: `pip install -r requirements.txt` (or leave default).
3. Set **Start Command**: `python run.py`.
4. Add **Environment Variables**:
   - **SECRET_KEY**: A long random string for session cookies (e.g. generate with `python -c "import secrets; print(secrets.token_hex(32))"`).
   - **DATABASE_URL** (optional): Your PostgreSQL connection URL if you added a Postgres database. If not set, the app uses SQLite on the server (data may not persist across restarts on free tier).
5. Deploy; new pushes to your branch will trigger a new deploy.

### Production (self-hosted)
```bash
gunicorn web.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker
```bash
docker build -t datascribe .
docker run -p 8000:8000 datascribe
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Visualization powered by [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
- Data analysis with [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ANUSHREE1403/DataScribe/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ANUSHREE1403/DataScribe/discussions)
- **Email**: workanushree14@gmail.com

## 🔮 Roadmap

- [x] Core EDA Engine
- [x] Visualization System
- [x] Report Generation
- [x] Web Interface
- [x] Machine Learning Integration
- [x] Modern UI/UX Design
- [x] Multiple ML Models (LR, RF, XGBoost, LightGBM, SVM, KNN)
- [x] Comprehensive ML Metrics
- [x] Professional PDF Reports
- [x] User Authentication (login / sign up)
- [x] History Dashboard (My Analyses)
- [x] Database (SQLite + PostgreSQL support)
- [x] Cloud Deployment (Render / Railway)
- [ ] Feedback System
- [ ] Mobile App

---

**Made with ❤️ by the DataScribe Team**

*Democratizing data analysis, one dataset at a time.*
