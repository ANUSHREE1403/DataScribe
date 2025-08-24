# DataScribe

**Democratizing Data Analysis: Automated EDA with Human-Readable Insights**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Overview

DataScribe is an AI-powered Exploratory Data Analysis (EDA) platform that automates the entire data analysis workflow. It transforms raw datasets into comprehensive, human-readable insights with beautiful visualizations and actionable recommendations.

## ✨ Features

### Version 1 MVP (Current)
- **📊 Automated EDA**: Comprehensive data analysis with configurable sections
- **🔍 Data Quality Assessment**: Missing values, duplicates, outliers detection
- **📈 Smart Visualizations**: Univariate, bivariate, and multivariate analysis plots
- **📋 Report Generation**: HTML, PDF, and Excel export capabilities
- **🎯 Target Analysis**: Optional target variable analysis for ML tasks
- **💡 AI Insights**: Automated recommendations and data quality scoring

### Planned Features (Version 2)
- **🔐 User Authentication**: Login/signup with OAuth support
- **💾 History Dashboard**: Save and retrieve past analyses
- **📝 Feedback System**: Rate and comment on analysis quality
- **🌐 Cloud Deployment**: Multi-user access and collaboration
- **🔧 Code Export**: Python/R scripts reproducing analysis steps

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
   git clone https://github.com/yourusername/datascribe.git
   cd datascribe
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python web/main.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8000`

## 📖 Usage

### Web Interface
1. Upload your dataset (CSV, Excel, Parquet)
2. Configure analysis options (target column, visualization preferences)
3. Run the analysis
4. View results and download reports

### API Usage
```python
import requests

# Upload and analyze dataset
files = {'file': open('dataset.csv', 'rb')}
data = {
    'target_column': 'target',
    'include_plots': True,
    'include_code': False
}

response = requests.post('http://localhost:8000/analyze', 
                       files=files, data=data)
result = response.json()

# Get analysis results
job_id = result['job_id']
results = requests.get(f'http://localhost:8000/api/results/{job_id}').json()
```

## 🔧 Configuration

Create a `.env` file in the root directory:

```env
# App Configuration
APP_NAME=DataScribe
DEBUG=False

# Database (for future versions)
DATABASE_URL=postgresql://user:password@localhost/datascribe

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

## 🎨 Visualization Features

- **Dataset Overview**: Data types, missing values, column distributions
- **Data Quality**: Missing values heatmap, quality scoring
- **Univariate Analysis**: Histograms, distributions, outlier detection
- **Bivariate Analysis**: Correlation matrices, feature relationships
- **Multivariate Analysis**: PCA, feature importance
- **Target Analysis**: Classification/regression target insights

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
python web/main.py
```

### Production
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

- **Issues**: [GitHub Issues](https://github.com/yourusername/datascribe/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/datascribe/discussions)
- **Email**: workanushree14@gmail.com

## 🔮 Roadmap

- [x] Core EDA Engine
- [x] Visualization System
- [x] Report Generation
- [x] Web Interface
- [ ] User Authentication
- [ ] History Dashboard
- [ ] Feedback System
- [ ] Cloud Deployment
- [ ] Mobile App

---

**Made with ❤️ by the DataScribe Team**

*Democratizing data analysis, one dataset at a time.*
