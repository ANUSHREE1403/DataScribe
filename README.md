# Auto Genie 🧞‍♂️

**Your Personal AutoML Assistant** - A powerful Python tool for automated machine learning with both CLI and web interfaces.

## 🚀 Features

### Core Capabilities
- **📊 Automatic EDA (Exploratory Data Analysis)**: Shape, missing values, data types, correlations, target distribution
- **🔧 Smart Preprocessing**: Missing value imputation, one-hot encoding, feature scaling
- **🤖 Multi-Model Training**: Logistic/Linear Regression, RandomForest, XGBoost, KNN, SVM, LightGBM
- **📈 Model Evaluation**: Comprehensive classification and regression metrics
- **🏆 Model Selection**: Automatic best model selection with comparison tables
- **💾 Model Persistence**: Save trained models as .pkl files

### Dual Interface
- **🖥️ Web Interface**: User-friendly web UI with real-time training progress
- **💻 CLI Interface**: Command-line tool for automation and scripting

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ANUSHREE1403/Auto-Genie.git
   cd Auto-Genie/auto-genie
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Web Interface (Recommended)

1. **Start the web server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8000`

3. **Upload your CSV file** and configure:
   - Target column selection
   - Task type (classification/regression)
   - Test size
   - Model selection

4. **Monitor training progress** in real-time

5. **Download trained models** directly from the web interface

### CLI Interface

```bash
python main.py --csv path/to/data.csv --target target_column_name [--task classification|regression] [--test-size 0.2] [--output best_model.pkl]
```

#### Parameters:
- `--csv`: Path to your CSV file (required)
- `--target`: Name of the target column (required)
- `--task`: Task type (`classification` or `regression`). Auto-inferred if not provided
- `--test-size`: Fraction for test set (default: 0.2)
- `--output`: Output path for saved model (default: best_model.pkl)

#### Example:
```bash
python main.py --csv data/iris.csv --target species --task classification
```

## 🏗️ Project Structure

```
auto-genie/
├── app.py              # Web interface (FastAPI)
├── main.py             # CLI interface
├── eda.py              # Exploratory Data Analysis
├── preprocess.py       # Data preprocessing
├── train_models.py     # Model training
├── evaluate.py         # Model evaluation
├── utils.py            # Utility functions
├── templates/          # Web interface templates
│   ├── form.html       # Upload form
│   └── results.html    # Results display
├── jobs/               # Job tracking for web interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🎯 Supported Models

### Classification Models
- Logistic Regression
- Random Forest
- XGBoost
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- LightGBM (optional)

### Regression Models
- Linear Regression
- Random Forest
- XGBoost
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- LightGBM (optional)

## 📊 Evaluation Metrics

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Regression Metrics
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## 🔧 Requirements

- **Python**: 3.7+
- **Key Dependencies**:
  - pandas
  - scikit-learn
  - xgboost
  - fastapi
  - uvicorn
  - jinja2
  - joblib

See `requirements.txt` for complete dependency list.

## 🚀 Quick Start

1. **For Web Interface**:
   ```bash
   python app.py
   # Open http://localhost:8000 in your browser
   ```

2. **For CLI**:
   ```bash
   python main.py --csv your_data.csv --target target_column
   ```

## 📝 Notes

- **LightGBM**: Optional dependency. If not installed, it will be skipped automatically
- **Model Persistence**: All trained models are saved as .pkl files using joblib
- **Real-time Progress**: Web interface shows live training progress
- **Job Tracking**: Web interface maintains job history for model downloads
- **Auto-inference**: Task type (classification/regression) is automatically detected if not specified

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Made with ❤️ for the ML community**
