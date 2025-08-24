# Core Directory

## Overview
The `core/` directory contains the heart of DataScribe - the EDA engine and analysis components that power all data analysis functionality.

## 📁 Contents

### `eda_engine.py`
**Core EDA Engine for DataScribe MVP**

The main analysis engine that handles:
- **Dataset Overview**: Shape, memory usage, data types, column analysis
- **Data Quality Analysis**: Missing values, duplicates, constant columns, quality scoring
- **Statistical Summaries**: Comprehensive statistics for numerical and categorical columns
- **Univariate Analysis**: Distribution analysis, outlier detection, descriptive statistics
- **Bivariate Analysis**: Feature-target relationships, correlation analysis
- **Multivariate Analysis**: PCA, feature importance, dimensionality analysis
- **Target Analysis**: Classification/regression target insights
- **AI Insights**: Automated recommendations and actionable insights

**Key Classes:**
- `DataScribeEDA`: Main EDA engine class
- `run_eda()`: Main function for running complete analysis

### `visualization_engine.py`
**Comprehensive Visualization Engine**

Generates publication-ready visualizations:
- **Overview Plots**: Dataset structure, data types, missing values
- **Data Quality Plots**: Missing values heatmaps, quality scores, issue summaries
- **Univariate Plots**: Histograms, distributions, outlier visualizations
- **Bivariate Plots**: Correlation matrices, high correlation highlights
- **Multivariate Plots**: PCA analysis, feature importance charts
- **Target Analysis Plots**: Classification/regression target visualizations

**Key Classes:**
- `DataScribeVisualizer`: Main visualization class
- `generate_visualizations()`: Main function for creating all plots

## 🔧 Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Base plotting library
- **seaborn**: Statistical visualizations
- **plotly**: Interactive plots (future enhancement)
- **scikit-learn**: PCA and statistical analysis

### Architecture
- **Modular Design**: Each analysis component is self-contained
- **Configurable**: Analysis parameters can be tuned via configuration
- **Extensible**: Easy to add new analysis types and visualizations
- **Error Handling**: Robust error handling with fallback options

### Performance
- **Memory Efficient**: Processes large datasets in chunks
- **Optimized**: Uses vectorized operations where possible
- **Scalable**: Handles datasets up to 100K rows and 100 columns
- **Fast**: Optimized algorithms for statistical computations

## 🚀 Usage Examples

### Basic EDA Analysis
```python
from core.eda_engine import run_eda
import pandas as pd

# Load dataset
df = pd.read_csv('dataset.csv')

# Run complete analysis
results = run_eda(df, target_col='target')

# Access specific results
quality_score = results['data_quality']['data_quality_score']
correlations = results['bivariate']['correlations']
insights = results['insights']
```

### Generate Visualizations
```python
from core.visualization_engine import generate_visualizations

# Generate all plots
plot_files = generate_visualizations(df, results, target_col='target')

# Access specific plots
overview_plot = plot_files['overview']
quality_plot = plot_files['data_quality']
univariate_plot = plot_files['univariate']
```

## 🔍 Analysis Types

### 1. Data Quality Analysis
- **Missing Values**: Count, percentage, patterns
- **Duplicates**: Row-level duplicate detection
- **Constant Columns**: Columns with no variation
- **Quality Scoring**: 0-100 score based on data issues

### 2. Statistical Analysis
- **Numerical**: Mean, median, std, quartiles, skewness, kurtosis
- **Categorical**: Unique counts, value distributions, missing counts
- **Outlier Detection**: IQR method with configurable thresholds

### 3. Relationship Analysis
- **Correlations**: Pearson correlation for numerical features
- **Feature-Target**: ANOVA, chi-square, correlation analysis
- **High Correlations**: Automatic detection of highly correlated features

### 4. Advanced Analysis
- **PCA**: Principal component analysis for dimensionality
- **Feature Importance**: Correlation-based importance ranking
- **Class Balance**: Assessment for classification targets

## 📊 Output Format

All analysis results are returned as structured dictionaries:

```python
{
    'overview': {
        'shape': (rows, cols),
        'memory_usage': 'MB',
        'data_types': {...},
        'columns': {...}
    },
    'data_quality': {
        'missing_values': {...},
        'duplicates': {...},
        'constant_columns': [...],
        'data_quality_score': 85.5
    },
    'statistics': {
        'numerical': {...},
        'categorical': {...}
    },
    'univariate': {...},
    'bivariate': {...},
    'multivariate': {...},
    'target_analysis': {...},
    'insights': {
        'data_quality_insights': [...],
        'feature_insights': [...],
        'recommendations': [...]
    }
}
```

## 🎯 Configuration

Analysis behavior can be configured via the main config file:

```python
# EDA Configuration
max_rows_for_analysis: 100000
max_columns_for_analysis: 100
correlation_threshold: 0.7
outlier_method: "iqr"  # iqr, zscore, isolation_forest
```

## 🔮 Future Enhancements

- **Advanced Outlier Detection**: Isolation Forest, Local Outlier Factor
- **Statistical Tests**: Normality tests, homogeneity tests
- **Time Series Analysis**: Trend analysis, seasonality detection
- **Text Analysis**: NLP-based categorical analysis
- **Interactive Plots**: Plotly-based interactive visualizations
- **Custom Metrics**: User-defined analysis metrics

## 🧪 Testing

Run core component tests:

```bash
pytest tests/test_core/
```

## 📚 Related Documentation

- [EDA Engine API Reference](../docs/eda-engine-api.md)
- [Visualization Guide](../docs/visualization-guide.md)
- [Analysis Configuration](../docs/analysis-config.md)
- [Performance Tuning](../docs/performance-tuning.md)

---

**Core Directory** - The analytical foundation of DataScribe
