# Examples Directory

## Overview
The `examples/` directory contains practical examples, sample datasets, and use cases that demonstrate DataScribe's capabilities. These examples help users understand how to use the platform effectively and provide templates for common analysis scenarios.

## 📁 Contents

### `sample-datasets/`
**Example Datasets for Testing**

Ready-to-use datasets for learning and testing:
- **`iris.csv`**: Classic iris flower dataset (150 rows, 4 features)
- **`titanic.csv`**: Titanic passenger survival data (891 rows, 12 features)
- **`housing.csv`**: California housing prices (20640 rows, 8 features)
- **`diabetes.csv`**: Diabetes progression dataset (442 rows, 10 features)
- **`breast_cancer.csv`**: Breast cancer diagnosis data (569 rows, 30 features)
- **`credit_card.csv`**: Credit card fraud detection (284807 rows, 29 features)

### `notebooks/`
**Jupyter Notebook Examples**

Interactive notebooks demonstrating DataScribe usage:
- **`01_basic_analysis.ipynb`**: Basic EDA workflow
- **`02_advanced_analysis.ipynb`**: Advanced analysis techniques
- **`03_custom_reports.ipynb`**: Custom report generation
- **`04_performance_optimization.ipynb`**: Performance tuning examples
- **`05_integration_examples.ipynb`**: Third-party integrations
- **`06_case_studies.ipynb`**: Real-world analysis examples

### `scripts/`
**Python Script Examples**

Standalone Python scripts for automation:
- **`batch_analysis.py`**: Process multiple datasets
- **`custom_visualizations.py`**: Create custom plots
- **`data_cleaning_pipeline.py`**: Automated data cleaning
- **`report_automation.py`**: Automated report generation
- **`api_integration.py`**: API usage examples
- **`performance_benchmark.py`**: Performance testing

### `configs/`
**Configuration Examples**

Sample configuration files for different use cases:
- **`basic_config.yaml`**: Basic configuration template
- **`production_config.yaml`**: Production environment settings
- **`custom_analysis_config.yaml`**: Custom analysis parameters
- **`high_performance_config.yaml`**: Performance-optimized settings
- **`multi_user_config.yaml`**: Multi-user environment configuration

### `templates/`
**Report and Visualization Templates**

Custom templates for reports and visualizations:
- **`custom_report_template.html`**: Custom HTML report template
- **`branded_report_template.html`**: Branded report template
- **`executive_summary_template.html`**: Executive summary template
- **`technical_report_template.html`**: Technical report template
- **`custom_css/`**: Custom CSS styling templates
- **`plot_templates/`**: Matplotlib and Plotly plot templates

## 🚀 Getting Started Examples

### 1. Basic Analysis Workflow
**Simple EDA Example**

```python
# examples/scripts/basic_analysis.py
import pandas as pd
from datascribe import DataScribe

# Load sample dataset
df = pd.read_csv('examples/sample-datasets/iris.csv')

# Initialize DataScribe
ds = DataScribe()

# Run basic analysis
results = ds.analyze(df)

# Generate reports
ds.generate_reports(results, output_dir='output/')

print("Analysis complete! Check the output/ directory for results.")
```

### 2. Custom Analysis Configuration
**Tailored Analysis Example**

```python
# examples/scripts/custom_analysis.py
from datascribe import DataScribe
from datascribe.config import AnalysisConfig

# Create custom configuration
config = AnalysisConfig(
    include_correlations=True,
    correlation_threshold=0.7,
    outlier_detection=True,
    outlier_method='iqr',
    custom_plots=['pairplot', 'heatmap'],
    report_formats=['html', 'pdf', 'excel']
)

# Initialize with custom config
ds = DataScribe(config=config)

# Load and analyze data
df = pd.read_csv('examples/sample-datasets/housing.csv')
results = ds.analyze(df, target_col='median_house_value')

# Generate custom reports
ds.generate_reports(results, template='executive_summary')
```

### 3. Batch Processing
**Multiple Dataset Analysis**

```python
# examples/scripts/batch_analysis.py
import os
import pandas as pd
from datascribe import DataScribe

def analyze_multiple_datasets(data_dir, output_dir):
    """Analyze multiple datasets in batch"""
    ds = DataScribe()
    
    # Get all CSV files in directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        
        # Load dataset
        file_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(file_path)
        
        # Create output subdirectory
        dataset_name = csv_file.replace('.csv', '')
        dataset_output = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output, exist_ok=True)
        
        # Run analysis
        results = ds.analyze(df)
        
        # Generate reports
        ds.generate_reports(results, output_dir=dataset_output)
        
        print(f"Completed analysis for {csv_file}")

# Usage
analyze_multiple_datasets(
    data_dir='examples/sample-datasets/',
    output_dir='batch_output/'
)
```

## 📊 Sample Dataset Descriptions

### Iris Dataset
**Classic Classification Dataset**

```python
# Dataset characteristics
iris_info = {
    'rows': 150,
    'columns': 4,
    'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    'target': 'species',
    'target_values': ['setosa', 'versicolor', 'virginica'],
    'data_types': 'All numerical (float)',
    'missing_values': 0,
    'use_case': 'Classification, Pattern Recognition',
    'difficulty': 'Beginner'
}

# Load and preview
import pandas as pd
df = pd.read_csv('examples/sample-datasets/iris.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Target distribution:\n{df['species'].value_counts()}")
```

### Titanic Dataset
**Survival Analysis Dataset**

```python
# Dataset characteristics
titanic_info = {
    'rows': 891,
    'columns': 12,
    'features': ['passenger_id', 'survived', 'pclass', 'name', 'sex', 'age', 
                'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked'],
    'target': 'survived',
    'target_values': [0, 1],  # 0 = died, 1 = survived
    'data_types': 'Mixed (numerical, categorical, text)',
    'missing_values': 'Yes (age, cabin, embarked)',
    'use_case': 'Binary Classification, Survival Analysis',
    'difficulty': 'Intermediate'
}

# Load and preview
df = pd.read_csv('examples/sample-datasets/titanic.csv')
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Survival rate: {df['survived'].mean():.2%}")
```

### Housing Dataset
**Regression Dataset**

```python
# Dataset characteristics
housing_info = {
    'rows': 20640,
    'columns': 8,
    'features': ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                'total_bedrooms', 'population', 'households', 'median_income'],
    'target': 'median_house_value',
    'target_type': 'Continuous numerical',
    'data_types': 'All numerical (float)',
    'missing_values': 0,
    'use_case': 'Regression, Housing Price Prediction',
    'difficulty': 'Intermediate'
}

# Load and preview
df = pd.read_csv('examples/sample-datasets/housing.csv')
print(f"Shape: {df.shape}")
print(f"Target statistics:\n{df['median_house_value'].describe()}")
```

## 🔧 Configuration Examples

### Basic Configuration
**Simple Configuration Template**

```yaml
# examples/configs/basic_config.yaml
app:
  name: "DataScribe Basic"
  debug: false
  host: "0.0.0.0"
  port: 8000

analysis:
  max_file_size: 104857600  # 100MB
  allowed_extensions: [".csv", ".xlsx", ".parquet"]
  include_correlations: true
  correlation_threshold: 0.5
  outlier_detection: true
  outlier_method: "iqr"

visualization:
  style: "seaborn"
  figure_size: [12, 8]
  dpi: 300
  color_palette: "viridis"

reports:
  formats: ["html", "pdf"]
  include_code: true
  template: "default"
  output_dir: "reports/"
```

### Production Configuration
**Production Environment Settings**

```yaml
# examples/configs/production_config.yaml
app:
  name: "DataScribe Production"
  debug: false
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_requests: 1000
  timeout: 300

database:
  url: "postgresql://user:pass@localhost/datascribe"
  pool_size: 20
  max_overflow: 30

storage:
  type: "s3"
  bucket: "datascribe-reports"
  region: "us-east-1"
  access_key: "${AWS_ACCESS_KEY}"
  secret_key: "${AWS_SECRET_KEY}"

security:
  enable_auth: true
  jwt_secret: "${JWT_SECRET}"
  cors_origins: ["https://yourdomain.com"]
  rate_limit: 100  # requests per minute

monitoring:
  enable_metrics: true
  prometheus_port: 9090
  log_level: "INFO"
  log_file: "/var/log/datascribe/app.log"
```

## 📝 Report Template Examples

### Custom HTML Report Template
**Branded Report Template**

```html
<!-- examples/templates/branded_report_template.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }} - DataScribe Report</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/report.css') }}">
</head>
<body>
    <header class="report-header">
        <div class="logo">
            <img src="{{ url_for('static', filename='images/logo.png') }}" alt="DataScribe">
        </div>
        <div class="report-info">
            <h1>{{ report_title }}</h1>
            <p class="timestamp">Generated: {{ generation_time }}</p>
        </div>
    </header>
    
    <main class="report-content">
        <!-- Dataset Overview -->
        <section class="overview-section">
            <h2>Dataset Overview</h2>
            <div class="overview-grid">
                <div class="overview-card">
                    <h3>Shape</h3>
                    <p>{{ results.overview.shape.rows }} rows × {{ results.overview.shape.columns }} columns</p>
                </div>
                <div class="overview-card">
                    <h3>Memory Usage</h3>
                    <p>{{ results.overview.memory_usage }}</p>
                </div>
            </div>
        </section>
        
        <!-- Data Quality -->
        <section class="quality-section">
            <h2>Data Quality Assessment</h2>
            <div class="quality-score">
                <span class="score">{{ results.data_quality.data_quality_score }}%</span>
                <span class="label">Overall Quality Score</span>
            </div>
            <!-- More quality metrics... -->
        </section>
        
        <!-- Visualizations -->
        <section class="visualizations-section">
            <h2>Data Visualizations</h2>
            {% for plot_name, plot_path in results.visualizations.items() %}
            <div class="plot-container">
                <h3>{{ plot_name|title }}</h3>
                <img src="{{ plot_path }}" alt="{{ plot_name }}">
            </div>
            {% endfor %}
        </section>
    </main>
    
    <footer class="report-footer">
        <p>&copy; 2024 DataScribe. Generated with ❤️ using automated EDA.</p>
    </footer>
</body>
</html>
```

## 🎨 Custom CSS Templates

### Modern Report Styling
**Professional Report Design**

```css
/* examples/templates/custom_css/modern_report.css */
:root {
    --primary-color: #2c3e50;
    --secondary-color: #3498db;
    --accent-color: #e74c3c;
    --text-color: #2c3e50;
    --background-color: #ecf0f1;
    --card-background: #ffffff;
    --border-color: #bdc3c7;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: var(--text-color);
    background-color: var(--background-color);
    margin: 0;
    padding: 0;
}

.report-header {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.overview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.overview-card {
    background: var(--card-background);
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    border-left: 4px solid var(--secondary-color);
}

.quality-score {
    text-align: center;
    margin: 2rem 0;
}

.quality-score .score {
    font-size: 3rem;
    font-weight: bold;
    color: var(--secondary-color);
    display: block;
}

.plot-container {
    background: var(--card-background);
    padding: 1.5rem;
    margin: 1.5rem 0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.plot-container img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
}
```

## 🔮 Advanced Examples

### Custom Analysis Pipeline
**Extending DataScribe Functionality**

```python
# examples/scripts/custom_analysis_pipeline.py
from datascribe import DataScribe
from datascribe.core import BaseAnalyzer
import pandas as pd
import numpy as np

class CustomAnalyzer(BaseAnalyzer):
    """Custom analyzer for domain-specific analysis"""
    
    def analyze_customer_segments(self, df):
        """Analyze customer segmentation"""
        # RFM Analysis
        if 'last_purchase_date' in df.columns:
            df['recency'] = (pd.Timestamp.now() - pd.to_datetime(df['last_purchase_date'])).dt.days
            df['frequency'] = df.groupby('customer_id')['customer_id'].transform('count')
            df['monetary'] = df.groupby('customer_id')['amount'].transform('sum')
            
            # Segment customers
            df['segment'] = self._assign_rfm_segment(df)
            
        return df
    
    def _assign_rfm_segment(self, df):
        """Assign RFM segments"""
        r_labels = ['High', 'Medium', 'Low']
        f_labels = ['Low', 'Medium', 'High']
        m_labels = ['Low', 'Medium', 'High']
        
        r_quartiles = pd.qcut(df['recency'], q=3, labels=r_labels)
        f_quartiles = pd.qcut(df['frequency'], q=3, labels=f_labels)
        m_quartiles = pd.qcut(df['monetary'], q=3, labels=m_labels)
        
        return r_quartiles.astype(str) + f_quartiles.astype(str) + m_quartiles.astype(str)

# Usage
ds = DataScribe()
ds.add_analyzer(CustomAnalyzer())

# Load customer data
customer_df = pd.read_csv('examples/sample-datasets/customer_data.csv')

# Run custom analysis
results = ds.analyze(customer_df)
customer_df = ds.analyzers['CustomAnalyzer'].analyze_customer_segments(customer_df)

print("Customer segments created!")
print(f"Segment distribution:\n{customer_df['segment'].value_counts()}")
```

## 📚 Related Resources

- [User Guide](../docs/user-guide/)
- [API Reference](../docs/api-docs/)
- [Tutorials](../docs/tutorials/)
- [Sample Datasets](sample-datasets/)
- [Configuration Guide](../docs/developer-guide/configuration.md)

---

**Examples Directory** - Practical examples and use cases for DataScribe
