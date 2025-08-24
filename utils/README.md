# Utils Directory

## Overview
The `utils/` directory contains utility functions, helper classes, and common functionality used throughout DataScribe. These utilities provide reusable code for data processing, validation, formatting, and other common operations.

## 📁 Contents

### `config.py`
**Configuration Management System**

Centralized configuration management for DataScribe:
- **App Settings**: Application name, version, and basic configuration
- **Server Configuration**: Host, port, and development settings
- **Database Settings**: Connection strings and database configuration
- **File Storage**: Upload directories, file size limits, allowed extensions
- **EDA Configuration**: Analysis parameters and thresholds
- **Report Settings**: Export format preferences and styling options

**Key Features:**
- Environment variable support via `.env` files
- Type-safe configuration with Pydantic
- Default values for all settings
- Environment-specific configuration

### `data_loader.py`
**Data Loading and Validation Utilities**

Comprehensive data loading system:
- **File Format Support**: CSV, Excel, Parquet file loading
- **Data Validation**: Schema validation and data type inference
- **Error Handling**: Graceful handling of malformed files
- **Memory Optimization**: Efficient loading of large datasets
- **Encoding Detection**: Automatic character encoding detection

### `data_cleaner.py`
**Data Cleaning and Preprocessing Utilities**

Automated data cleaning functions:
- **Missing Value Handling**: Detection and imputation strategies
- **Outlier Detection**: Statistical outlier identification
- **Data Type Conversion**: Automatic type inference and conversion
- **Duplicate Removal**: Duplicate row and column detection
- **Data Standardization**: Normalization and scaling utilities

### `formatters.py`
**Data Formatting and Display Utilities**

Data presentation and formatting:
- **Number Formatting**: Scientific notation, percentage, currency
- **Date Formatting**: Date and time string formatting
- **Table Formatting**: HTML table generation and styling
- **Chart Formatting**: Plot styling and color schemes
- **Report Formatting**: Consistent text and data formatting

### `validators.py`
**Data Validation and Quality Checks**

Comprehensive validation system:
- **Schema Validation**: Column structure and data type validation
- **Range Validation**: Numerical value range checking
- **Format Validation**: String format and pattern validation
- **Business Logic**: Domain-specific validation rules
- **Quality Metrics**: Data quality scoring and assessment

## 🔧 Technical Details

### Dependencies
- **Pydantic**: Configuration validation and settings management
- **Pandas**: Data manipulation and validation
- **NumPy**: Numerical operations and validation
- **Python-dotenv**: Environment variable loading

### Architecture
- **Modular Design**: Each utility is self-contained and focused
- **Reusable Functions**: Common operations extracted into utilities
- **Error Handling**: Comprehensive error handling and logging
- **Performance Optimized**: Efficient algorithms for large datasets

### Configuration Management
```python
# Configuration Structure
class Settings(BaseSettings):
    # App Configuration
    app_name: str = "DataScribe"
    app_subtitle: str = "Democratizing Data Analysis"
    app_version: str = "1.0.0"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # Database Configuration
    database_url: str = "postgresql://user:password@localhost/datascribe"
    
    # File Storage
    upload_dir: str = "uploads"
    reports_dir: str = "reports"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    # EDA Configuration
    max_rows_for_analysis: int = 100000
    correlation_threshold: float = 0.7
    outlier_method: str = "iqr"
```

## 🚀 Utility Functions

### Data Loading
```python
from utils.data_loader import load_dataset, validate_dataset

# Load dataset with validation
df = load_dataset('data.csv')

# Validate dataset structure
validation_result = validate_dataset(df, expected_schema)
```

### Data Cleaning
```python
from utils.data_cleaner import clean_dataset, detect_outliers

# Clean dataset
cleaned_df = clean_dataset(df, 
                         handle_missing='impute',
                         remove_duplicates=True,
                         detect_outliers=True)

# Detect outliers
outliers = detect_outliers(df['column'], method='iqr')
```

### Data Formatting
```python
from utils.formatters import format_number, format_table

# Format numbers
formatted = format_number(1234.5678, format='currency', locale='en_US')

# Format table
html_table = format_table(df, 
                         title='Dataset Summary',
                         style='bootstrap')
```

### Data Validation
```python
from utils.validators import validate_schema, check_data_quality

# Validate schema
schema_valid = validate_schema(df, expected_columns, expected_types)

# Check data quality
quality_score = check_data_quality(df)
```

## 🔍 Data Quality Functions

### Missing Value Analysis
```python
def analyze_missing_values(df):
    """Analyze missing values in dataset"""
    missing_info = {
        'total_missing': df.isnull().sum().sum(),
        'missing_by_column': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'missing_patterns': analyze_missing_patterns(df)
    }
    return missing_info
```

### Outlier Detection
```python
def detect_outliers_iqr(data, factor=1.5):
    """Detect outliers using IQR method"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers, lower_bound, upper_bound
```

### Data Type Inference
```python
def infer_data_types(df):
    """Automatically infer optimal data types"""
    type_mapping = {}
    
    for column in df.columns:
        if df[column].dtype == 'object':
            # Try to convert to more specific types
            if can_convert_to_datetime(df[column]):
                type_mapping[column] = 'datetime64[ns]'
            elif can_convert_to_category(df[column]):
                type_mapping[column] = 'category'
            else:
                type_mapping[column] = 'object'
        else:
            type_mapping[column] = df[column].dtype
    
    return type_mapping
```

## 📊 Formatting Utilities

### Number Formatting
```python
def format_number(value, format_type='default', **kwargs):
    """Format numbers for display"""
    if format_type == 'currency':
        return f"${value:,.2f}"
    elif format_type == 'percentage':
        return f"{value:.2f}%"
    elif format_type == 'scientific':
        return f"{value:.2e}"
    else:
        return f"{value:,.2f}"
```

### Table Generation
```python
def generate_html_table(df, title=None, style='default'):
    """Generate HTML table from DataFrame"""
    table_html = df.to_html(classes=f'table table-{style}')
    
    if title:
        table_html = f'<h3>{title}</h3>\n{table_html}'
    
    return table_html
```

### Chart Styling
```python
def get_chart_style(style_name='default'):
    """Get predefined chart styling"""
    styles = {
        'default': {
            'figure.figsize': (10, 6),
            'figure.dpi': 100,
            'axes.grid': True,
            'grid.alpha': 0.3
        },
        'professional': {
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'axes.grid': True,
            'grid.alpha': 0.2,
            'font.size': 12
        }
    }
    return styles.get(style_name, styles['default'])
```

## 🔧 Configuration Utilities

### Environment Loading
```python
def load_environment_config():
    """Load configuration from environment variables"""
    config = {}
    
    # Load from .env file
    load_dotenv()
    
    # Map environment variables to config
    env_mapping = {
        'DATASCRIBE_DEBUG': 'debug',
        'DATASCRIBE_HOST': 'host',
        'DATASCRIBE_PORT': 'port',
        'DATASCRIBE_DATABASE_URL': 'database_url'
    }
    
    for env_var, config_key in env_mapping.items():
        if env_var in os.environ:
            config[config_key] = os.environ[env_var]
    
    return config
```

### Configuration Validation
```python
def validate_config(config):
    """Validate configuration values"""
    errors = []
    
    # Validate port number
    if 'port' in config:
        try:
            port = int(config['port'])
            if not (1 <= port <= 65535):
                errors.append(f"Port must be between 1 and 65535, got {port}")
        except ValueError:
            errors.append(f"Port must be an integer, got {config['port']}")
    
    # Validate file paths
    if 'upload_dir' in config:
        if not os.path.exists(config['upload_dir']):
            try:
                os.makedirs(config['upload_dir'])
            except OSError:
                errors.append(f"Cannot create upload directory: {config['upload_dir']}")
    
    return errors
```

## 🚀 Performance Utilities

### Memory Optimization
```python
def optimize_memory_usage(df):
    """Optimize DataFrame memory usage"""
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    for column in df.columns:
        if df[column].dtype == 'object':
            # Convert object columns to category if beneficial
            if df[column].nunique() / len(df) < 0.5:
                df[column] = df[column].astype('category')
        
        elif df[column].dtype == 'int64':
            # Downcast integers if possible
            if df[column].min() >= 0:
                if df[column].max() < 255:
                    df[column] = df[column].astype('uint8')
                elif df[column].max() < 65535:
                    df[column] = df[column].astype('uint16')
                else:
                    df[column] = df[column].astype('uint32')
            else:
                if df[column].min() > -128 and df[column].max() < 127:
                    df[column] = df[column].astype('int8')
                elif df[column].min() > -32768 and df[column].max() < 32767:
                    df[column] = df[column].astype('int16')
                else:
                    df[column] = df[column].astype('int32')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    memory_saved = initial_memory - final_memory
    
    return df, memory_saved
```

### Batch Processing
```python
def process_in_batches(df, batch_size=1000, processor_func=None):
    """Process DataFrame in batches for memory efficiency"""
    results = []
    
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]
        
        if processor_func:
            batch_result = processor_func(batch)
            results.append(batch_result)
        else:
            results.append(batch)
    
    return results
```

## 🔮 Future Enhancements

### Advanced Utilities
- **Data Profiling**: Comprehensive data profiling and statistics
- **Schema Evolution**: Automatic schema change detection
- **Data Lineage**: Track data transformations and sources
- **Performance Monitoring**: Real-time performance metrics
- **Caching System**: Intelligent caching for repeated operations

### Integration Utilities
- **Database Connectors**: Support for various database systems
- **API Integrations**: External service integrations
- **Cloud Storage**: Cloud platform storage utilities
- **Message Queues**: Asynchronous processing support
- **Monitoring**: Application monitoring and alerting

## 🧪 Testing

### Utility Testing
```bash
# Test utility functions
pytest tests/test_utils/
```

### Performance Testing
```bash
# Test performance utilities
pytest tests/test_utils/ -k "performance"
```

## 📚 Related Documentation

- [Configuration Guide](../docs/configuration.md)
- [Data Loading](../docs/data-loading.md)
- [Data Cleaning](../docs/data-cleaning.md)
- [Performance Optimization](../docs/performance.md)

---

**Utils Directory** - Reusable utilities and helper functions for DataScribe
