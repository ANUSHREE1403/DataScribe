# Tests Directory

## Overview
The `tests/` directory contains the comprehensive test suite for DataScribe, ensuring code quality, reliability, and functionality across all components. The testing framework covers unit tests, integration tests, and performance tests for the entire application.

## 📁 Contents

### `test_core/`
**Core EDA Engine Tests**

Tests for the core analysis components:
- **`test_eda_engine.py`**: EDA engine functionality tests
- **`test_visualization_engine.py`**: Visualization generation tests
- **`test_analysis_methods.py`**: Statistical analysis method tests
- **`test_data_quality.py`**: Data quality assessment tests

### `test_web/`
**Web Application Tests**

Tests for the web interface and API:
- **`test_main.py`**: Main application endpoint tests
- **`test_api.py`**: API functionality tests
- **`test_file_upload.py`**: File upload and processing tests
- **`test_job_management.py`**: Job tracking and management tests

### `test_reports/`
**Report Generation Tests**

Tests for report creation and export:
- **`test_report_generator.py`**: Report generation functionality
- **`test_html_reports.py`**: HTML report creation tests
- **`test_pdf_reports.py`**: PDF export tests
- **`test_excel_reports.py`**: Excel workbook generation tests

### `test_utils/`
**Utility Function Tests**

Tests for utility and helper functions:
- **`test_config.py`**: Configuration management tests
- **`test_data_loader.py`**: Data loading utility tests
- **`test_data_cleaner.py`**: Data cleaning function tests
- **`test_formatters.py`**: Formatting utility tests

### `test_integration/`
**Integration Tests**

End-to-end system tests:
- **`test_full_workflow.py`**: Complete analysis workflow tests
- **`test_large_datasets.py`**: Performance with large datasets
- **`test_error_handling.py`**: Error scenario handling tests
- **`test_api_integration.py`**: API integration tests

### `test_data/`
**Test Data and Fixtures**

Sample datasets and test fixtures:
- **`sample_datasets/`**: Small test datasets for testing
- **`test_configs/`**: Test configuration files
- **`expected_results/`**: Expected output for validation
- **`mock_data/`**: Mock data generators

## 🔧 Testing Framework

### Test Runner
- **Pytest**: Primary testing framework
- **Pytest-asyncio**: Async test support
- **Pytest-cov**: Coverage reporting
- **Pytest-html**: HTML test reports

### Test Configuration
```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=core
    --cov=web
    --cov=reports
    --cov=utils
    --cov-report=html
    --cov-report=term-missing
```

### Test Structure
```python
# Example test structure
import pytest
from core.eda_engine import DataScribeEDA

class TestDataScribeEDA:
    """Test suite for DataScribe EDA engine"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_dataset_overview(self, sample_df):
        """Test dataset overview generation"""
        eda = DataScribeEDA(sample_df)
        overview = eda._get_dataset_overview()
        
        assert overview['shape'] == (5, 3)
        assert overview['columns']['total'] == 3
        assert overview['columns']['numerical'] == 2
        assert overview['columns']['categorical'] == 1
    
    def test_data_quality_analysis(self, sample_df):
        """Test data quality analysis"""
        eda = DataScribeEDA(sample_df)
        quality = eda._analyze_data_quality()
        
        assert 'missing_values' in quality
        assert 'duplicates' in quality
        assert 'data_quality_score' in quality
        assert isinstance(quality['data_quality_score'], float)
```

## 🧪 Test Categories

### 1. Unit Tests
**Individual Component Testing**

Test individual functions and methods:
- **Input Validation**: Test with valid and invalid inputs
- **Edge Cases**: Test boundary conditions and edge cases
- **Error Handling**: Test error scenarios and exceptions
- **Return Values**: Verify correct output formats and values

**Example:**
```python
def test_outlier_detection():
    """Test outlier detection functionality"""
    data = pd.Series([1, 2, 3, 100, 4, 5])
    outliers = detect_outliers_iqr(data)
    
    assert len(outliers[0]) == 1  # One outlier detected
    assert outliers[0].iloc[0] == 100  # Correct outlier value
    assert outliers[1] < outliers[2]  # Lower bound < upper bound
```

### 2. Integration Tests
**Component Interaction Testing**

Test how components work together:
- **Data Flow**: Test data passing between components
- **API Integration**: Test API endpoint interactions
- **File Processing**: Test complete file processing pipeline
- **Error Propagation**: Test error handling across components

**Example:**
```python
def test_complete_analysis_workflow():
    """Test complete analysis workflow"""
    # Load test dataset
    df = load_test_dataset()
    
    # Run EDA analysis
    results = run_eda(df, target_col='target')
    
    # Generate visualizations
    plots = generate_visualizations(df, results, 'target')
    
    # Generate reports
    reports = generate_reports(df, results, plots, 'target')
    
    # Verify all outputs
    assert 'overview' in results
    assert 'data_quality' in results
    assert 'overview' in plots
    assert 'html' in reports
    assert 'pdf' in reports
    assert 'excel' in reports
```

### 3. Performance Tests
**System Performance Testing**

Test system performance and scalability:
- **Large Datasets**: Test with datasets of various sizes
- **Memory Usage**: Monitor memory consumption
- **Processing Time**: Measure analysis completion time
- **Scalability**: Test performance scaling with data size

**Example:**
```python
@pytest.mark.performance
def test_large_dataset_performance():
    """Test performance with large dataset"""
    # Generate large test dataset
    large_df = generate_large_dataset(rows=10000, cols=50)
    
    # Measure processing time
    start_time = time.time()
    results = run_eda(large_df)
    processing_time = time.time() - start_time
    
    # Performance assertions
    assert processing_time < 30  # Should complete within 30 seconds
    assert 'overview' in results
    assert 'data_quality' in results
```

### 4. Error Handling Tests
**Error Scenario Testing**

Test system behavior under error conditions:
- **Invalid Inputs**: Test with malformed data
- **File Errors**: Test with corrupted or invalid files
- **System Errors**: Test with insufficient resources
- **Recovery**: Test error recovery mechanisms

**Example:**
```python
def test_invalid_file_handling():
    """Test handling of invalid files"""
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_dataset("invalid_file.txt")
    
    with pytest.raises(FileNotFoundError):
        load_dataset("nonexistent_file.csv")

def test_malformed_data_handling():
    """Test handling of malformed data"""
    # Create DataFrame with mixed data types
    malformed_df = pd.DataFrame({
        'mixed_column': [1, 'text', 3.14, None, 'more_text']
    })
    
    # Should handle gracefully
    results = run_eda(malformed_df)
    assert 'error' not in results
    assert 'data_quality' in results
```

## 📊 Test Data Management

### Sample Datasets
**Test Data Creation**

Create various test datasets:
```python
def create_test_datasets():
    """Create various test datasets for testing"""
    datasets = {}
    
    # Small numerical dataset
    datasets['small_numeric'] = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'target': np.random.binomial(1, 0.5, 100)
    })
    
    # Mixed data types dataset
    datasets['mixed_types'] = pd.DataFrame({
        'numeric': np.random.normal(0, 1, 50),
        'categorical': np.random.choice(['A', 'B', 'C'], 50),
        'datetime': pd.date_range('2023-01-01', periods=50, freq='D'),
        'target': np.random.binomial(1, 0.3, 50)
    })
    
    # Dataset with missing values
    datasets['with_missing'] = datasets['small_numeric'].copy()
    datasets['with_missing'].iloc[0:10, 0] = np.nan
    datasets['with_missing'].iloc[5:15, 1] = np.nan
    
    return datasets
```

### Mock Data Generators
**Dynamic Test Data**

Generate test data dynamically:
```python
def generate_mock_dataset(rows=100, cols=10, missing_pct=0.1):
    """Generate mock dataset with specified characteristics"""
    # Generate numerical data
    data = np.random.normal(0, 1, (rows, cols))
    
    # Add missing values
    if missing_pct > 0:
        missing_mask = np.random.random((rows, cols)) < missing_pct
        data[missing_mask] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(cols)])
    
    # Add target column
    df['target'] = np.random.binomial(1, 0.5, rows)
    
    return df
```

## 🔍 Test Coverage

### Coverage Goals
**Comprehensive Testing Coverage**

Target coverage percentages:
- **Core Engine**: 95%+ coverage
- **Web Interface**: 90%+ coverage
- **Report Generation**: 85%+ coverage
- **Utilities**: 90%+ coverage
- **Overall**: 90%+ coverage

### Coverage Reporting
**Detailed Coverage Analysis**

Generate coverage reports:
```bash
# Generate coverage report
pytest --cov=core --cov=web --cov=reports --cov=utils --cov-report=html

# View coverage report
open htmlcov/index.html
```

## 🚀 Test Execution

### Running Tests
**Test Execution Commands**

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_core/
pytest tests/test_web/
pytest tests/test_reports/

# Run specific test file
pytest tests/test_core/test_eda_engine.py

# Run specific test function
pytest tests/test_core/test_eda_engine.py::TestDataScribeEDA::test_dataset_overview

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov

# Run performance tests only
pytest -m performance

# Run tests in parallel
pytest -n auto
```

### Continuous Integration
**Automated Testing**

GitHub Actions workflow:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    - name: Run tests
      run: |
        pytest --cov=core --cov=web --cov=reports --cov=utils --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 🔧 Test Configuration

### Environment Setup
**Test Environment Configuration**

```python
# conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory"""
    return "tests/test_data/sample_datasets"

@pytest.fixture(scope="session")
def sample_datasets():
    """Load all sample datasets"""
    datasets = {}
    data_dir = test_data_dir()
    
    # Load various test datasets
    datasets['iris'] = pd.read_csv(f"{data_dir}/iris.csv")
    datasets['titanic'] = pd.read_csv(f"{data_dir}/titanic.csv")
    datasets['housing'] = pd.read_csv(f"{data_dir}/housing.csv")
    
    return datasets

@pytest.fixture
def mock_config():
    """Create mock configuration for testing"""
    return {
        'app_name': 'DataScribe Test',
        'debug': True,
        'max_file_size': 1024 * 1024,  # 1MB
        'allowed_extensions': ['.csv', '.xlsx']
    }
```

## 🔮 Future Enhancements

### Advanced Testing
- **Property-Based Testing**: Hypothesis framework integration
- **Mutation Testing**: Code mutation testing
- **Load Testing**: Performance under load
- **Security Testing**: Security vulnerability testing
- **Accessibility Testing**: UI accessibility testing

### Test Automation
- **Auto Test Generation**: Automatic test case generation
- **Test Data Synthesis**: Synthetic test data generation
- **Performance Regression**: Automatic performance regression detection
- **Test Parallelization**: Improved test parallelization

## 📚 Related Documentation

- [Testing Guide](../docs/testing-guide.md)
- [Test Data Management](../docs/test-data.md)
- [Performance Testing](../docs/performance-testing.md)
- [Coverage Analysis](../docs/coverage-analysis.md)

---

**Tests Directory** - Comprehensive testing framework for DataScribe
