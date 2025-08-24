# Reports Directory

## Overview
The `reports/` directory contains the report generation engine for DataScribe - the system that transforms analysis results into professional, shareable reports in multiple formats including HTML, PDF, and Excel.

## 📁 Contents

### `report_generator.py`
**Comprehensive Report Generation Engine**

The main report generation system that creates:
- **HTML Reports**: Interactive, responsive web-based reports
- **PDF Reports**: Print-ready, professional PDF documents
- **Excel Reports**: Multi-sheet Excel workbooks with structured data
- **Custom Templates**: Jinja2-based template system for customization

**Key Classes:**
- `DataScribeReportGenerator`: Main report generation class
- `generate_reports()`: Main function for creating all report formats

## 🔧 Technical Details

### Dependencies
- **Jinja2**: Template engine for HTML generation
- **WeasyPrint**: HTML to PDF conversion
- **OpenPyXL**: Excel workbook creation and formatting
- **Pandas**: Data manipulation for Excel export

### Architecture
- **Template-Based**: Jinja2 templates for consistent formatting
- **Multi-Format**: Single engine supporting multiple output formats
- **Configurable**: Customizable report sections and styling
- **Extensible**: Easy to add new report formats and templates

### Output Formats
- **HTML**: Interactive web reports with embedded visualizations
- **PDF**: High-quality print documents
- **Excel**: Structured data workbooks with multiple sheets

## 🚀 Report Types

### 1. HTML Reports
**Interactive Web Reports**

**Features:**
- Responsive design for all devices
- Embedded visualizations and plots
- Professional styling with CSS
- Interactive elements and navigation
- Mobile-friendly layout

**Structure:**
- Header with DataScribe branding
- Dataset overview section
- Data quality analysis
- Visualization galleries
- Insights and recommendations
- Footer with metadata

**Styling:**
- Modern gradient backgrounds
- Professional color scheme
- Clean typography
- Responsive grid layouts
- Hover effects and animations

### 2. PDF Reports
**Print-Ready Documents**

**Features:**
- High-quality graphics and plots
- Professional formatting
- Print-optimized layout
- Consistent styling
- Easy sharing and distribution

**Generation Process:**
1. HTML report creation
2. WeasyPrint conversion
3. High-resolution output
4. Optimized for printing

### 3. Excel Reports
**Structured Data Workbooks**

**Features:**
- Multiple specialized sheets
- Formatted tables and charts
- Color-coded data quality
- Statistical summaries
- Correlation matrices

**Sheet Structure:**
- **Dataset Overview**: Basic information and metadata
- **Data Quality**: Quality scores and issue summaries
- **Statistical Summary**: Comprehensive statistics
- **Correlations**: Correlation analysis and matrices
- **Insights & Recommendations**: AI-generated insights

## 📊 Report Content

### Dataset Overview
- **Basic Information**: Rows, columns, memory usage
- **Column Types**: Numerical, categorical, datetime counts
- **Data Types**: Distribution of data types
- **Target Variable**: If specified, target column information

### Data Quality Analysis
- **Quality Score**: Overall 0-100 quality rating
- **Missing Values**: Counts and percentages by column
- **Duplicates**: Duplicate row detection and counts
- **Constant Columns**: Columns with no variation
- **Quality Insights**: Automated quality assessment

### Statistical Summaries
- **Numerical Statistics**: Mean, median, std, quartiles, skewness, kurtosis
- **Categorical Statistics**: Unique counts, value distributions, missing counts
- **Outlier Analysis**: Outlier detection and statistics
- **Distribution Analysis**: Data distribution characteristics

### Correlation Analysis
- **Correlation Matrix**: Full numerical feature correlations
- **High Correlations**: Features with |r| > 0.7
- **Feature Relationships**: Feature-target correlations
- **Correlation Insights**: Automated correlation interpretation

### Visualizations
- **Overview Plots**: Dataset structure and composition
- **Quality Plots**: Missing values and data issues
- **Univariate Plots**: Feature distributions and statistics
- **Bivariate Plots**: Correlation matrices and relationships
- **Multivariate Plots**: PCA and feature importance
- **Target Plots**: Target variable analysis

### Insights & Recommendations
- **Data Quality Insights**: Automated quality observations
- **Feature Insights**: Feature characteristics and patterns
- **Recommendations**: Actionable next steps
- **Best Practices**: Data science recommendations

## 🎨 Template System

### Jinja2 Templates
**Flexible Template Engine**

**Features:**
- HTML-based templates
- Dynamic content insertion
- Conditional rendering
- Loop structures for data
- Custom filters and functions

**Template Structure:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>{{ css_styles }}</style>
</head>
<body>
    <div class="header">
        <h1>{{ app_name }}</h1>
        <p>{{ subtitle }}</p>
    </div>
    
    {% for section in sections %}
    <div class="section">
        <h2>{{ section.title }}</h2>
        {{ section.content }}
    </div>
    {% endfor %}
</body>
</html>
```

### CSS Styling
**Professional Visual Design**

**Features:**
- Modern gradient backgrounds
- Professional color scheme
- Responsive grid layouts
- Hover effects and animations
- Mobile-friendly design

**Color Palette:**
- Primary: #667eea (Blue)
- Secondary: #764ba2 (Purple)
- Accent: #f093fb (Pink)
- Success: #56ab2f (Green)
- Warning: #ffc000 (Yellow)
- Error: #d62728 (Red)

## 🔧 Configuration

### Report Settings
```python
# Report Configuration
default_report_sections: [
    "overview", "missing_values", "outliers", 
    "distributions", "correlations", "target_analysis", 
    "recommendations"
]

# Export Formats
enable_pdf_export: True
enable_excel_export: True
enable_code_export: True

# Report Styling
report_theme: "professional"
include_branding: True
custom_css: None
```

### Template Customization
```python
# Custom Templates
html_template_path: "templates/custom_report.html"
css_template_path: "templates/custom_styles.css"

# Branding
company_logo: "assets/logo.png"
company_name: "DataScribe"
company_website: "https://datascribe.ai"
```

## 🚀 Usage Examples

### Generate All Report Formats
```python
from reports.report_generator import generate_reports

# Generate reports
reports = generate_reports(df, analysis_results, plot_files, target_col='target')

# Access specific formats
html_report = reports['html']
pdf_report = reports['pdf']
excel_report = reports['excel']
```

### Custom Report Generation
```python
from reports.report_generator import DataScribeReportGenerator

# Create generator
generator = DataScribeReportGenerator(df, analysis_results, target_col='target')

# Generate specific format
html_file = generator.generate_html_report(plot_files)
pdf_file = generator.generate_pdf_report(plot_files)
excel_file = generator.generate_excel_report()
```

### Template Customization
```python
# Custom template data
template_data = {
    'company_name': 'My Company',
    'report_title': 'Custom Analysis Report',
    'additional_sections': ['custom_metrics', 'business_insights']
}

# Generate with custom data
html_content = generator.generate_html_report(plot_files, template_data)
```

## 📈 Report Customization

### Section Selection
- **Core Sections**: Always included (overview, quality, statistics)
- **Optional Sections**: Configurable inclusion (visualizations, insights)
- **Custom Sections**: User-defined additional content
- **Section Ordering**: Configurable section sequence

### Styling Options
- **Theme Selection**: Professional, modern, minimalist themes
- **Color Schemes**: Customizable color palettes
- **Typography**: Font family and size customization
- **Layout**: Grid layouts and spacing options

### Content Filtering
- **Data Thresholds**: Configurable quality thresholds
- **Insight Filtering**: Relevance-based insight selection
- **Visualization Limits**: Maximum plot counts and sizes
- **Detail Levels**: Summary vs. detailed report options

## 🔮 Future Enhancements

### Advanced Templates
- **React/Vue Templates**: Modern JavaScript frameworks
- **LaTeX Templates**: Academic paper formatting
- **PowerPoint Export**: Presentation-ready slides
- **Word Export**: Document format export

### Interactive Features
- **Dynamic Charts**: Interactive Plotly visualizations
- **Drill-Down**: Clickable chart elements
- **Real-time Updates**: Live data updates
- **Custom Dashboards**: User-defined dashboard layouts

### Advanced Export
- **Email Integration**: Direct email sending
- **Cloud Storage**: Automatic cloud upload
- **API Integration**: External system integration
- **Scheduled Reports**: Automated report generation

## 🧪 Testing

### Report Generation Tests
```bash
# Test report generation
pytest tests/test_reports/
```

### Manual Testing
```python
# Test report generation
from reports.report_generator import generate_reports

# Generate test reports
reports = generate_reports(test_df, test_results, test_plots)

# Verify output files
assert os.path.exists(reports['html'])
assert os.path.exists(reports['pdf'])
assert os.path.exists(reports['excel'])
```

## 📚 Related Documentation

- [Report Templates](../docs/report-templates.md)
- [Customization Guide](../docs/report-customization.md)
- [Export Formats](../docs/export-formats.md)
- [Template Development](../docs/template-development.md)

---

**Reports Directory** - Professional report generation for DataScribe
