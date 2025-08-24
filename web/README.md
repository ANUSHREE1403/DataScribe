# Web Directory

## Overview
The `web/` directory contains the web application layer of DataScribe - the FastAPI-based web server, API endpoints, and web interface components that provide the user-facing functionality.

## 📁 Contents

### `main.py`
**Main FastAPI Application**

The core web application that provides:
- **Web Interface**: HTML-based user interface for dataset upload and analysis
- **REST API**: Programmatic access to all DataScribe functionality
- **File Handling**: Secure file upload and processing
- **Job Management**: Asynchronous analysis job tracking
- **Result Serving**: Dynamic result display and file downloads

**Key Components:**
- FastAPI application setup and configuration
- File upload and validation endpoints
- Analysis job management and tracking
- Result serving and visualization display
- Health check and monitoring endpoints

### `templates/`
**HTML Templates Directory**

Contains Jinja2 templates for the web interface:
- **`index.html`**: Main upload and configuration page
- **`results.html`**: Analysis results display page
- **`error.html`**: Error handling and display page

### `static/`
**Static Assets Directory**

Stores generated visualizations and static files:
- **Generated Plots**: PNG files from analysis results
- **CSS/JS**: Future enhancement for custom styling
- **Images**: Icons and branding assets

## 🔧 Technical Details

### Framework
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **Jinja2**: Template engine for HTML generation
- **Python-multipart**: File upload handling

### Architecture
- **RESTful API**: Standard REST endpoints for all operations
- **Async Support**: Non-blocking I/O for better performance
- **File Processing**: Secure file handling with validation
- **Job Queue**: Asynchronous job processing system
- **Error Handling**: Comprehensive error handling and user feedback

### Security
- **File Validation**: Type and size validation for uploads
- **Path Sanitization**: Secure file path handling
- **Input Validation**: Request parameter validation
- **Error Sanitization**: Safe error message display

## 🚀 API Endpoints

### Core Endpoints

#### `GET /`
**Home Page**
- Displays the main upload form
- Shows DataScribe branding and information
- Entry point for users

#### `POST /analyze`
**Dataset Analysis**
- Accepts file uploads (CSV, Excel, Parquet)
- Configurable analysis parameters
- Returns job ID for tracking

**Parameters:**
- `file`: Dataset file (required)
- `target_column`: Target variable column (optional)
- `include_plots`: Generate visualizations (default: true)
- `include_code`: Generate code export (default: false)

**Response:**
```json
{
    "job_id": "uuid-string",
    "status": "completed",
    "message": "Analysis completed successfully",
    "dataset_shape": [1000, 10],
    "analysis_summary": {
        "data_quality_score": 85.5,
        "missing_values": 0,
        "duplicates": 0
    }
}
```

#### `GET /results/{job_id}`
**Analysis Results Page**
- Displays complete analysis results
- Shows visualizations and insights
- Provides download options

#### `GET /api/results/{job_id}`
**API Results Endpoint**
- Returns analysis results as JSON
- Programmatic access to results
- Includes all analysis data and plot file paths

#### `GET /download/{job_id}/dataset`
**Dataset Download**
- Downloads the analyzed dataset
- Returns CSV file for user download

#### `GET /health`
**Health Check**
- System status monitoring
- Application health information
- Active job count

## 🌐 Web Interface

### Upload Form (`index.html`)
- **File Selection**: Drag-and-drop or browse file selection
- **Configuration Options**: Target column, visualization preferences
- **Validation**: Real-time file validation and feedback
- **Progress Tracking**: Upload progress indication

### Results Display (`results.html`)
- **Analysis Overview**: Dataset summary and quality score
- **Visualizations**: Interactive plot display
- **Insights**: AI-generated recommendations
- **Download Options**: Report and dataset downloads

### Error Handling (`error.html`)
- **User-Friendly Errors**: Clear error messages
- **Recovery Options**: Suggestions for fixing issues
- **Support Information**: Help and contact details

## 🔄 Job Management

### Job Lifecycle
1. **Creation**: Job created when analysis starts
2. **Processing**: Analysis runs asynchronously
3. **Completion**: Results stored and job marked complete
4. **Retrieval**: Users can access results via job ID
5. **Cleanup**: Old jobs can be cleaned up (future enhancement)

### Job Storage
- **In-Memory**: Jobs stored in application memory
- **Persistence**: Future enhancement for database storage
- **File Storage**: Generated files stored on disk
- **Cleanup**: Automatic cleanup of temporary files

## 📊 File Handling

### Supported Formats
- **CSV**: Comma-separated values
- **Excel**: .xlsx and .xls files
- **Parquet**: Columnar storage format

### File Validation
- **Size Limits**: Configurable maximum file size
- **Type Checking**: File extension validation
- **Content Validation**: Basic content structure checking
- **Security**: Path traversal protection

### File Processing
- **Upload**: Secure file upload handling
- **Storage**: Temporary storage during analysis
- **Processing**: Pandas-based data loading
- **Cleanup**: Automatic temporary file removal

## 🎨 User Experience

### Responsive Design
- **Mobile Friendly**: Responsive layout for all devices
- **Modern UI**: Clean, professional interface design
- **Accessibility**: Screen reader and keyboard navigation support
- **Performance**: Fast loading and smooth interactions

### Interactive Elements
- **Real-time Validation**: Immediate feedback on user input
- **Progress Indicators**: Clear progress tracking
- **Error Handling**: Helpful error messages and recovery
- **Success Feedback**: Clear confirmation of completed actions

## 🔧 Configuration

### Server Settings
```python
# Server Configuration
host: "0.0.0.0"
port: 8000
reload: true  # Development mode
```

### File Settings
```python
# File Storage
upload_dir: "uploads"
reports_dir: "reports"
max_file_size: 100MB
allowed_extensions: [".csv", ".xlsx", ".xls", ".parquet"]
```

## 🚀 Deployment

### Development
```bash
python web/main.py
```

### Production
```bash
gunicorn web.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "web/main.py"]
```

## 🔮 Future Enhancements

- **User Authentication**: Login/signup system
- **Session Management**: User session handling
- **Database Integration**: Persistent job storage
- **Real-time Updates**: WebSocket-based progress updates
- **Advanced UI**: React/Vue.js frontend
- **API Documentation**: Interactive API docs (Swagger)
- **Rate Limiting**: API usage limits
- **Caching**: Result caching for performance

## 🧪 Testing

### API Testing
```bash
# Test endpoints
pytest tests/test_web/
```

### Manual Testing
```bash
# Start server
python web/main.py

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/analyze \
  -F "file=@test_data.csv" \
  -F "target_column=target"
```

## 📚 Related Documentation

- [API Reference](../docs/api-reference.md)
- [Web Interface Guide](../docs/web-interface.md)
- [Deployment Guide](../docs/deployment.md)
- [API Testing](../docs/api-testing.md)

---

**Web Directory** - The user interface and API layer of DataScribe
