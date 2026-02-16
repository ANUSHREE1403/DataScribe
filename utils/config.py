from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    # App Configuration
    app_name: str = "DataScribe"
    app_subtitle: str = "Democratizing Data Analysis: Automated EDA with Human-Readable Insights"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True
    
    # Database Configuration
    # Default: local SQLite file for easy development
    database_url: str = "sqlite:///./datascribe.db"
    redis_url: str = "redis://localhost:6379"
    
    # Authentication
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # File Storage
    upload_dir: str = "uploads"
    reports_dir: str = "reports"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = [".csv", ".xlsx", ".xls", ".parquet"]
    
    # EDA Configuration
    max_rows_for_analysis: int = 100000
    max_columns_for_analysis: int = 100
    correlation_threshold: float = 0.7
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    
    # Report Generation
    default_report_sections: List[str] = [
        "overview", "missing_values", "outliers", "distributions", 
        "correlations", "target_analysis", "recommendations"
    ]
    
    # Export Formats
    enable_pdf_export: bool = True
    enable_excel_export: bool = True
    enable_code_export: bool = True
    
    # ML Models
    enable_baseline_models: bool = True
    default_test_size: float = 0.2
    cross_validation_folds: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.reports_dir, exist_ok=True)
