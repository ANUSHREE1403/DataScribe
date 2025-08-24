"""
DataScribe Reports Package

This package contains the report generation engine for various output formats.
"""

from .report_generator import DataScribeReportGenerator, generate_reports

__all__ = [
    'DataScribeReportGenerator',
    'generate_reports'
]
