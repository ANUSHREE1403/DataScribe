"""
DataScribe Setup

Setup script for DataScribe - Automated EDA Platform
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="datascribe",
    version="1.0.0",
    author="DataScribe Team",
    author_email="team@datascribe.ai",
    description="Democratizing Data Analysis: Automated EDA with Human-Readable Insights",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/datascribe",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "mkdocs>=1.0",
            "mkdocs-material>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "datascribe=run:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="data-analysis, eda, automated, visualization, reporting, datascience",
    project_urls={
        "Bug Reports": "https://github.com/your-org/datascribe/issues",
        "Source": "https://github.com/your-org/datascribe",
        "Documentation": "https://datascribe.readthedocs.io/",
    },
)
