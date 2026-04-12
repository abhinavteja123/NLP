"""
IQAS Setup Configuration
=========================
"""

from setuptools import setup, find_packages

setup(
    name="iqas",
    version="1.0.0",
    description="Intelligent Question Answering System — NLP-powered document QA",
    author="Abhinav Teja",
    author_email="",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.32.0",
        "spacy>=3.7.0",
        "sentence-transformers>=2.6.0",
        "faiss-cpu>=1.7.4",
        "rank-bm25>=0.2.2",
        "pymupdf>=1.23.0",
        "python-docx>=1.1.0",
        "transformers>=4.38.0",
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "plotly>=5.19.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "loguru>=0.7.0",
        "pydantic>=2.6.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0"],
    },
    entry_points={
        "console_scripts": [
            "iqas=app.main:main",
        ],
    },
)
