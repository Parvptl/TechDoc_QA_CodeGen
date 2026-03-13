from setuptools import setup, find_packages

setup(
    name="ds_mentor",
    version="1.0.0",
    description="Data Science Mentor QA System — NLP Course Project",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
)