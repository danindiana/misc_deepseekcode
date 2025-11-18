#!/usr/bin/env python3
"""
Setup script for Inter-System Communication Language
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="inter-system-communication",
    version="1.0.0",
    author="Inter-System Communication Language Team",
    author_email="",
    description="Multi-language framework for RNN/LSTM encoder-decoder architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danindiana/misc_deepseekcode",
    project_urls={
        "Bug Tracker": "https://github.com/danindiana/misc_deepseekcode/issues",
        "Documentation": "https://github.com/danindiana/misc_deepseekcode/tree/main/docs",
        "Source Code": "https://github.com/danindiana/misc_deepseekcode",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="examples/python"),
    package_dir={"": "examples/python"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "iscl=inter_system_communication:main",
        ],
    },
)
