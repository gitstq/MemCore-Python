#!/usr/bin/env python3
"""
MemCore - Lightweight AI Agent Persistent Memory Engine
Setup script for installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ""

setup(
    name="memcore",
    version="1.0.0",
    author="gitstq",
    author_email="",
    description="Lightweight AI Agent Persistent Memory Engine - Zero Dependencies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gitstq/MemCore",
    py_modules=["memcore"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "memcore=memcore:run_cli",
        ],
    },
    keywords="ai agent memory persistent mcp claude cursor codex semantic-search",
    project_urls={
        "Bug Reports": "https://github.com/gitstq/MemCore/issues",
        "Source": "https://github.com/gitstq/MemCore",
    },
)
