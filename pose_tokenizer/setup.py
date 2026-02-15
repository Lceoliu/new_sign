#!/usr/bin/env python3
"""Setup script for Pose Tokenizer package."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="pose-tokenizer",
    version="0.1.0",
    author="Pose Tokenizer Team",
    author_email="contact@example.com",
    description="A PyTorch implementation of pose tokenizer using ST-GCN and RVQ",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/pose-tokenizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
        ],
        "export": [
            "onnx>=1.12.0",
            "onnxruntime>=1.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "pose-tokenizer-train=pose_tokenizer.scripts.train:main",
            "pose-tokenizer-eval=pose_tokenizer.scripts.evaluate:main",
            "pose-tokenizer-export=pose_tokenizer.scripts.export:main",
            "pose-tokenizer-demo=pose_tokenizer.demo:run_quick_training_demo",
        ],
    },
    include_package_data=True,
    package_data={
        "pose_tokenizer": [
            "configs/*.yaml",
        ],
    },
    zip_safe=False,
)