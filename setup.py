#!/usr/bin/env python3
"""
Qwen3 训练框架安装脚本
"""

from setuptools import setup, find_packages
import os

# 读取 README 文件
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# 读取 requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="qwen3-trainer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="一个基于 Huggingface 的 Qwen3 文本生成模型训练框架",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/qwen3-trainer",
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
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "web": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qwen3-train=scripts.train:main",
            "qwen3-infer=scripts.inference:main",
            "qwen3-eval=scripts.evaluate:main",
            "qwen3-deploy=scripts.deploy:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["configs/*.yaml", "configs/*.json"],
    },
    keywords="qwen3 llm finetuning training huggingface transformers",
    project_urls={
        "Documentation": "https://github.com/yourusername/qwen3-trainer/blob/main/README.md",
        "Source": "https://github.com/yourusername/qwen3-trainer",
        "Tracker": "https://github.com/yourusername/qwen3-trainer/issues",
    },
) 