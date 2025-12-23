"""
IntentLog: Version Control for Human Reasoning
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="intentlog",
    version="0.1.0",
    author="IntentLog Contributors",
    description="Version control system for human reasoning and intent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kase1111-hash/IntentLog",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Version Control",
        "License :: Other/Proprietary License",  # CC BY-SA 4.0
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    license="CC-BY-SA-4.0",
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "intentlog=intentlog.cli:main",
            "ilog=intentlog.cli:main",
        ],
    },
)
