"""
IntentLog: Version Control for Human Reasoning

This file is a thin compatibility shim. All project metadata (version,
description, dependencies, optional extras, entry points, classifiers)
lives in pyproject.toml, which is the single source of truth. Keeping
this file empty of metadata prevents the two from drifting apart; it
exists only so that tooling which still expects a setup.py keeps working.
"""

from setuptools import setup

setup()
