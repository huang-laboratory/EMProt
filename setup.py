#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages

import emprot

project_root = os.path.join(os.path.realpath(os.path.dirname(__file__)), "emprot")

setup(
    name="emprot",
    entry_points={
        "console_scripts": [
            "emprot = emprot.__main__:main",
        ],
    },
    packages=find_packages(),
    version=emprot.__version__,
)
