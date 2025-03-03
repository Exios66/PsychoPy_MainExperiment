#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="psychopy-webgazer",
    version="0.1.0",
    description="Integration of WebGazer.js with PsychoPy for web-based eye tracking",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/PsychoPy_MainExperiment",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask>=3.1.0",
        "flask-cors>=5.0.1",
        "websockets>=15.0",
        "numpy>=2.2.3",
        "pandas>=2.2.3",
        "matplotlib>=3.10.1",
        "scipy>=1.15.2",
        "psychopy>=2023.1.0",
        "opencv-python>=4.5.0",
        "pillow>=11.0.0",
        "pyopengl>=3.1.0",
        "pyglet>=1.4.0",
        "pyyaml>=5.1",
        "requests>=2.22.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
    ],
    python_requires=">=3.9",
) 