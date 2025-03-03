#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal test script to verify PsychoPy core functionality
"""

try:
    # Import only the most basic PsychoPy components
    import psychopy
    print(f"PsychoPy version: {psychopy.__version__}")
    
    # Test Flask for WebGazer bridge
    import flask
    from flask_cors import CORS
    print(f"Flask version: {flask.__version__}")
    print("Flask and Flask-CORS imported successfully!")
    
    print("\nBasic imports successful! Your environment has the essential packages for WebGazer integration.")
    
except ImportError as e:
    print(f"Error importing: {e}") 