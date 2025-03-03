#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script to verify PsychoPy installation for WebGazer integration
"""

try:
    import psychopy
    print(f"PsychoPy version: {psychopy.__version__}")
    print("PsychoPy imported successfully!")
    
    # Test importing only the core modules we need
    from psychopy import core, data, logging
    print("Core PsychoPy modules imported successfully!")
    
    # Test Flask for WebGazer bridge
    import flask
    from flask_cors import CORS
    print(f"Flask version: {flask.__version__}")
    print("Flask and Flask-CORS imported successfully!")
    
    print("\nCore functionality test successful! Your environment is ready for WebGazer integration.")
    
except ImportError as e:
    print(f"Error importing: {e}") 