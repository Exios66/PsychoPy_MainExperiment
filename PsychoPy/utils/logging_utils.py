"""
Logging utilities for the PsychoPy GazeTracking application.

This module provides functions for setting up loggers, creating log files,
and logging messages. It is designed to work without dependencies on PsychoPy.
"""

import os
import logging
import datetime
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with console and optional file handlers.
    
    Parameters
    ----------
    name : str
        Name of the logger
    log_file : str or Path, optional
        Path to the log file
    level : int, optional
        Logging level (default: logging.INFO)
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def create_log_file(directory, prefix="log", extension=".log"):
    """
    Create a log file with a timestamp in the filename.
    
    Parameters
    ----------
    directory : str or Path
        Directory to create the log file in
    prefix : str, optional
        Prefix for the log file name (default: "log")
    extension : str, optional
        Extension for the log file (default: ".log")
        
    Returns
    -------
    str
        Path to the created log file
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create log file path
    log_file = os.path.join(directory, f"{prefix}_{timestamp}{extension}")
    
    return log_file

def log_message(logger, level, message):
    """
    Log a message with the specified level.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger to use
    level : str
        Logging level ('debug', 'info', 'warning', 'error', 'critical')
    message : str
        Message to log
    """
    if level.lower() == 'debug':
        logger.debug(message)
    elif level.lower() == 'info':
        logger.info(message)
    elif level.lower() == 'warning':
        logger.warning(message)
    elif level.lower() == 'error':
        logger.error(message)
    elif level.lower() == 'critical':
        logger.critical(message)
    else:
        logger.info(message)

def close_log_file(logger):
    """
    Close all file handlers for a logger.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger to close file handlers for
    """
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler) 