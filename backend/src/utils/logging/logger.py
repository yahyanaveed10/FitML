import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional

from pythonjsonlogger import jsonlogger

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging"""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add environment information
        log_record['environment'] = os.getenv('ENV', 'development')


def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name and configuration.
    
    Args:
        name: The name of the logger
        log_level: Optional log level override
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set log level from environment or parameter
    level = log_level or os.getenv('LOG_LEVEL', 'INFO')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Create formatter
    formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(module)s %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.handlers = []  # Remove existing handlers
    logger.addHandler(handler)
    
    return logger


# Create file handler for logging to files
def setup_file_logging(logger: logging.Logger, filename: str = 'app.log') -> None:
    """
    Set up file logging for the given logger.
    
    Args:
        logger: The logger to configure
        filename: The log file name
    """
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(module)s %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
