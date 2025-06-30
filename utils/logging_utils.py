import logging
import os
from datetime import datetime

class LoggingService:
    def __init__(self, log_level=logging.INFO):
        """Initialize the logger with the specified log level"""
        # Create logs directory if it doesn't exist
        self.log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = self._setup_logger(log_level)
    
    def _setup_logger(self, log_level):
        """Configure and return a logger with file and console handlers"""
        # Create log filename with current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        log_file = os.path.join(self.log_dir, f"rag_app_{current_date}.log")
        
        # Configure logger
        logger = logging.getLogger("rag_app")
        logger.setLevel(log_level)
        
        # Clear any existing handlers to prevent duplicates
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file, 'w', 'utf-8')
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message):
        """Log an informational message"""
        self.logger.info(message)
    
    def debug(self, message):
        """Log a debug message"""
        self.logger.debug(message)
    
    def warning(self, message):
        """Log a warning message"""
        self.logger.warning(message)
    
    def error(self, message, exc_info=False):
        """Log an error message, optionally with exception info"""
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message, exc_info=False):
        """Log a critical message, optionally with exception info"""
        self.logger.critical(message, exc_info=exc_info)
    
    

# Create a singleton instance
logger = LoggingService()