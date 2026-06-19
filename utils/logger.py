import logging
import os
import sys

def get_logger(name: str = "CyberSecurityApp"):
    """
    Returns a configured logger that writes to both console and app.log.
    """
    logger = logging.getLogger(name)
    
    # Only configure if it hasn't been configured yet to avoid duplicate handlers
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

# Create a default instance that can be imported directly
app_logger = get_logger()
