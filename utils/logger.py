# utils/logger.py - Professional logging system
import logging
import sys
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

def setup_logger(name='crypto_bot', level=logging.INFO, log_to_file=True):
    """
    Set up comprehensive logging system
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler (what you see in terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    if log_to_file:
        try:
            # Create logs directory
            os.makedirs('logs', exist_ok=True)
            
            # Main log file (rotates daily)
            main_log_handler = TimedRotatingFileHandler(
                'logs/crypto_bot.log',
                when='midnight',
                interval=1,
                backupCount=30,
                encoding='utf-8'
            )
            main_log_handler.setLevel(logging.INFO)
            main_log_handler.setFormatter(detailed_formatter)
            logger.addHandler(main_log_handler)
            
            # Error log file (rotates when it gets too big)
            error_log_handler = RotatingFileHandler(
                'logs/crypto_bot_errors.log',
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5,
                encoding='utf-8'
            )
            error_log_handler.setLevel(logging.WARNING)
            error_log_handler.setFormatter(detailed_formatter)
            logger.addHandler(error_log_handler)
            
            # Trading log file (for trade records)
            trade_log_handler = RotatingFileHandler(
                'logs/trades.log',
                maxBytes=50*1024*1024,  # 50 MB
                backupCount=10,
                encoding='utf-8'
            )
            trade_log_handler.setLevel(logging.INFO)
            trade_log_formatter = logging.Formatter(
                '%(asctime)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            trade_log_handler.setFormatter(trade_log_formatter)
            
            # Create separate logger for trades
            trade_logger = logging.getLogger('trades')
            trade_logger.setLevel(logging.INFO)
            if not trade_logger.handlers:
                trade_logger.addHandler(trade_log_handler)
            
        except Exception as e:
            logger.warning(f"Could not create file loggers: {e}")
    
    return logger

def get_trade_logger():
    """Get the dedicated trade logger"""
    return logging.getLogger('trades')