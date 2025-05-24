# config/settings.py - Fixed configuration with working symbols
import os
from typing import List

class Config:
    """Complete configuration settings for the trading bot"""
    
    # ================================
    # BASIC SETTINGS
    # ================================
    
    # Trading pairs to monitor (FIXED - removed problematic MATIC)
    SYMBOLS: List[str] = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 
        'SOL/USDT', 'DOT/USDT', 'AVAX/USDT', 'XRP/USDT'  # Replaced MATIC with XRP
    ]
    
    # Portfolio settings
    INITIAL_BALANCE: float = 10000.0  # Starting with $10K
    MAX_POSITION_SIZE: float = 0.1    # Max 10% of balance per trade
    MAX_TOTAL_EXPOSURE: float = 0.8   # Max 80% of balance invested
    
    # ================================
    # EXCHANGE SETTINGS
    # ================================
    
    EXCHANGE: str = 'binance'
    SANDBOX_MODE: bool = True  # Paper trading mode
    
    # API credentials (set via environment variables)
    BINANCE_API_KEY: str = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY: str = os.getenv('BINANCE_SECRET_KEY', '')
    
    # ================================
    # RISK MANAGEMENT
    # ================================
    
    # Loss limits
    STOP_LOSS_PERCENT: float = 0.05    # 5% stop loss per trade
    MAX_DAILY_LOSS: float = 0.05       # 5% max daily loss
    MAX_DRAWDOWN: float = 0.15         # 15% max drawdown before pause
    
    # Position limits
    MAX_OPEN_POSITIONS: int = 5        # Max 5 positions at once
    MIN_ORDER_SIZE: float = 10.0       # Minimum $10 order
    
    # ================================
    # TECHNICAL ANALYSIS SETTINGS
    # ================================
    
    # Indicator periods
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 30
    RSI_OVERBOUGHT: float = 70
    
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD_DEV: float = 2.0
    
    # Moving averages
    SMA_SHORT: int = 10
    SMA_LONG: int = 20
    EMA_SHORT: int = 10
    EMA_LONG: int = 20
    
    # ================================
    # GRID TRADING SETTINGS
    # ================================
    
    # Grid parameters
    GRID_SIZE: int = 8                 # 8 buy + 8 sell orders
    GRID_SPACING_PERCENT: float = 0.5  # 0.5% between grid levels
    BASE_ORDER_SIZE: float = 100       # $100 per grid order
    MAX_GRID_INVESTMENT: float = 1600  # Max $1600 per grid (8 x $200)
    
    # Grid conditions
    GRID_RSI_MIN: float = 35          # Don't start grid if RSI < 35
    GRID_RSI_MAX: float = 65          # Don't start grid if RSI > 65
    GRID_MIN_VOLATILITY: float = 0.02  # Minimum volatility for grid
    
    # ================================
    # AI/ML SETTINGS
    # ================================
    
    # Prediction settings
    PREDICTION_TIMEFRAMES: List[str] = ['5m', '15m', '1h', '4h']
    PREDICTION_CONFIDENCE_THRESHOLD: float = 0.7
    
    # Sentiment analysis
    SENTIMENT_WEIGHT: float = 0.3      # Weight of sentiment in final decision
    NEWS_LOOKBACK_HOURS: int = 24      # Hours of news to analyze
    
    # Model settings
    MODEL_RETRAIN_INTERVAL: int = 24   # Retrain models every 24 hours
    HISTORICAL_DATA_PERIODS: int = 200 # Periods of data for training
    
    # ================================
    # DATA COLLECTION SETTINGS
    # ================================
    
    # Update intervals (seconds)
    PRICE_UPDATE_INTERVAL: int = 2     # Update prices every 2 seconds
    HISTORICAL_UPDATE_INTERVAL: int = 60 # Update historical data every minute
    ORDER_BOOK_UPDATE_INTERVAL: int = 5  # Update order books every 5 seconds
    SENTIMENT_UPDATE_INTERVAL: int = 300 # Update sentiment every 5 minutes
    
    # Data retention
    MAX_PRICE_HISTORY: int = 1000      # Keep last 1000 price updates
    MAX_NEWS_ITEMS: int = 100          # Keep last 100 news items
    
    # ================================
    # STRATEGY SETTINGS
    # ================================
    
    # Strategy execution
    STRATEGY_UPDATE_INTERVAL: int = 30  # Run strategies every 30 seconds
    SIGNAL_STRENGTH_THRESHOLD: float = 1.5  # INCREASED threshold to reduce trades
    
    # Strategy weights (must sum to 1.0)
    TECHNICAL_WEIGHT: float = 0.4      # Technical analysis weight
    GRID_WEIGHT: float = 0.3           # Grid trading weight
    AI_WEIGHT: float = 0.2             # AI prediction weight
    SENTIMENT_WEIGHT_STRATEGY: float = 0.1  # Sentiment analysis weight
    
    # ================================
    # LOGGING AND MONITORING
    # ================================
    
    # Logging levels
    LOG_LEVEL: str = 'INFO'
    LOG_TO_FILE: bool = True
    LOG_ROTATION: str = 'daily'
    
    # Performance monitoring
    PERFORMANCE_LOG_INTERVAL: int = 300 # Log performance every 5 minutes
    SAVE_TRADES_TO_FILE: bool = True
    
    # ================================
    # NOTIFICATION SETTINGS
    # ================================
    
    # Alerts (placeholders for future implementation)
    ENABLE_TRADE_NOTIFICATIONS: bool = True
    ENABLE_ERROR_NOTIFICATIONS: bool = True
    NOTIFICATION_THRESHOLD: float = 0.02  # Notify on 2%+ moves
    
    # ================================
    # BACKTESTING SETTINGS
    # ================================
    
    # Backtesting parameters
    BACKTEST_START_DATE: str = '2024-01-01'
    BACKTEST_END_DATE: str = '2024-12-31'
    BACKTEST_INITIAL_BALANCE: float = 10000.0
    
    # ================================
    # ADVANCED SETTINGS
    # ================================
    
    # Rate limiting
    EXCHANGE_RATE_LIMIT: bool = True
    MAX_REQUESTS_PER_MINUTE: int = 100
    
    # Error handling
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 5.0          # Seconds to wait between retries
    
    # Performance optimization
    ENABLE_CACHING: bool = True
    CACHE_EXPIRY: int = 60            # Cache expires after 60 seconds
    
    # ================================
    # VALIDATION METHODS
    # ================================
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        try:
            # Check that weights sum to 1.0
            total_weight = (self.TECHNICAL_WEIGHT + self.GRID_WEIGHT + 
                           self.AI_WEIGHT + self.SENTIMENT_WEIGHT_STRATEGY)
            if abs(total_weight - 1.0) > 0.01:
                print(f"❌ Strategy weights must sum to 1.0, got {total_weight}")
                return False
            
            # Check balance and position sizes
            if self.MAX_POSITION_SIZE > 1.0:
                print("❌ MAX_POSITION_SIZE cannot exceed 1.0 (100%)")
                return False
            
            if self.MAX_TOTAL_EXPOSURE > 1.0:
                print("❌ MAX_TOTAL_EXPOSURE cannot exceed 1.0 (100%)")
                return False
            
            # Check risk parameters
            if self.STOP_LOSS_PERCENT <= 0 or self.STOP_LOSS_PERCENT > 0.5:
                print("❌ STOP_LOSS_PERCENT must be between 0 and 0.5")
                return False
            
            # Check symbols
            if not self.SYMBOLS or len(self.SYMBOLS) == 0:
                print("❌ SYMBOLS list cannot be empty")
                return False
            
            print("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def get_summary(self) -> List[str]:
        """Get configuration summary"""
        return [
            f"Trading Symbols: {len(self.SYMBOLS)} pairs",
            f"Initial Balance: ${self.INITIAL_BALANCE:,.2f}",
            f"Max Position Size: {self.MAX_POSITION_SIZE*100:.1f}%",
            f"Grid Size: {self.GRID_SIZE} orders per side",
            f"Grid Spacing: {self.GRID_SPACING_PERCENT}%",
            f"Stop Loss: {self.STOP_LOSS_PERCENT*100:.1f}%",
            f"Max Daily Loss: {self.MAX_DAILY_LOSS*100:.1f}%",
            f"Signal Threshold: {self.SIGNAL_STRENGTH_THRESHOLD} (higher = fewer trades)",
            f"Mode: {'Paper Trading' if self.SANDBOX_MODE else 'Live Trading'}"
        ]