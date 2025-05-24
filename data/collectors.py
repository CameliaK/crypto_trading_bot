# data/collectors.py - Real-time data collection system
import ccxt
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

class RealTimeCollector:
    """
    Complete data collection system for crypto trading
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Exchange connection
        self.exchange = None
        
        # Data storage
        self.price_data = {}           # Latest price data
        self.historical_data = {}      # OHLCV historical data
        self.order_books = {}          # Order book data
        self.volatility_data = {}      # Volatility calculations
        self.volume_data = {}          # Volume analysis
        self.sentiment_data = {}       # News sentiment
        
        # Initialize exchange
        self.setup_exchange()
    
    def setup_exchange(self):
        """Setup exchange connection"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.config.BINANCE_API_KEY,
                'secret': self.config.BINANCE_SECRET_KEY,
                'sandbox': self.config.SANDBOX_MODE,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            
            # Test connection
            self.exchange.load_markets()
            self.logger.info("✅ Connected to Binance exchange")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to exchange: {e}")
            self.logger.info("📊 Continuing in simulation mode")
            # Continue in simulation mode
            self.exchange = None
    
    async def collect_prices(self):
        """Collect real-time price data"""
        while True:
            try:
                if self.exchange:
                    # Real exchange data
                    tickers = self.exchange.fetch_tickers(self.config.SYMBOLS)
                    
                    for symbol, ticker in tickers.items():
                        self.price_data[symbol] = {
                            'price': ticker['last'],
                            'bid': ticker['bid'],
                            'ask': ticker['ask'],
                            'volume': ticker['baseVolume'],
                            'change': ticker['percentage'] or 0,
                            'high_24h': ticker['high'],
                            'low_24h': ticker['low'],
                            'timestamp': datetime.now()
                        }
                else:
                    # Simulation mode
                    await self._simulate_price_data()
                
                await asyncio.sleep(self.config.PRICE_UPDATE_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"❌ Error collecting prices: {e}")
                # Fall back to simulation if exchange fails
                await self._simulate_price_data()
                await asyncio.sleep(10)
    
    async def _simulate_price_data(self):
        """Simulate realistic price data when exchange is not available"""
        # Initialize prices if not set
        if not self.price_data:
            base_prices = {
                'BTC/USDT': 43250.0,
                'ETH/USDT': 2650.0,
                'BNB/USDT': 308.0,
                'ADA/USDT': 0.49,
                'SOL/USDT': 98.5,
                'DOT/USDT': 7.2,
                'AVAX/USDT': 38.5,
                'MATIC/USDT': 0.85
            }
            
            for symbol in self.config.SYMBOLS:
                if symbol in base_prices:
                    self.price_data[symbol] = {
                        'price': base_prices[symbol],
                        'bid': base_prices[symbol] * 0.999,
                        'ask': base_prices[symbol] * 1.001,
                        'volume': np.random.uniform(1000, 5000),
                        'change': 0.0,
                        'high_24h': base_prices[symbol] * 1.05,
                        'low_24h': base_prices[symbol] * 0.95,
                        'timestamp': datetime.now()
                    }
        
        # Update prices with realistic movements
        for symbol in self.config.SYMBOLS:
            if symbol in self.price_data:
                current_price = self.price_data[symbol]['price']
                
                # Realistic price movement (-0.5% to +0.5% per update)
                change_percent = np.random.normal(0, 0.002)
                new_price = current_price * (1 + change_percent)
                
                # Update all price data
                self.price_data[symbol].update({
                    'price': new_price,
                    'bid': new_price * 0.999,
                    'ask': new_price * 1.001,
                    'volume': np.random.uniform(1000, 5000),
                    'change': change_percent * 100,
                    'timestamp': datetime.now()
                })
    
    async def collect_historical_data(self):
        """Collect historical OHLCV data"""
        while True:
            try:
                for symbol in self.config.SYMBOLS:
                    if self.exchange:
                        # Real historical data
                        ohlcv = self.exchange.fetch_ohlcv(
                            symbol, 
                            '5m', 
                            limit=self.config.HISTORICAL_DATA_PERIODS
                        )
                        
                        df = pd.DataFrame(
                            ohlcv, 
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                    else:
                        # Simulated historical data
                        df = self._generate_simulated_ohlcv(symbol)
                    
                    self.historical_data[symbol] = df
                    
                    # Calculate additional metrics
                    self._calculate_volatility(symbol, df)
                    self._calculate_volume_metrics(symbol, df)
                
                await asyncio.sleep(self.config.HISTORICAL_UPDATE_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"❌ Error collecting historical data: {e}")
                # Generate simulated data on error
                for symbol in self.config.SYMBOLS:
                    if symbol not in self.historical_data:
                        self.historical_data[symbol] = self._generate_simulated_ohlcv(symbol)
                await asyncio.sleep(30)
    
    def _generate_simulated_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Generate realistic OHLCV data for simulation"""
        periods = self.config.HISTORICAL_DATA_PERIODS
        
        # Get current price or use default
        if symbol in self.price_data:
            current_price = self.price_data[symbol]['price']
        else:
            default_prices = {
                'BTC/USDT': 43250.0, 'ETH/USDT': 2650.0, 'BNB/USDT': 308.0,
                'ADA/USDT': 0.49, 'SOL/USDT': 98.5, 'DOT/USDT': 7.2,
                'AVAX/USDT': 38.5, 'MATIC/USDT': 0.85
            }
            current_price = default_prices.get(symbol, 100.0)
        
        # Generate historical data
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        start_time = datetime.now() - timedelta(minutes=periods * 5)
        price = current_price * 0.95  # Start slightly lower to show some growth
        
        for i in range(periods):
            # Realistic price movement with some trend
            trend = np.random.normal(0, 0.001)
            volatility = np.random.normal(0, 0.005)
            
            open_price = price
            close_price = open_price * (1 + trend + volatility)
            
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.002)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.002)))
            
            volume = np.random.uniform(500, 2000) * (1 + abs(volatility) * 10)
            
            timestamps.append(start_time + timedelta(minutes=i * 5))
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
            
            price = close_price
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
    
    def _calculate_volatility(self, symbol: str, df: pd.DataFrame):
        """Calculate volatility metrics"""
        try:
            if len(df) < 20:
                self.volatility_data[symbol] = 0.02
                return
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            # Various volatility measures
            volatility_5m = returns.rolling(window=12).std().iloc[-1]  # 1-hour volatility
            volatility_1h = returns.rolling(window=60).std().iloc[-1]   # 5-hour volatility
            volatility_daily = returns.rolling(window=288).std().iloc[-1] # Daily volatility
            
            self.volatility_data[symbol] = {
                '5m': volatility_5m if not pd.isna(volatility_5m) else 0.02,
                '1h': volatility_1h if not pd.isna(volatility_1h) else 0.02,
                'daily': volatility_daily if not pd.isna(volatility_daily) else 0.02
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating volatility for {symbol}: {e}")
            self.volatility_data[symbol] = 0.02
    
    def _calculate_volume_metrics(self, symbol: str, df: pd.DataFrame):
        """Calculate volume-based metrics"""
        try:
            if len(df) < 20:
                return
            
            # Volume moving averages
            volume_sma_20 = df['volume'].rolling(window=20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            # Volume relative to average
            volume_ratio = current_volume / volume_sma_20 if volume_sma_20 > 0 else 1.0
            
            # Price-volume correlation
            price_change = df['close'].pct_change()
            volume_change = df['volume'].pct_change()
            correlation = price_change.corr(volume_change)
            
            self.volume_data[symbol] = {
                'current': current_volume,
                'average_20': volume_sma_20,
                'ratio': volume_ratio,
                'price_correlation': correlation if not pd.isna(correlation) else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating volume metrics for {symbol}: {e}")
    
    async def collect_order_books(self):
        """Collect order book data"""
        while True:
            try:
                for symbol in self.config.SYMBOLS:
                    if self.exchange:
                        order_book = self.exchange.fetch_order_book(symbol, limit=20)
                        
                        self.order_books[symbol] = {
                            'bids': order_book['bids'][:10],
                            'asks': order_book['asks'][:10],
                            'spread': order_book['asks'][0][0] - order_book['bids'][0][0] if order_book['asks'] and order_book['bids'] else 0,
                            'timestamp': datetime.now()
                        }
                    else:
                        # Simulated order book
                        if symbol in self.price_data:
                            price = self.price_data[symbol]['price']
                            spread = price * 0.001  # 0.1% spread
                            
                            self.order_books[symbol] = {
                                'bids': [[price - spread/2, 100]],
                                'asks': [[price + spread/2, 100]],
                                'spread': spread,
                                'timestamp': datetime.now()
                            }
                
                await asyncio.sleep(self.config.ORDER_BOOK_UPDATE_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"❌ Error collecting order books: {e}")
                await asyncio.sleep(15)
    
    async def collect_sentiment(self):
        """Collect market sentiment from news and social media"""
        while True:
            try:
                # Simulate sentiment collection
                # In a real implementation, this would connect to news APIs
                for symbol in self.config.SYMBOLS:
                    # Simulate sentiment score between -1 and 1
                    sentiment_score = np.random.normal(0, 0.3)
                    sentiment_score = max(-1.0, min(1.0, sentiment_score))
                    
                    self.sentiment_data[symbol] = {
                        'score': sentiment_score,
                        'confidence': np.random.uniform(0.5, 0.9),
                        'news_count': np.random.randint(5, 20),
                        'timestamp': datetime.now()
                    }
                
                await asyncio.sleep(self.config.SENTIMENT_UPDATE_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"❌ Error collecting sentiment: {e}")
                await asyncio.sleep(60)
    
    # ================================
    # DATA ACCESS METHODS
    # ================================
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        if symbol in self.price_data:
            return self.price_data[symbol]['price']
        return None
    
    def get_price_data(self, symbol: str) -> Optional[Dict]:
        """Get complete price data for a symbol"""
        return self.price_data.get(symbol)
    
    def get_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data"""
        return self.historical_data.get(symbol)
    
    def get_volatility(self, symbol: str, timeframe: str = '1h') -> float:
        """Get volatility for a symbol and timeframe"""
        if symbol in self.volatility_data:
            if isinstance(self.volatility_data[symbol], dict):
                return self.volatility_data[symbol].get(timeframe, 0.02)
            else:
                return self.volatility_data[symbol]
        return 0.02
    
    def get_volume_data(self, symbol: str) -> Optional[Dict]:
        """Get volume analysis data"""
        return self.volume_data.get(symbol)
    
    def get_order_book(self, symbol: str) -> Optional[Dict]:
        """Get order book data"""
        return self.order_books.get(symbol)
    
    def get_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get sentiment data"""
        return self.sentiment_data.get(symbol)
    
    def get_market_summary(self) -> Dict:
        """Get complete market summary"""
        summary = {
            'symbols_monitored': len(self.config.SYMBOLS),
            'data_points': len(self.price_data),
            'avg_volatility': 0,
            'market_sentiment': 0,
            'total_volume': 0,
            'last_update': datetime.now()
        }
        
        if self.price_data:
            # Calculate average volatility
            volatilities = []
            sentiments = []
            volumes = []
            
            for symbol in self.config.SYMBOLS:
                if symbol in self.volatility_data:
                    vol = self.get_volatility(symbol)
                    volatilities.append(vol)
                
                if symbol in self.sentiment_data:
                    sent = self.sentiment_data[symbol]['score']
                    sentiments.append(sent)
                
                if symbol in self.price_data:
                    vol = self.price_data[symbol]['volume']
                    volumes.append(vol)
            
            if volatilities:
                summary['avg_volatility'] = np.mean(volatilities)
            if sentiments:
                summary['market_sentiment'] = np.mean(sentiments)
            if volumes:
                summary['total_volume'] = sum(volumes)
        
        return summary
    
    def is_data_ready(self) -> bool:
        """Check if we have sufficient data to start trading"""
        required_data = len(self.config.SYMBOLS)
        
        return (len(self.price_data) >= required_data and 
                len(self.historical_data) >= required_data)