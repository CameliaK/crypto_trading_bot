# technical/indicators.py - Complete technical analysis system
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

class TechnicalAnalyzer:
    """
    Comprehensive technical analysis with multiple indicators
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            period = period or self.config.RSI_PERIOD
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
            
        except Exception as e:
            self.logger.error(f"❌ RSI calculation error: {e}")
            return 50.0
    
    def calculate_macd(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            fast_ema = prices.ewm(span=self.config.MACD_FAST).mean()
            slow_ema = prices.ewm(span=self.config.MACD_SLOW).mean()
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=self.config.MACD_SIGNAL).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0.0,
                'signal': signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0.0,
                'histogram': histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"❌ MACD calculation error: {e}")
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    def calculate_bollinger_bands(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            period = self.config.BOLLINGER_PERIOD
            std_dev = self.config.BOLLINGER_STD_DEV
            
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            current_price = prices.iloc[-1]
            upper = upper_band.iloc[-1] if not pd.isna(upper_band.iloc[-1]) else current_price * 1.02
            middle = sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else current_price
            lower = lower_band.iloc[-1] if not pd.isna(lower_band.iloc[-1]) else current_price * 0.98
            
            # Calculate band position (0 = lower band, 1 = upper band)
            band_position = (current_price - lower) / (upper - lower) if upper != lower else 0.5
            
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower,
                'position': band_position,
                'width': (upper - lower) / middle if middle > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Bollinger Bands calculation error: {e}")
            current = prices.iloc[-1]
            return {
                'upper': current * 1.02,
                'middle': current,
                'lower': current * 0.98,
                'position': 0.5,
                'width': 0.04
            }
    
    def calculate_moving_averages(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate various moving averages"""
        try:
            current_price = prices.iloc[-1]
            
            sma_short = prices.rolling(window=self.config.SMA_SHORT).mean().iloc[-1]
            sma_long = prices.rolling(window=self.config.SMA_LONG).mean().iloc[-1]
            ema_short = prices.ewm(span=self.config.EMA_SHORT).mean().iloc[-1]
            ema_long = prices.ewm(span=self.config.EMA_LONG).mean().iloc[-1]
            
            return {
                'sma_short': sma_short if not pd.isna(sma_short) else current_price,
                'sma_long': sma_long if not pd.isna(sma_long) else current_price,
                'ema_short': ema_short if not pd.isna(ema_short) else current_price,
                'ema_long': ema_long if not pd.isna(ema_long) else current_price,
                'sma_cross': 1 if sma_short > sma_long else -1,
                'ema_cross': 1 if ema_short > ema_long else -1,
                'price_vs_sma': (current_price - sma_short) / sma_short if sma_short > 0 else 0,
                'price_vs_ema': (current_price - ema_short) / ema_short if ema_short > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Moving averages calculation error: {e}")
            current = prices.iloc[-1]
            return {
                'sma_short': current, 'sma_long': current,
                'ema_short': current, 'ema_long': current,
                'sma_cross': 0, 'ema_cross': 0,
                'price_vs_sma': 0, 'price_vs_ema': 0
            }
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum-based indicators"""
        try:
            prices = df['close']
            volumes = df['volume']
            
            # Rate of Change (ROC)
            roc_5 = ((prices.iloc[-1] - prices.iloc[-6]) / prices.iloc[-6]) * 100 if len(prices) > 5 else 0
            roc_10 = ((prices.iloc[-1] - prices.iloc[-11]) / prices.iloc[-11]) * 100 if len(prices) > 10 else 0
            
            # Stochastic Oscillator
            high_14 = df['high'].rolling(window=14).max()
            low_14 = df['low'].rolling(window=14).min()
            k_percent = ((prices - low_14) / (high_14 - low_14)) * 100
            k_percent = k_percent.fillna(50)
            d_percent = k_percent.rolling(window=3).mean()
            
            stoch_k = k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else 50.0
            stoch_d = d_percent.iloc[-1] if not pd.isna(d_percent.iloc[-1]) else 50.0
            
            # Williams %R
            williams_r = ((high_14.iloc[-1] - prices.iloc[-1]) / (high_14.iloc[-1] - low_14.iloc[-1])) * -100
            williams_r = williams_r if not pd.isna(williams_r) else -50.0
            
            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            atr = atr if not pd.isna(atr) else prices.iloc[-1] * 0.02
            
            return {
                'roc_5': roc_5,
                'roc_10': roc_10,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d,
                'williams_r': williams_r,
                'atr': atr
            }
            
        except Exception as e:
            self.logger.error(f"❌ Momentum indicators calculation error: {e}")
            return {
                'roc_5': 0, 'roc_10': 0, 'stoch_k': 50, 'stoch_d': 50,
                'williams_r': -50, 'atr': 0
            }
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based indicators"""
        try:
            prices = df['close']
            volumes = df['volume']
            
            # On-Balance Volume (OBV)
            obv = np.where(prices > prices.shift(1), volumes, 
                          np.where(prices < prices.shift(1), -volumes, 0)).cumsum()
            obv_current = obv[-1] if len(obv) > 0 else 0
            
            # Volume Price Trend (VPT)
            vpt = (volumes * ((prices - prices.shift(1)) / prices.shift(1))).cumsum()
            vpt_current = vpt.iloc[-1] if not pd.isna(vpt.iloc[-1]) else 0
            
            # Money Flow Index (MFI)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * volumes
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
            
            mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
            mfi_current = mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else 50.0
            
            # Volume Moving Average
            volume_sma = volumes.rolling(window=20).mean().iloc[-1]
            volume_ratio = volumes.iloc[-1] / volume_sma if volume_sma > 0 else 1.0
            
            return {
                'obv': obv_current,
                'vpt': vpt_current,
                'mfi': mfi_current,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            self.logger.error(f"❌ Volume indicators calculation error: {e}")
            return {'obv': 0, 'vpt': 0, 'mfi': 50, 'volume_ratio': 1.0}
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Comprehensive analysis of a symbol
        """
        if df is None or len(df) < 50:
            return {
                'signal': 'HOLD',
                'strength': 0,
                'rsi': 50,
                'price': 0,
                'reasons': ['Insufficient data']
            }
        
        try:
            prices = df['close']
            current_price = prices.iloc[-1]
            
            # Calculate all indicators
            rsi = self.calculate_rsi(prices)
            macd_data = self.calculate_macd(prices)
            bb_data = self.calculate_bollinger_bands(prices)
            ma_data = self.calculate_moving_averages(prices)
            momentum_data = self.calculate_momentum_indicators(df)
            volume_data = self.calculate_volume_indicators(df)
            
            # Price analysis
            price_change_1h = self._calculate_price_change(prices, 12)  # 12 periods = 1 hour (5min candles)
            price_change_4h = self._calculate_price_change(prices, 48)  # 48 periods = 4 hours
            price_change_24h = self._calculate_price_change(prices, 288) # 288 periods = 24 hours
            
            # Signal generation
            signals = []
            
            # RSI Signals
            if rsi < self.config.RSI_OVERSOLD:
                if rsi < 20:
                    signals.append(('BUY', 'RSI Extremely Oversold', 1.2))
                else:
                    signals.append(('BUY', 'RSI Oversold', 0.8))
            elif rsi > self.config.RSI_OVERBOUGHT:
                if rsi > 80:
                    signals.append(('SELL', 'RSI Extremely Overbought', 1.2))
                else:
                    signals.append(('SELL', 'RSI Overbought', 0.8))
            
            # MACD Signals
            if macd_data['macd'] > macd_data['signal'] and macd_data['histogram'] > 0:
                signals.append(('BUY', 'MACD Bullish Crossover', 0.9))
            elif macd_data['macd'] < macd_data['signal'] and macd_data['histogram'] < 0:
                signals.append(('SELL', 'MACD Bearish Crossover', 0.9))
            
            # Bollinger Bands Signals
            bb_pos = bb_data['position']
            if bb_pos < 0.1:
                signals.append(('BUY', 'Price Near Lower BB', 0.7))
            elif bb_pos > 0.9:
                signals.append(('SELL', 'Price Near Upper BB', 0.7))
            
            # Moving Average Signals
            if ma_data['ema_cross'] > 0 and ma_data['price_vs_ema'] > 0.02:
                signals.append(('BUY', 'Above Rising EMA', 0.6))
            elif ma_data['ema_cross'] < 0 and ma_data['price_vs_ema'] < -0.02:
                signals.append(('SELL', 'Below Falling EMA', 0.6))
            
            # Momentum Signals
            if momentum_data['stoch_k'] < 20 and momentum_data['stoch_d'] < 20:
                signals.append(('BUY', 'Stochastic Oversold', 0.5))
            elif momentum_data['stoch_k'] > 80 and momentum_data['stoch_d'] > 80:
                signals.append(('SELL', 'Stochastic Overbought', 0.5))
            
            # Volume Confirmation
            if volume_data['volume_ratio'] > 1.5:  # High volume
                if any(s[0] == 'BUY' for s in signals):
                    signals.append(('BUY', 'High Volume Confirmation', 0.4))
                elif any(s[0] == 'SELL' for s in signals):
                    signals.append(('SELL', 'High Volume Confirmation', 0.4))
            
            # Price momentum signals
            if price_change_1h > 3:
                signals.append(('BUY', 'Strong 1h Momentum', 0.5))
            elif price_change_1h < -3:
                signals.append(('SELL', 'Strong 1h Decline', 0.5))
            
            # Calculate final signal
            buy_strength = sum(s[2] for s in signals if s[0] == 'BUY')
            sell_strength = sum(s[2] for s in signals if s[0] == 'SELL')
            
            # Determine final signal with threshold
            threshold = self.config.SIGNAL_STRENGTH_THRESHOLD
            
            if buy_strength > sell_strength and buy_strength >= threshold:
                final_signal = 'BUY'
                strength = buy_strength
                reasons = [s[1] for s in signals if s[0] == 'BUY']
            elif sell_strength > buy_strength and sell_strength >= threshold:
                final_signal = 'SELL'
                strength = sell_strength
                reasons = [s[1] for s in signals if s[0] == 'SELL']
            else:
                final_signal = 'HOLD'
                strength = 0
                reasons = ['No strong signal' if signals else 'No signals generated']
            
            # Compile comprehensive analysis
            analysis = {
                'signal': final_signal,
                'strength': strength,
                'reasons': reasons,
                'price': current_price,
                'rsi': rsi,
                'macd': macd_data,
                'bollinger': bb_data,
                'moving_averages': ma_data,
                'momentum': momentum_data,
                'volume': volume_data,
                'price_changes': {
                    '1h': price_change_1h,
                    '4h': price_change_4h,
                    '24h': price_change_24h
                },
                'all_signals': signals,
                'buy_strength': buy_strength,
                'sell_strength': sell_strength
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Analysis error for {symbol}: {e}")
            return {
                'signal': 'HOLD',
                'strength': 0,
                'rsi': 50,
                'price': current_price if 'current_price' in locals() else 0,
                'reasons': [f'Analysis error: {str(e)}']
            }
    
    def _calculate_price_change(self, prices: pd.Series, periods: int) -> float:
        """Calculate price change over specified periods"""
        try:
            if len(prices) <= periods:
                return 0.0
            
            current = prices.iloc[-1]
            previous = prices.iloc[-periods-1]
            
            return ((current - previous) / previous) * 100 if previous > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def get_support_resistance_levels(self, df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels
        """
        try:
            if len(df) < window * 2:
                return {'support': [], 'resistance': []}
            
            highs = df['high']
            lows = df['low']
            
            # Find local maxima and minima
            resistance_levels = []
            support_levels = []
            
            for i in range(window, len(df) - window):
                # Check if current high is a local maximum
                if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
                    resistance_levels.append(highs.iloc[i])
                
                # Check if current low is a local minimum
                if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
                    support_levels.append(lows.iloc[i])
            
            # Remove duplicates and sort
            resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]
            support_levels = sorted(list(set(support_levels)))[:5]
            
            return {
                'resistance': resistance_levels,
                'support': support_levels
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error finding support/resistance: {e}")
            return {'support': [], 'resistance': []}