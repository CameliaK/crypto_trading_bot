# trading/mean_reversion_strategy.py - Mean reversion trading strategy
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

class MeanReversionStrategy:
    """
    Mean reversion strategy that profits from price returning to average
    Buys oversold dips and sells overbought peaks
    """
    
    def __init__(self, config, data_collector, risk_manager, trade_executor):
        self.config = config
        self.data_collector = data_collector
        self.risk_manager = risk_manager
        self.trade_executor = trade_executor
        self.logger = logging.getLogger(__name__)
        
        # Mean reversion parameters
        self.bb_entry_threshold = 0.1       # Enter when price is 10% into BB
        self.bb_exit_threshold = 0.5        # Exit when price returns to middle
        self.rsi_oversold = 35              # RSI oversold level
        self.rsi_overbought = 65            # RSI overbought level
        self.min_reversion_strength = 0.02  # Minimum deviation from mean
        self.max_hold_days = 5              # Maximum hold time
        
        # Position tracking
        self.mean_reversion_positions = {}  # symbol -> position info
        self.reversion_profits = {}         # symbol -> total profits
        self.reversion_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0,
            'avg_hold_time_hours': 0,
            'success_rate': 0,
            'best_trade': 0,
            'worst_trade': 0
        }
        
        # Strategy state
        self.reversion_signals = {}         # symbol -> current signal strength
        self.price_deviations = {}          # symbol -> deviation from mean
        
    def analyze_mean_reversion_opportunity(self, symbol: str, analysis: Dict, 
                                         historical_data: pd.DataFrame) -> Optional[Dict]:
        """Analyze if there's a mean reversion opportunity"""
        try:
            if len(historical_data) < 50:
                return None
                
            current_price = historical_data['close'].iloc[-1]
            rsi = analysis.get('rsi', 50)
            bb_data = analysis.get('bollinger', {})
            
            # Get moving averages for mean calculation
            ma_data = analysis.get('moving_averages', {})
            sma_20 = ma_data.get('sma_short', current_price)
            
            # Calculate price deviation from mean
            price_deviation = (current_price - sma_20) / sma_20
            self.price_deviations[symbol] = price_deviation
            
            # Bollinger Bands analysis
            bb_position = bb_data.get('position', 0.5)
            bb_width = bb_data.get('width', 0)
            
            # Check for mean reversion setup
            opportunity = None
            
            # Oversold mean reversion (BUY setup)
            if (bb_position < self.bb_entry_threshold and 
                rsi < self.rsi_oversold and 
                price_deviation < -self.min_reversion_strength):
                
                strength = self._calculate_reversion_strength(
                    rsi, bb_position, price_deviation, 'buy'
                )
                
                opportunity = {
                    'direction': 'buy',
                    'strength': strength,
                    'entry_reason': f"Oversold: RSI {rsi:.1f}, BB {bb_position:.2f}",
                    'target_price': sma_20,  # Target is the mean
                    'deviation': price_deviation,
                    'bb_width': bb_width
                }
            
            # Overbought mean reversion (SELL setup)
            elif (bb_position > (1 - self.bb_entry_threshold) and 
                  rsi > self.rsi_overbought and 
                  price_deviation > self.min_reversion_strength):
                
                strength = self._calculate_reversion_strength(
                    rsi, bb_position, price_deviation, 'sell'
                )
                
                opportunity = {
                    'direction': 'sell',
                    'strength': strength,
                    'entry_reason': f"Overbought: RSI {rsi:.1f}, BB {bb_position:.2f}",
                    'target_price': sma_20,  # Target is the mean
                    'deviation': price_deviation,
                    'bb_width': bb_width
                }
            
            # Store signal strength for monitoring
            if opportunity:
                self.reversion_signals[symbol] = opportunity['strength']
            else:
                self.reversion_signals[symbol] = 0
                
            return opportunity
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing mean reversion for {symbol}: {e}")
            return None
    
    def _calculate_reversion_strength(self, rsi: float, bb_position: float, 
                                    price_deviation: float, direction: str) -> float:
        """Calculate the strength of mean reversion signal"""
        try:
            strength = 0.0
            
            # RSI component
            if direction == 'buy':
                # More oversold = stronger signal
                rsi_strength = max(0, (40 - rsi) / 15)  # Normalize 25-40 RSI to 0-1
            else:  # sell
                # More overbought = stronger signal
                rsi_strength = max(0, (rsi - 60) / 15)  # Normalize 60-75 RSI to 0-1
            
            # Bollinger Bands component
            if direction == 'buy':
                bb_strength = max(0, (0.2 - bb_position) / 0.2)  # Stronger when closer to lower band
            else:  # sell
                bb_strength = max(0, (bb_position - 0.8) / 0.2)  # Stronger when closer to upper band
            
            # Price deviation component
            deviation_strength = min(1.0, abs(price_deviation) / 0.05)  # Cap at 5% deviation
            
            # Combine components
            strength = (rsi_strength * 0.4 + bb_strength * 0.4 + deviation_strength * 0.2)
            
            return strength
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating reversion strength: {e}")
            return 0.0
    
    async def execute_mean_reversion(self, symbol: str, opportunity: Dict, current_price: float):
        """Execute a mean reversion trade"""
        try:
            direction = opportunity['direction']
            strength = opportunity['strength']
            
            # Only trade on strong signals
            if strength < 0.6:
                return False
                
            # Check if we already have a position
            if symbol in self.mean_reversion_positions:
                return False
                
            # Calculate position size (moderate risk)
            available_balance = self.risk_manager.day_start_balance * 0.9
            position_size = self.risk_manager.calculate_position_size(
                symbol, current_price, available_balance, risk_multiplier=0.6
            )
            
            if position_size <= 0:
                return False
                
            # Execute the trade
            success = await self.trade_executor.execute_trade(
                symbol, direction, position_size, current_price, 'market'
            )
            
            if success:
                # Register mean reversion position
                self._register_reversion_position(symbol, opportunity, current_price, position_size)
                
                self.logger.info(f"🔄 MEAN REVERSION {direction.upper()}: {symbol} "
                               f"{position_size:.6f} @ ${current_price:.2f}")
                self.logger.info(f"   Reason: {opportunity['entry_reason']}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error executing mean reversion for {symbol}: {e}")
            return False
    
    def _register_reversion_position(self, symbol: str, opportunity: Dict, 
                                   entry_price: float, quantity: float):
        """Register a mean reversion position"""
        try:
            direction = opportunity['direction']
            target_price = opportunity['target_price']
            
            # Calculate stop loss (wider than scalping)
            if direction == 'buy':
                stop_loss = entry_price * (1 - 0.04)  # 4% stop loss
                take_profit = target_price  # Target is the mean
            else:  # sell
                stop_loss = entry_price * (1 + 0.04)  # 4% stop loss
                take_profit = target_price  # Target is the mean
            
            position = {
                'side': direction,
                'entry_price': entry_price,
                'quantity': quantity,
                'target_price': target_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'entry_time': datetime.now(),
                'entry_reason': opportunity['entry_reason'],
                'initial_deviation': opportunity['deviation'],
                'bb_width_at_entry': opportunity['bb_width'],
                'current_pnl': 0,
                'max_profit': 0,
                'max_loss': 0,
                'status': 'active'
            }
            
            self.mean_reversion_positions[symbol] = position
            self.reversion_stats['total_trades'] += 1
            
            self.logger.info(f"📝 Mean reversion position registered: {symbol}")
            self.logger.info(f"   Target: ${take_profit:.2f} | Stop: ${stop_loss:.2f}")
            self.logger.info(f"   Expected reversion: {opportunity['deviation']:.2%}")
            
        except Exception as e:
            self.logger.error(f"❌ Error registering reversion position: {e}")
    
    async def monitor_reversion_positions(self):
        """Monitor active mean reversion positions"""
        try:
            for symbol in list(self.mean_reversion_positions.keys()):
                current_price = self.data_collector.get_latest_price(symbol)
                if current_price:
                    await self._check_reversion_exit(symbol, current_price)
                    
        except Exception as e:
            self.logger.error(f"❌ Error monitoring reversion positions: {e}")
    
    async def _check_reversion_exit(self, symbol: str, current_price: float):
        """Check if a mean reversion position should be exited"""
        try:
            if symbol not in self.mean_reversion_positions:
                return
                
            position = self.mean_reversion_positions[symbol]
            
            # Calculate current P&L
            if position['side'] == 'buy':
                current_pnl = (current_price - position['entry_price']) * position['quantity']
            else:  # sell
                current_pnl = (position['entry_price'] - current_price) * position['quantity']
            
            position['current_pnl'] = current_pnl
            position['max_profit'] = max(position['max_profit'], current_pnl)
            position['max_loss'] = min(position['max_loss'], current_pnl)
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # 1. Stop Loss Hit
            if ((position['side'] == 'buy' and current_price <= position['stop_loss']) or
                (position['side'] == 'sell' and current_price >= position['stop_loss'])):
                should_exit = True
                exit_reason = "Stop Loss Hit"
            
            # 2. Target (Mean) Reached
            elif self._is_near_target(current_price, position['target_price'], 0.01):  # 1% of target
                should_exit = True
                exit_reason = "Mean Reversion Target Reached"
            
            # 3. Maximum Hold Time Reached
            elif (datetime.now() - position['entry_time']).days >= self.max_hold_days:
                should_exit = True
                exit_reason = "Maximum Hold Time Reached"
            
            # 4. Profit Protection (if we have good profit, protect it)
            elif current_pnl > position['quantity'] * position['entry_price'] * 0.03:  # 3% profit
                # Use trailing stop
                if position['side'] == 'buy':
                    trailing_stop = current_price * 0.98  # 2% below current price
                    if current_price <= trailing_stop:
                        should_exit = True
                        exit_reason = "Profit Protection Trailing Stop"
                else:  # sell
                    trailing_stop = current_price * 1.02  # 2% above current price
                    if current_price >= trailing_stop:
                        should_exit = True
                        exit_reason = "Profit Protection Trailing Stop"
            
            # 5. Bollinger Bands mean reversion completion check
            elif self._check_bb_reversion_complete(symbol):
                should_exit = True
                exit_reason = "Bollinger Bands Mean Reversion Complete"
            
            # Execute exit if needed
            if should_exit:
                await self._exit_reversion_position(symbol, current_price, exit_reason)
                
        except Exception as e:
            self.logger.error(f"❌ Error checking reversion exit for {symbol}: {e}")
    
    def _is_near_target(self, current_price: float, target_price: float, tolerance: float) -> bool:
        """Check if current price is near target within tolerance"""
        try:
            distance = abs(current_price - target_price) / target_price
            return distance <= tolerance
        except:
            return False
    
    def _check_bb_reversion_complete(self, symbol: str) -> bool:
        """Check if Bollinger Bands show mean reversion is complete"""
        try:
            # Get current analysis
            historical_data = self.data_collector.get_historical_data(symbol)
            if historical_data is None or len(historical_data) < 20:
                return False
                
            # Calculate current Bollinger Bands position
            prices = historical_data['close']
            sma = prices.rolling(window=20).mean()
            std = prices.rolling(window=20).std()
            
            current_price = prices.iloc[-1]
            current_sma = sma.iloc[-1]
            current_std = std.iloc[-1]
            
            if current_std == 0:
                return False
                
            # Calculate current position (0 = lower band, 1 = upper band)
            lower_band = current_sma - (2 * current_std)
            upper_band = current_sma + (2 * current_std)
            bb_position = (current_price - lower_band) / (upper_band - lower_band)
            
            # Check if price has reverted to middle area (0.4 to 0.6)
            return 0.4 <= bb_position <= 0.6
            
        except Exception as e:
            self.logger.error(f"❌ Error checking BB reversion: {e}")
            return False
    
    async def _exit_reversion_position(self, symbol: str, exit_price: float, reason: str):
        """Exit a mean reversion position"""
        try:
            if symbol not in self.mean_reversion_positions:
                return
                
            position = self.mean_reversion_positions[symbol]
            
            # Execute opposite trade
            opposite_side = 'sell' if position['side'] == 'buy' else 'buy'
            success = await self.trade_executor.execute_trade(
                symbol, opposite_side, position['quantity'], exit_price, 'market'
            )
            
            if success:
                # Calculate final metrics
                final_pnl = position['current_pnl']
                hold_time = datetime.now() - position['entry_time']
                hold_hours = hold_time.total_seconds() / 3600
                
                # Update statistics
                self.reversion_stats['total_profit'] += final_pnl
                
                if final_pnl > 0:
                    self.reversion_stats['winning_trades'] += 1
                
                # Track best/worst trades
                self.reversion_stats['best_trade'] = max(self.reversion_stats['best_trade'], final_pnl)
                self.reversion_stats['worst_trade'] = min(self.reversion_stats['worst_trade'], final_pnl)
                
                # Update average hold time
                total_hours = self.reversion_stats['avg_hold_time_hours'] * (self.reversion_stats['total_trades'] - 1)
                self.reversion_stats['avg_hold_time_hours'] = (total_hours + hold_hours) / self.reversion_stats['total_trades']
                
                # Update success rate
                self.reversion_stats['success_rate'] = (self.reversion_stats['winning_trades'] / 
                                                       self.reversion_stats['total_trades']) * 100
                
                # Add to symbol profits
                if symbol not in self.reversion_profits:
                    self.reversion_profits[symbol] = 0
                self.reversion_profits[symbol] += final_pnl
                
                # Log the exit
                self.logger.info(f"🔄 MEAN REVERSION EXIT: {symbol} | Reason: {reason}")
                self.logger.info(f"   P&L: ${final_pnl:.2f} | Hold: {hold_hours:.1f}h | "
                               f"Entry: ${position['entry_price']:.2f} | Exit: ${exit_price:.2f}")
                
                # Clean up
                del self.mean_reversion_positions[symbol]
                
        except Exception as e:
            self.logger.error(f"❌ Error exiting reversion position for {symbol}: {e}")
    
    def get_reversion_signals(self, symbol: str, analysis: Dict, historical_data: pd.DataFrame) -> Optional[Dict]:
        """Get mean reversion signals"""
        try:
            opportunity = self.analyze_mean_reversion_opportunity(symbol, analysis, historical_data)
            
            if opportunity and opportunity['strength'] > 0.6:
                return {
                    'signal': opportunity['direction'].upper(),
                    'strength': opportunity['strength'],
                    'strategy': 'mean_reversion',
                    'reason': opportunity['entry_reason'],
                    'target_price': opportunity['target_price'],
                    'deviation': opportunity['deviation']
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting reversion signals for {symbol}: {e}")
            return None
    
    def get_reversion_status(self, symbol: str) -> str:
        """Get mean reversion status for a symbol"""
        try:
            if symbol in self.mean_reversion_positions:
                position = self.mean_reversion_positions[symbol]
                hold_time = (datetime.now() - position['entry_time']).total_seconds() / 3600
                deviation = abs(position['current_pnl']) / (position['quantity'] * position['entry_price']) * 100
                return f"MeanRev {position['side'].upper()}({hold_time:.1f}h, {deviation:+.1f}%)"
            else:
                profit = self.reversion_profits.get(symbol, 0)
                signal_strength = self.reversion_signals.get(symbol, 0)
                
                if profit != 0:
                    return f"MeanRev(${profit:.1f})"
                elif signal_strength > 0.3:
                    return f"MeanRev Signal({signal_strength:.1f})"
                else:
                    return "MeanRev Ready"
                    
        except Exception as e:
            self.logger.error(f"❌ Error getting reversion status: {e}")
            return "MeanRev Error"
    
    def get_reversion_summary(self) -> Dict:
        """Get comprehensive mean reversion summary"""
        try:
            return {
                'active_positions': len(self.mean_reversion_positions),
                'total_trades': self.reversion_stats['total_trades'],
                'winning_trades': self.reversion_stats['winning_trades'],
                'success_rate': self.reversion_stats['success_rate'],
                'total_profit': self.reversion_stats['total_profit'],
                'avg_hold_time_hours': self.reversion_stats['avg_hold_time_hours'],
                'best_trade': self.reversion_stats['best_trade'],
                'worst_trade': self.reversion_stats['worst_trade'],
                'profit_by_symbol': self.reversion_profits.copy(),
                'active_positions_details': {
                    symbol: {
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'target_price': pos['target_price'],
                        'current_pnl': pos['current_pnl'],
                        'hold_time_hours': (datetime.now() - pos['entry_time']).total_seconds() / 3600,
                        'initial_deviation': pos['initial_deviation']
                    } for symbol, pos in self.mean_reversion_positions.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating reversion summary: {e}")
            return {
                'active_positions': 0, 'total_trades': 0, 'winning_trades': 0,
                'success_rate': 0, 'total_profit': 0, 'avg_hold_time_hours': 0,
                'best_trade': 0, 'worst_trade': 0, 'profit_by_symbol': {},
                'active_positions_details': {}
            }