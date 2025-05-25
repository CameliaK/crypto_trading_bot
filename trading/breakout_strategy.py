# trading/breakout_strategy.py - Breakout trading strategy
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

class BreakoutStrategy:
    """
    Breakout strategy that catches big moves when price breaks out of consolidation
    Detects support/resistance levels and trades the breakouts with volume confirmation
    """
    
    def __init__(self, config, data_collector, risk_manager, trade_executor):
        self.config = config
        self.data_collector = data_collector
        self.risk_manager = risk_manager
        self.trade_executor = trade_executor
        self.logger = logging.getLogger(__name__)
        
        # Breakout parameters
        self.consolidation_periods = 20     # Periods to check for consolidation
        self.breakout_threshold = 0.02      # 2% move required for breakout
        self.volume_confirmation = 1.5      # Volume must be 1.5x average
        self.max_consolidation_range = 0.05 # 5% max range for consolidation
        self.min_consolidation_time = 4     # Minimum 4 periods of consolidation
        
        # Position tracking
        self.breakout_positions = {}        # symbol -> position info
        self.breakout_profits = {}          # symbol -> total profits
        self.breakout_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0,
            'avg_hold_time_hours': 0,
            'success_rate': 0,
            'false_breakouts': 0,
            'true_breakouts': 0
        }
        
        # Breakout detection state
        self.support_resistance_levels = {} # symbol -> levels
        self.consolidation_zones = {}       # symbol -> consolidation info
        self.breakout_signals = {}          # symbol -> current breakout signal
        
    def detect_consolidation(self, symbol: str, historical_data: pd.DataFrame) -> Optional[Dict]:
        """Detect if price is in a consolidation phase"""
        try:
            if len(historical_data) < self.consolidation_periods + 10:
                return None
                
            # Get recent price data
            recent_data = historical_data.tail(self.consolidation_periods)
            highs = recent_data['high']
            lows = recent_data['low']
            closes = recent_data['close']
            volumes = recent_data['volume']
            
            # Calculate consolidation range
            highest_high = highs.max()
            lowest_low = lows.min()
            current_price = closes.iloc[-1]
            
            # Calculate range as percentage
            price_range = (highest_high - lowest_low) / current_price
            
            # Check if we're in consolidation (tight range)
            if price_range > self.max_consolidation_range:
                return None
                
            # Check for sufficient consolidation time
            # Price should stay within range for minimum periods
            range_breaks = 0
            for i in range(len(recent_data)):
                high = highs.iloc[i]
                low = lows.iloc[i]
                if high > highest_high * 0.98 or low < lowest_low * 1.02:
                    range_breaks += 1
            
            if range_breaks < self.min_consolidation_time:
                return None
                
            # Calculate support and resistance levels
            resistance = highest_high
            support = lowest_low
            middle = (resistance + support) / 2
            
            # Determine consolidation quality
            avg_volume = volumes.mean()
            recent_volume = volumes.tail(5).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate volatility during consolidation
            returns = closes.pct_change().dropna()
            consolidation_volatility = returns.std()
            
            consolidation_info = {
                'resistance': resistance,
                'support': support,
                'middle': middle,
                'range_percent': price_range,
                'periods_consolidating': len(recent_data),
                'current_price': current_price,
                'volume_ratio': volume_ratio,
                'volatility': consolidation_volatility,
                'quality_score': self._calculate_consolidation_quality(
                    price_range, len(recent_data), volume_ratio, consolidation_volatility
                )
            }
            
            # Store consolidation info
            self.consolidation_zones[symbol] = consolidation_info
            
            return consolidation_info
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting consolidation for {symbol}: {e}")
            return None
    
    def _calculate_consolidation_quality(self, price_range: float, periods: int, 
                                       volume_ratio: float, volatility: float) -> float:
        """Calculate quality score for consolidation (0-1)"""
        try:
            # Better consolidation = tighter range, longer time, normal volume, low volatility
            range_score = max(0, 1 - (price_range / 0.03))  # Prefer < 3% range
            time_score = min(1.0, periods / 30)              # Prefer longer consolidation
            volume_score = max(0.5, min(1.0, volume_ratio))  # Prefer normal to high volume
            volatility_score = max(0, 1 - (volatility * 50)) # Prefer low volatility
            
            # Weighted average
            quality = (range_score * 0.3 + time_score * 0.3 + 
                      volume_score * 0.2 + volatility_score * 0.2)
            
            return quality
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating consolidation quality: {e}")
            return 0.0
    
    def detect_breakout(self, symbol: str, current_price: float, volume_data: Dict) -> Optional[Dict]:
        """Detect if a breakout is occurring"""
        try:
            if symbol not in self.consolidation_zones:
                return None
                
            consolidation = self.consolidation_zones[symbol]
            resistance = consolidation['resistance']
            support = consolidation['support']
            
            # Check volume confirmation
            volume_ratio = volume_data.get('ratio', 1.0) if volume_data else 1.0
            
            if volume_ratio < self.volume_confirmation:
                return None  # Need volume confirmation
            
            breakout_info = None
            
            # Upward breakout (resistance break)
            if current_price > resistance:
                breakout_strength = (current_price - resistance) / resistance
                
                if breakout_strength >= self.breakout_threshold:
                    breakout_info = {
                        'direction': 'buy',
                        'type': 'upward_breakout',
                        'breakout_level': resistance,
                        'breakout_strength': breakout_strength,
                        'volume_confirmation': volume_ratio,
                        'consolidation_quality': consolidation['quality_score'],
                        'target_price': self._calculate_breakout_target(
                            current_price, resistance, support, 'up'
                        ),
                        'stop_loss': support,  # Stop below support
                        'entry_reason': f"Upward breakout above ${resistance:.2f}"
                    }
            
            # Downward breakout (support break)
            elif current_price < support:
                breakout_strength = (support - current_price) / support
                
                if breakout_strength >= self.breakout_threshold:
                    breakout_info = {
                        'direction': 'sell',
                        'type': 'downward_breakout',
                        'breakout_level': support,
                        'breakout_strength': breakout_strength,
                        'volume_confirmation': volume_ratio,
                        'consolidation_quality': consolidation['quality_score'],
                        'target_price': self._calculate_breakout_target(
                            current_price, resistance, support, 'down'
                        ),
                        'stop_loss': resistance,  # Stop above resistance
                        'entry_reason': f"Downward breakout below ${support:.2f}"
                    }
            
            # Calculate overall breakout signal strength
            if breakout_info:
                signal_strength = self._calculate_breakout_signal_strength(breakout_info)
                breakout_info['signal_strength'] = signal_strength
                
                # Store breakout signal
                self.breakout_signals[symbol] = breakout_info
            
            return breakout_info
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting breakout for {symbol}: {e}")
            return None
    
    def _calculate_breakout_target(self, current_price: float, resistance: float, 
                                 support: float, direction: str) -> float:
        """Calculate price target for breakout"""
        try:
            range_size = resistance - support
            
            if direction == 'up':
                # Target is resistance + range size (measured move)
                target = resistance + range_size
            else:  # down
                # Target is support - range size (measured move)
                target = support - range_size
            
            return target
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating breakout target: {e}")
            return current_price
    
    def _calculate_breakout_signal_strength(self, breakout_info: Dict) -> float:
        """Calculate overall signal strength for breakout (0-1)"""
        try:
            # Components of breakout strength
            breakout_strength = min(1.0, breakout_info['breakout_strength'] / 0.05)  # Normalize to 5% max
            volume_strength = min(1.0, breakout_info['volume_confirmation'] / 3.0)   # Normalize to 3x volume max  
            consolidation_quality = breakout_info['consolidation_quality']
            
            # Weighted combination
            signal_strength = (breakout_strength * 0.4 + 
                             volume_strength * 0.4 + 
                             consolidation_quality * 0.2)
            
            return signal_strength
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating breakout signal strength: {e}")
            return 0.0
    
    async def execute_breakout(self, symbol: str, breakout_info: Dict, current_price: float):
        """Execute a breakout trade"""
        try:
            direction = breakout_info['direction']
            signal_strength = breakout_info['signal_strength']
            
            # Only trade on strong breakout signals
            if signal_strength < 0.7:
                return False
                
            # Check if we already have a position
            if symbol in self.breakout_positions:
                return False
                
            # Calculate position size (higher risk for breakouts)
            available_balance = self.risk_manager.day_start_balance * 0.95
            position_size = self.risk_manager.calculate_position_size(
                symbol, current_price, available_balance, risk_multiplier=0.8  # Higher risk
            )
            
            if position_size <= 0:
                return False
                
            # Execute the trade
            success = await self.trade_executor.execute_trade(
                symbol, direction, position_size, current_price, 'market'
            )
            
            if success:
                # Register breakout position
                self._register_breakout_position(symbol, breakout_info, current_price, position_size)
                
                self.logger.info(f"🚀 BREAKOUT {direction.upper()}: {symbol} "
                               f"{position_size:.6f} @ ${current_price:.2f}")
                self.logger.info(f"   Type: {breakout_info['type']}")
                self.logger.info(f"   Target: ${breakout_info['target_price']:.2f}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error executing breakout for {symbol}: {e}")
            return False
    
    def _register_breakout_position(self, symbol: str, breakout_info: Dict, 
                                  entry_price: float, quantity: float):
        """Register a breakout position"""
        try:
            position = {
                'side': breakout_info['direction'],
                'entry_price': entry_price,
                'quantity': quantity,
                'breakout_level': breakout_info['breakout_level'],
                'target_price': breakout_info['target_price'],
                'stop_loss': breakout_info['stop_loss'],
                'entry_time': datetime.now(),
                'breakout_type': breakout_info['type'],
                'entry_reason': breakout_info['entry_reason'],
                'signal_strength': breakout_info['signal_strength'],
                'volume_confirmation': breakout_info['volume_confirmation'],
                'current_pnl': 0,
                'max_profit': 0,
                'max_loss': 0,
                'trailing_stop': None,
                'status': 'active'
            }
            
            self.breakout_positions[symbol] = position
            self.breakout_stats['total_trades'] += 1
            
            self.logger.info(f"📝 Breakout position registered: {symbol}")
            self.logger.info(f"   Target: ${position['target_price']:.2f} | Stop: ${position['stop_loss']:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error registering breakout position: {e}")
    
    async def monitor_breakout_positions(self):
        """Monitor active breakout positions"""
        try:
            for symbol in list(self.breakout_positions.keys()):
                current_price = self.data_collector.get_latest_price(symbol)
                if current_price:
                    await self._check_breakout_exit(symbol, current_price)
                    
        except Exception as e:
            self.logger.error(f"❌ Error monitoring breakout positions: {e}")
    
    async def _check_breakout_exit(self, symbol: str, current_price: float):
        """Check if a breakout position should be exited"""
        try:
            if symbol not in self.breakout_positions:
                return
                
            position = self.breakout_positions[symbol]
            
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
            
            # 1. Stop Loss Hit (False breakout)
            if ((position['side'] == 'buy' and current_price <= position['stop_loss']) or
                (position['side'] == 'sell' and current_price >= position['stop_loss'])):
                should_exit = True
                exit_reason = "Stop Loss Hit - False Breakout"
                self.breakout_stats['false_breakouts'] += 1
            
            # 2. Target Price Reached
            elif ((position['side'] == 'buy' and current_price >= position['target_price']) or
                  (position['side'] == 'sell' and current_price <= position['target_price'])):
                should_exit = True
                exit_reason = "Target Price Reached"
                self.breakout_stats['true_breakouts'] += 1
            
            # 3. Trailing Stop Management
            elif self._should_use_trailing_stop(position, current_price):
                if position['trailing_stop'] is None:
                    # Initialize trailing stop
                    position['trailing_stop'] = self._calculate_trailing_stop(position, current_price)
                else:
                    # Update trailing stop
                    new_trailing_stop = self._calculate_trailing_stop(position, current_price)
                    
                    if position['side'] == 'buy':
                        position['trailing_stop'] = max(position['trailing_stop'], new_trailing_stop)
                        if current_price <= position['trailing_stop']:
                            should_exit = True
                            exit_reason = "Trailing Stop Hit"
                    else:  # sell
                        position['trailing_stop'] = min(position['trailing_stop'], new_trailing_stop)
                        if current_price >= position['trailing_stop']:
                            should_exit = True
                            exit_reason = "Trailing Stop Hit"
            
            # 4. Time-based exit (if breakout isn't working after reasonable time)
            elif (datetime.now() - position['entry_time']).days >= 3:
                # If we're not making progress after 3 days, consider exit
                profit_percent = current_pnl / (position['quantity'] * position['entry_price'])
                if abs(profit_percent) < 0.01:  # Less than 1% move in 3 days
                    should_exit = True
                    exit_reason = "Stalled Breakout - Time Exit"
            
            # Execute exit if needed
            if should_exit:
                await self._exit_breakout_position(symbol, current_price, exit_reason)
                
        except Exception as e:
            self.logger.error(f"❌ Error checking breakout exit for {symbol}: {e}")
    
    def _should_use_trailing_stop(self, position: Dict, current_price: float) -> bool:
        """Determine if we should use trailing stop"""
        try:
            # Use trailing stop if we have significant profit
            profit_threshold = position['quantity'] * position['entry_price'] * 0.04  # 4% profit
            return position['current_pnl'] > profit_threshold
        except:
            return False
    
    def _calculate_trailing_stop(self, position: Dict, current_price: float) -> float:
        """Calculate trailing stop level"""
        try:
            # Use 2% trailing stop
            if position['side'] == 'buy':
                return current_price * 0.98  # 2% below current price
            else:  # sell
                return current_price * 1.02  # 2% above current price
        except:
            return position['stop_loss']
    
    async def _exit_breakout_position(self, symbol: str, exit_price: float, reason: str):
        """Exit a breakout position"""
        try:
            if symbol not in self.breakout_positions:
                return
                
            position = self.breakout_positions[symbol]
            
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
                self.breakout_stats['total_profit'] += final_pnl
                
                if final_pnl > 0:
                    self.breakout_stats['winning_trades'] += 1
                
                # Update average hold time
                total_hours = self.breakout_stats['avg_hold_time_hours'] * (self.breakout_stats['total_trades'] - 1)
                self.breakout_stats['avg_hold_time_hours'] = (total_hours + hold_hours) / self.breakout_stats['total_trades']
                
                # Update success rate
                self.breakout_stats['success_rate'] = (self.breakout_stats['winning_trades'] / 
                                                      self.breakout_stats['total_trades']) * 100
                
                # Add to symbol profits
                if symbol not in self.breakout_profits:
                    self.breakout_profits[symbol] = 0
                self.breakout_profits[symbol] += final_pnl
                
                # Log the exit
                self.logger.info(f"🚀 BREAKOUT EXIT: {symbol} | Reason: {reason}")
                self.logger.info(f"   P&L: ${final_pnl:.2f} | Hold: {hold_hours:.1f}h | "
                               f"Entry: ${position['entry_price']:.2f} | Exit: ${exit_price:.2f}")
                
                # Clean up consolidation zone after trade
                if symbol in self.consolidation_zones:
                    del self.consolidation_zones[symbol]
                
                # Clean up position
                del self.breakout_positions[symbol]
                
        except Exception as e:
            self.logger.error(f"❌ Error exiting breakout position for {symbol}: {e}")
    
    def get_breakout_signals(self, symbol: str, historical_data: pd.DataFrame, 
                           current_price: float, volume_data: Dict) -> Optional[Dict]:
        """Get breakout signals for a symbol"""
        try:
            # First, detect consolidation
            consolidation = self.detect_consolidation(symbol, historical_data)
            
            if not consolidation:
                return None
            
            # Then, check for breakout
            breakout = self.detect_breakout(symbol, current_price, volume_data)
            
            if breakout and breakout['signal_strength'] > 0.7:
                return {
                    'signal': breakout['direction'].upper(),
                    'strength': breakout['signal_strength'],
                    'strategy': 'breakout',
                    'reason': breakout['entry_reason'],
                    'target_price': breakout['target_price'],
                    'stop_loss': breakout['stop_loss'],
                    'breakout_type': breakout['type']
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting breakout signals for {symbol}: {e}")
            return None
    
    def get_breakout_status(self, symbol: str) -> str:
        """Get breakout status for a symbol"""
        try:
            if symbol in self.breakout_positions:
                position = self.breakout_positions[symbol]
                hold_time = (datetime.now() - position['entry_time']).total_seconds() / 3600
                profit_percent = (position['current_pnl'] / (position['quantity'] * position['entry_price'])) * 100
                return f"Breakout {position['side'].upper()}({hold_time:.1f}h, {profit_percent:+.1f}%)"
            
            elif symbol in self.consolidation_zones:
                consolidation = self.consolidation_zones[symbol]
                return f"Consolidating({consolidation['range_percent']:.1f}%, Q:{consolidation['quality_score']:.1f})"
            
            else:
                profit = self.breakout_profits.get(symbol, 0)
                if profit != 0:
                    return f"Breakout(${profit:.1f})"
                else:
                    return "Scanning Breakouts"
                    
        except Exception as e:
            self.logger.error(f"❌ Error getting breakout status: {e}")
            return "Breakout Error"
    
    def get_breakout_summary(self) -> Dict:
        """Get comprehensive breakout summary"""
        try:
            false_breakout_rate = 0
            if self.breakout_stats['total_trades'] > 0:
                false_breakout_rate = (self.breakout_stats['false_breakouts'] / 
                                     self.breakout_stats['total_trades']) * 100
                
            return {
                'active_positions': len(self.breakout_positions),
                'consolidation_zones': len(self.consolidation_zones),
                'total_trades': self.breakout_stats['total_trades'],
                'winning_trades': self.breakout_stats['winning_trades'],
                'success_rate': self.breakout_stats['success_rate'],
                'total_profit': self.breakout_stats['total_profit'],
                'avg_hold_time_hours': self.breakout_stats['avg_hold_time_hours'],
                'true_breakouts': self.breakout_stats['true_breakouts'],
                'false_breakouts': self.breakout_stats['false_breakouts'],
                'false_breakout_rate': false_breakout_rate,
                'profit_by_symbol': self.breakout_profits.copy(),
                'active_positions_details': {
                    symbol: {
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'target_price': pos['target_price'],
                        'current_pnl': pos['current_pnl'],
                        'breakout_type': pos['breakout_type'],
                        'hold_time_hours': (datetime.now() - pos['entry_time']).total_seconds() / 3600
                    } for symbol, pos in self.breakout_positions.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating breakout summary: {e}")
            return {
                'active_positions': 0, 'consolidation_zones': 0, 'total_trades': 0,
                'winning_trades': 0, 'success_rate': 0, 'total_profit': 0,
                'avg_hold_time_hours': 0, 'true_breakouts': 0, 'false_breakouts': 0,
                'false_breakout_rate': 0, 'profit_by_symbol': {},
                'active_positions_details': {}
            }