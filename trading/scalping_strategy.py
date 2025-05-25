# trading/scalping_strategy.py - Advanced scalping strategy for quick profits
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

class ScalpingStrategy:
    """
    Advanced scalping strategy for high-frequency trading
    Targets quick 0.1-0.5% moves with tight risk management
    """
    
    def __init__(self, config, data_collector, risk_manager, trade_executor):
        self.config = config
        self.data_collector = data_collector
        self.risk_manager = risk_manager
        self.trade_executor = trade_executor
        self.logger = logging.getLogger(__name__)
        
        # Scalping configuration
        self.target_profit_percent = 0.003  # 0.3% target profit
        self.stop_loss_percent = 0.002      # 0.2% stop loss (tight)
        self.max_hold_time = 900            # 15 minutes max hold time
        self.min_volume_ratio = 1.5         # Minimum volume ratio
        self.rsi_oversold = 25              # Extreme oversold
        self.rsi_overbought = 75            # Extreme overbought
        
        # Scalping state
        self.active_scalp_positions = {}    # symbol -> position info
        self.scalp_profits = {}             # symbol -> total profits
        self.scalp_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0,
            'avg_hold_time': 0,
            'success_rate': 0
        }
        
        # Performance tracking
        self.entry_signals = {}             # symbol -> entry conditions
        self.exit_conditions = {}           # symbol -> exit monitoring
        
    def should_scalp(self, symbol: str, analysis: Dict, volatility: float) -> bool:
        """Determine if conditions are suitable for scalping"""
        try:
            # Get market data
            price_data = self.data_collector.get_price_data(symbol)
            volume_data = self.data_collector.get_volume_data(symbol)
            
            if not price_data or not volume_data:
                return False
                
            # Check volatility range (need some movement but not too much)
            if volatility < 0.015 or volatility > 0.08:
                return False
                
            # Check volume (need above-average volume)
            volume_ratio = volume_data.get('ratio', 1.0)
            if volume_ratio < self.min_volume_ratio:
                return False
                
            # Check market hours (scalping works better during active hours)
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 22:  # Avoid low-activity hours
                return False
                
            # Check for existing position
            if symbol in self.active_scalp_positions:
                return False
                
            # Check recent price action (avoid after big moves)
            rsi = analysis.get('rsi', 50)
            if rsi < self.rsi_oversold or rsi > self.rsi_overbought:
                return True  # Extreme levels good for scalping
                
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error checking scalp conditions for {symbol}: {e}")
            return False
    
    async def execute_scalp(self, symbol: str, analysis: Dict, current_price: float, direction: str):
        """Execute a scalp trade"""
        try:
            # Validate direction
            if direction not in ['buy', 'sell']:
                return False
                
            # Calculate position size (smaller for scalping)
            available_balance = self.risk_manager.day_start_balance * 0.8  # Use 80% of balance
            scalp_position_size = self.risk_manager.calculate_position_size(
                symbol, current_price, available_balance, risk_multiplier=0.3  # Small positions
            )
            
            if scalp_position_size <= 0:
                return False
                
            # Execute the trade
            success = await self.trade_executor.execute_trade(
                symbol, direction, scalp_position_size, current_price, 'market'
            )
            
            if success:
                # Register scalp position
                self._register_scalp_position(symbol, direction, current_price, scalp_position_size, analysis)
                
                self.logger.info(f"⚡ SCALP {direction.upper()}: {symbol} "
                               f"{scalp_position_size:.6f} @ ${current_price:.2f}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error executing scalp for {symbol}: {e}")
            return False
    
    def _register_scalp_position(self, symbol: str, side: str, entry_price: float, 
                                quantity: float, analysis: Dict):
        """Register a scalp position for tracking"""
        try:
            # Calculate targets
            if side == 'buy':
                take_profit = entry_price * (1 + self.target_profit_percent)
                stop_loss = entry_price * (1 - self.stop_loss_percent)
            else:  # sell
                take_profit = entry_price * (1 - self.target_profit_percent)
                stop_loss = entry_price * (1 + self.stop_loss_percent)
            
            position = {
                'side': side,
                'entry_price': entry_price,
                'quantity': quantity,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'entry_time': datetime.now(),
                'entry_rsi': analysis.get('rsi', 50),
                'entry_reason': self._get_entry_reason(analysis),
                'max_profit': 0,
                'current_pnl': 0,
                'status': 'active'
            }
            
            self.active_scalp_positions[symbol] = position
            self.scalp_stats['total_trades'] += 1
            
            # Set up exit monitoring
            self.exit_conditions[symbol] = {
                'entry_time': datetime.now(),
                'max_hold_reached': False,
                'profit_target_hit': False,
                'stop_loss_hit': False
            }
            
            self.logger.info(f"📝 Scalp position registered: {symbol}")
            self.logger.info(f"   Target: ${take_profit:.4f} | Stop: ${stop_loss:.4f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error registering scalp position: {e}")
    
    def _get_entry_reason(self, analysis: Dict) -> str:
        """Determine the reason for entry"""
        rsi = analysis.get('rsi', 50)
        
        if rsi < 25:
            return "RSI Extremely Oversold"
        elif rsi > 75:
            return "RSI Extremely Overbought"
        elif analysis.get('signal') == 'BUY':
            return "Technical BUY Signal"
        elif analysis.get('signal') == 'SELL':
            return "Technical SELL Signal"
        else:
            return "Scalp Opportunity"
    
    async def monitor_scalp_positions(self):
        """Monitor active scalp positions for exit conditions"""
        try:
            for symbol in list(self.active_scalp_positions.keys()):
                current_price = self.data_collector.get_latest_price(symbol)
                if current_price:
                    await self._check_scalp_exit(symbol, current_price)
                    
        except Exception as e:
            self.logger.error(f"❌ Error monitoring scalp positions: {e}")
    
    async def _check_scalp_exit(self, symbol: str, current_price: float):
        """Check if a scalp position should be exited"""
        try:
            if symbol not in self.active_scalp_positions:
                return
                
            position = self.active_scalp_positions[symbol]
            exit_cond = self.exit_conditions[symbol]
            
            # Calculate current P&L
            if position['side'] == 'buy':
                current_pnl = (current_price - position['entry_price']) * position['quantity']
                profit_percent = (current_price - position['entry_price']) / position['entry_price']
            else:  # sell
                current_pnl = (position['entry_price'] - current_price) * position['quantity']
                profit_percent = (position['entry_price'] - current_price) / position['entry_price']
            
            position['current_pnl'] = current_pnl
            position['max_profit'] = max(position['max_profit'], current_pnl)
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # 1. Take Profit Hit
            if ((position['side'] == 'buy' and current_price >= position['take_profit']) or
                (position['side'] == 'sell' and current_price <= position['take_profit'])):
                should_exit = True
                exit_reason = "Take Profit Hit"
                exit_cond['profit_target_hit'] = True
            
            # 2. Stop Loss Hit
            elif ((position['side'] == 'buy' and current_price <= position['stop_loss']) or
                  (position['side'] == 'sell' and current_price >= position['stop_loss'])):
                should_exit = True
                exit_reason = "Stop Loss Hit"
                exit_cond['stop_loss_hit'] = True
            
            # 3. Maximum Hold Time Reached
            elif (datetime.now() - position['entry_time']).total_seconds() > self.max_hold_time:
                should_exit = True
                exit_reason = "Max Hold Time"
                exit_cond['max_hold_reached'] = True
            
            # 4. Trailing Stop (if profit is significant)
            elif position['max_profit'] > position['quantity'] * position['entry_price'] * 0.002:  # 0.2% profit
                # Trail stop loss to break-even + small profit
                if position['side'] == 'buy':
                    trailing_stop = position['entry_price'] * 1.001  # 0.1% above entry
                    if current_price <= trailing_stop:
                        should_exit = True
                        exit_reason = "Trailing Stop"
                else:  # sell
                    trailing_stop = position['entry_price'] * 0.999  # 0.1% below entry
                    if current_price >= trailing_stop:
                        should_exit = True
                        exit_reason = "Trailing Stop"
            
            # Execute exit if needed
            if should_exit:
                await self._exit_scalp_position(symbol, current_price, exit_reason)
                
        except Exception as e:
            self.logger.error(f"❌ Error checking scalp exit for {symbol}: {e}")
    
    async def _exit_scalp_position(self, symbol: str, exit_price: float, reason: str):
        """Exit a scalp position"""
        try:
            if symbol not in self.active_scalp_positions:
                return
                
            position = self.active_scalp_positions[symbol]
            
            # Execute opposite trade
            opposite_side = 'sell' if position['side'] == 'buy' else 'buy'
            success = await self.trade_executor.execute_trade(
                symbol, opposite_side, position['quantity'], exit_price, 'market'
            )
            
            if success:
                # Calculate final P&L
                final_pnl = position['current_pnl']
                hold_time = (datetime.now() - position['entry_time']).total_seconds()
                
                # Update statistics
                self.scalp_stats['total_profit'] += final_pnl
                
                if final_pnl > 0:
                    self.scalp_stats['winning_trades'] += 1
                
                # Calculate average hold time
                total_time = self.scalp_stats['avg_hold_time'] * (self.scalp_stats['total_trades'] - 1)
                self.scalp_stats['avg_hold_time'] = (total_time + hold_time) / self.scalp_stats['total_trades']
                
                # Update success rate
                self.scalp_stats['success_rate'] = (self.scalp_stats['winning_trades'] / 
                                                   self.scalp_stats['total_trades']) * 100
                
                # Add to symbol profits
                if symbol not in self.scalp_profits:
                    self.scalp_profits[symbol] = 0
                self.scalp_profits[symbol] += final_pnl
                
                # Log the exit
                self.logger.info(f"⚡ SCALP EXIT: {symbol} | Reason: {reason}")
                self.logger.info(f"   P&L: ${final_pnl:.2f} | Hold: {hold_time/60:.1f}min | "
                               f"Entry: ${position['entry_price']:.4f} | Exit: ${exit_price:.4f}")
                
                # Clean up
                del self.active_scalp_positions[symbol]
                del self.exit_conditions[symbol]
                
        except Exception as e:
            self.logger.error(f"❌ Error exiting scalp position for {symbol}: {e}")
    
    def get_scalp_signals(self, symbol: str, analysis: Dict, volatility: float) -> Optional[str]:
        """Get scalping signals based on market conditions"""
        try:
            if not self.should_scalp(symbol, analysis, volatility):
                return None
                
            rsi = analysis.get('rsi', 50)
            macd_data = analysis.get('macd', {})
            bb_data = analysis.get('bollinger', {})
            
            # RSI-based scalping signals
            if rsi < self.rsi_oversold:
                # Extremely oversold - scalp buy
                return 'buy'
            elif rsi > self.rsi_overbought:
                # Extremely overbought - scalp sell
                return 'sell'
            
            # MACD momentum scalping
            macd = macd_data.get('macd', 0)
            signal_line = macd_data.get('signal', 0)
            
            if macd > signal_line and rsi < 45:
                # Bullish momentum + not overbought
                return 'buy'
            elif macd < signal_line and rsi > 55:
                # Bearish momentum + not oversold
                return 'sell'
            
            # Bollinger Bands mean reversion scalping
            bb_position = bb_data.get('position', 0.5)
            
            if bb_position < 0.1 and rsi < 40:
                # Near lower band + oversold
                return 'buy'
            elif bb_position > 0.9 and rsi > 60:
                # Near upper band + overbought
                return 'sell'
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting scalp signals for {symbol}: {e}")
            return None
    
    def get_scalp_status(self, symbol: str) -> str:
        """Get scalping status for a symbol"""
        try:
            if symbol in self.active_scalp_positions:
                position = self.active_scalp_positions[symbol]
                hold_time = (datetime.now() - position['entry_time']).total_seconds() / 60
                return f"Scalp {position['side'].upper()}({hold_time:.1f}m, ${position['current_pnl']:.2f})"
            else:
                profit = self.scalp_profits.get(symbol, 0)
                if profit != 0:
                    return f"Scalp(${profit:.1f})"
                else:
                    return "Scalp Ready"
                    
        except Exception as e:
            self.logger.error(f"❌ Error getting scalp status: {e}")
            return "Scalp Error"
    
    def get_scalp_summary(self) -> Dict:
        """Get comprehensive scalping summary"""
        try:
            return {
                'active_positions': len(self.active_scalp_positions),
                'total_trades': self.scalp_stats['total_trades'],
                'winning_trades': self.scalp_stats['winning_trades'],
                'success_rate': self.scalp_stats['success_rate'],
                'total_profit': self.scalp_stats['total_profit'],
                'avg_hold_time_minutes': self.scalp_stats['avg_hold_time'] / 60,
                'profit_by_symbol': self.scalp_profits.copy(),
                'active_positions_details': {
                    symbol: {
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'current_pnl': pos['current_pnl'],
                        'hold_time_minutes': (datetime.now() - pos['entry_time']).total_seconds() / 60
                    } for symbol, pos in self.active_scalp_positions.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating scalp summary: {e}")
            return {
                'active_positions': 0, 'total_trades': 0, 'winning_trades': 0,
                'success_rate': 0, 'total_profit': 0, 'avg_hold_time_minutes': 0,
                'profit_by_symbol': {}, 'active_positions_details': {}
            }