# trading/risk_manager.py - Comprehensive risk management system
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

class RiskManager:
    """Comprehensive risk management and position sizing"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk tracking
        self.daily_loss = 0.0
        self.total_drawdown = 0.0
        self.day_start_balance = config.INITIAL_BALANCE
        self.peak_balance = config.INITIAL_BALANCE
        self.last_reset = datetime.now().date()
        
        # Position tracking
        self.open_positions = {}
        self.daily_trades = 0
        self.total_trades = 0
        
        # Risk alerts
        self.risk_alerts_sent = set()
        
    def can_open_position(self, symbol: str, price: float) -> bool:
        """Check if we can safely open a new position"""
        try:
            # Check daily loss limit
            if abs(self.daily_loss) >= self.config.MAX_DAILY_LOSS * self.day_start_balance:
                self.logger.warning(f"🚨 Daily loss limit reached: ${self.daily_loss:.2f}")
                return False
            
            # Check maximum open positions
            if len(self.open_positions) >= self.config.MAX_OPEN_POSITIONS:
                self.logger.warning(f"🚨 Maximum open positions reached: {len(self.open_positions)}")
                return False
            
            # Check drawdown limit
            if self.total_drawdown >= self.config.MAX_DRAWDOWN:
                self.logger.warning(f"🚨 Maximum drawdown reached: {self.total_drawdown:.2%}")
                return False
            
            # Check if we already have a position in this symbol
            if symbol in self.open_positions:
                self.logger.info(f"⚠️ Already have position in {symbol}, skipping new entry")
                return False
            
            # Check minimum order value
            min_order_value = self.config.MIN_ORDER_SIZE
            if price * self.calculate_base_position_size(price, self.day_start_balance) < min_order_value:
                self.logger.info(f"⚠️ Order value too small for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error checking position limits: {e}")
            return False
    
    def calculate_position_size(self, symbol: str, price: float, available_balance: float, 
                              risk_multiplier: float = 1.0) -> float:
        """Calculate optimal position size with risk management"""
        try:
            if price <= 0 or available_balance <= 0:
                return 0.0
            
            # Base position size (percentage of balance)
            base_position_value = available_balance * self.config.MAX_POSITION_SIZE
            
            # Apply risk multiplier (for different strategy risk levels)
            adjusted_position_value = base_position_value * risk_multiplier
            
            # Reduce size if we're in a drawdown
            if self.total_drawdown > 0.05:  # If more than 5% drawdown
                drawdown_multiplier = 1 - (self.total_drawdown * 0.5)  # Reduce size
                adjusted_position_value *= max(0.3, drawdown_multiplier)
                self.logger.info(f"🛡️ Reducing position size due to drawdown: {self.total_drawdown:.2%}")
            
            # Ensure minimum order size
            min_position_value = self.config.MIN_ORDER_SIZE
            if adjusted_position_value < min_position_value:
                adjusted_position_value = min_position_value
            
            # Convert to quantity
            quantity = adjusted_position_value / price
            
            # Final safety check - never more than 50% in one trade
            max_single_trade = available_balance * 0.5
            if adjusted_position_value > max_single_trade:
                quantity = max_single_trade / price
                self.logger.warning(f"⚠️ Position size capped at 50% of balance")
            
            return max(0.0, quantity)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return 0.0
    
    def calculate_base_position_size(self, price: float, balance: float) -> float:
        """Calculate base position size without modifiers"""
        return (balance * self.config.MAX_POSITION_SIZE) / price if price > 0 else 0.0
    
    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price"""
        try:
            if side.lower() == 'buy':
                return entry_price * (1 - self.config.STOP_LOSS_PERCENT)
            else:  # sell
                return entry_price * (1 + self.config.STOP_LOSS_PERCENT)
        except Exception as e:
            self.logger.error(f"❌ Error calculating stop loss: {e}")
            return entry_price
    
    def calculate_take_profit(self, entry_price: float, side: str, risk_reward_ratio: float = 2.0) -> float:
        """Calculate take profit price based on risk-reward ratio"""
        try:
            stop_distance = entry_price * self.config.STOP_LOSS_PERCENT
            profit_distance = stop_distance * risk_reward_ratio
            
            if side.lower() == 'buy':
                return entry_price + profit_distance
            else:  # sell
                return entry_price - profit_distance
        except Exception as e:
            self.logger.error(f"❌ Error calculating take profit: {e}")
            return entry_price
    
    def register_position(self, symbol: str, entry_price: float, quantity: float, side: str):
        """Register a new position for tracking"""
        try:
            self.open_positions[symbol] = {
                'entry_price': entry_price,
                'quantity': quantity,
                'side': side.lower(),
                'entry_time': datetime.now(),
                'stop_loss': self.calculate_stop_loss(entry_price, side),
                'take_profit': self.calculate_take_profit(entry_price, side),
                'unrealized_pnl': 0.0
            }
            
            self.daily_trades += 1
            self.total_trades += 1
            
            self.logger.info(f"📝 Position registered: {symbol} {side.upper()} {quantity:.6f} @ ${entry_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error registering position: {e}")
    
    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close a position and calculate P&L"""
        try:
            if symbol not in self.open_positions:
                self.logger.warning(f"⚠️ No position found for {symbol}")
                return 0.0
            
            position = self.open_positions[symbol]
            entry_price = position['entry_price']
            quantity = position['quantity']
            side = position['side']
            
            # Calculate P&L
            if side == 'buy':
                pnl = (exit_price - entry_price) * quantity
            else:  # sell
                pnl = (entry_price - exit_price) * quantity
            
            # Update tracking
            self.update_daily_pnl(pnl)
            
            # Remove position
            del self.open_positions[symbol]
            
            self.logger.info(f"💰 Position closed: {symbol} P&L: ${pnl:.2f}")
            
            return pnl
            
        except Exception as e:
            self.logger.error(f"❌ Error closing position: {e}")
            return 0.0
    
    def update_unrealized_pnl(self, symbol: str, current_price: float):
        """Update unrealized P&L for open positions"""
        try:
            if symbol not in self.open_positions:
                return
            
            position = self.open_positions[symbol]
            entry_price = position['entry_price']
            quantity = position['quantity']
            side = position['side']
            
            # Calculate unrealized P&L
            if side == 'buy':
                unrealized_pnl = (current_price - entry_price) * quantity
            else:  # sell
                unrealized_pnl = (entry_price - current_price) * quantity
            
            position['unrealized_pnl'] = unrealized_pnl
            
        except Exception as e:
            self.logger.error(f"❌ Error updating unrealized P&L: {e}")
    
    def should_close_position(self, symbol: str, current_price: float) -> bool:
        """Check if position should be closed due to stop loss or take profit"""
        try:
            if symbol not in self.open_positions:
                return False
            
            position = self.open_positions[symbol]
            side = position['side']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            if side == 'buy':
                # Long position
                if current_price <= stop_loss:
                    self.logger.warning(f"🛑 Stop loss triggered for {symbol}: ${current_price:.2f} <= ${stop_loss:.2f}")
                    return True
                elif current_price >= take_profit:
                    self.logger.info(f"🎯 Take profit triggered for {symbol}: ${current_price:.2f} >= ${take_profit:.2f}")
                    return True
            else:
                # Short position
                if current_price >= stop_loss:
                    self.logger.warning(f"🛑 Stop loss triggered for {symbol}: ${current_price:.2f} >= ${stop_loss:.2f}")
                    return True
                elif current_price <= take_profit:
                    self.logger.info(f"🎯 Take profit triggered for {symbol}: ${current_price:.2f} <= ${take_profit:.2f}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error checking position close conditions: {e}")
            return False
    
    def update_daily_pnl(self, pnl_change: float):
        """Update daily P&L tracking"""
        try:
            self.daily_loss += pnl_change
            
            # Update drawdown tracking
            current_balance = self.day_start_balance + self.daily_loss
            
            if current_balance > self.peak_balance:
                # New peak - reset drawdown
                self.peak_balance = current_balance
                self.total_drawdown = 0
            else:
                # Calculate current drawdown
                drawdown = (self.peak_balance - current_balance) / self.peak_balance
                self.total_drawdown = max(self.total_drawdown, drawdown)
            
            # Send risk alerts if needed
            self._check_risk_alerts()
                    
        except Exception as e:
            self.logger.error(f"❌ Error updating daily P&L: {e}")
    
    def _check_risk_alerts(self):
        """Check and send risk alerts"""
        try:
            # Daily loss alert
            daily_loss_percent = abs(self.daily_loss) / self.day_start_balance
            if daily_loss_percent > 0.03 and 'daily_loss_3pct' not in self.risk_alerts_sent:
                self.logger.warning(f"⚠️ Daily loss exceeds 3%: ${self.daily_loss:.2f}")
                self.risk_alerts_sent.add('daily_loss_3pct')
            
            # Drawdown alert
            if self.total_drawdown > 0.10 and 'drawdown_10pct' not in self.risk_alerts_sent:
                self.logger.warning(f"⚠️ Drawdown exceeds 10%: {self.total_drawdown:.2%}")
                self.risk_alerts_sent.add('drawdown_10pct')
                
        except Exception as e:
            self.logger.error(f"❌ Error checking risk alerts: {e}")
    
    async def monitor_risk(self):
        """Continuous risk monitoring"""
        while True:
            try:
                current_date = datetime.now().date()
                
                # Reset daily counters at midnight
                if current_date != self.last_reset:
                    self.logger.info("📊 Resetting daily risk counters")
                    self.day_start_balance += self.daily_loss  # Carry over P&L
                    self.daily_loss = 0.0
                    self.daily_trades = 0
                    self.last_reset = current_date
                    self.risk_alerts_sent.clear()
                
                # Log risk status periodically
                current_hour = datetime.now().hour
                current_minute = datetime.now().minute
                
                if current_minute % 15 == 0 and current_minute == 0:  # Every hour
                    self.logger.info(f"🛡️ Risk Status - Daily P&L: ${self.daily_loss:.2f}, "
                                   f"Drawdown: {self.total_drawdown:.2%}, "
                                   f"Open Positions: {len(self.open_positions)}, "
                                   f"Daily Trades: {self.daily_trades}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"❌ Risk monitoring error: {e}")
                await asyncio.sleep(60)
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        try:
            total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.open_positions.values())
            current_balance = self.day_start_balance + self.daily_loss + total_unrealized_pnl
            
            return {
                'daily_pnl': self.daily_loss,
                'total_drawdown': self.total_drawdown,
                'open_positions': len(self.open_positions),
                'daily_trades': self.daily_trades,
                'total_trades': self.total_trades,
                'current_balance': current_balance,
                'peak_balance': self.peak_balance,
                'unrealized_pnl': total_unrealized_pnl,
                'daily_loss_percent': abs(self.daily_loss) / self.day_start_balance * 100,
                'positions_summary': {
                    symbol: {
                        'entry_price': pos['entry_price'],
                        'current_pnl': pos['unrealized_pnl'],
                        'side': pos['side']
                    } for symbol, pos in self.open_positions.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating risk summary: {e}")
            return {
                'daily_pnl': 0, 'total_drawdown': 0, 'open_positions': 0,
                'daily_trades': 0, 'total_trades': 0, 'current_balance': 0,
                'peak_balance': 0, 'unrealized_pnl': 0, 'daily_loss_percent': 0,
                'positions_summary': {}
            }
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed based on risk limits"""
        daily_loss_limit = abs(self.daily_loss) < self.config.MAX_DAILY_LOSS * self.day_start_balance
        drawdown_limit = self.total_drawdown < self.config.MAX_DRAWDOWN
        position_limit = len(self.open_positions) < self.config.MAX_OPEN_POSITIONS
        
        return daily_loss_limit and drawdown_limit and position_limit