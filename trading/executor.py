# trading/executor.py - Professional trade execution system
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, List
from utils.logger import get_trade_logger

class TradeExecutor:
    """Professional trade execution system with realistic simulation"""
    
    def __init__(self, config, exchange):
        self.config = config
        self.exchange = exchange
        self.logger = logging.getLogger(__name__)
        self.trade_logger = get_trade_logger()
        
        # Execution tracking
        self.pending_orders = {}
        self.executed_trades = []
        self.failed_trades = []
        self.total_fees = 0.0
        
        # Performance metrics
        self.total_trades = 0
        self.successful_trades = 0
        self.total_slippage = 0.0
        
    async def execute_trade(self, symbol: str, side: str, quantity: float, 
                          price: float, order_type: str = 'market') -> bool:
        """Execute a trade with comprehensive error handling"""
        try:
            trade_id = f"{symbol}_{side}_{int(datetime.now().timestamp())}"
            
            self.logger.info(f"🔄 Executing {side.upper()} order: {symbol} "
                           f"{quantity:.6f} @ ${price:.2f}")
            
            # Validate trade parameters
            if not self._validate_trade_params(symbol, side, quantity, price):
                return False
            
            if self.config.SANDBOX_MODE:
                # Simulate trade execution with realistic behavior
                success = await self._simulate_trade_execution(
                    trade_id, symbol, side, quantity, price, order_type
                )
            else:
                # Real trade execution
                success = await self._execute_real_trade(
                    trade_id, symbol, side, quantity, price, order_type
                )
            
            # Update statistics
            self.total_trades += 1
            if success:
                self.successful_trades += 1
                self.logger.info(f"✅ Trade executed: {side.upper()} {symbol}")
                self._log_successful_trade(trade_id, symbol, side, quantity, price)
            else:
                self.logger.error(f"❌ Trade failed: {side.upper()} {symbol}")
                self._log_failed_trade(trade_id, symbol, side, quantity, price)
                
            return success
                
        except Exception as e:
            self.logger.error(f"❌ Trade execution error: {e}")
            return False
    
    def _validate_trade_params(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """Validate trade parameters before execution"""
        try:
            # Check symbol
            if not symbol or '/' not in symbol:
                self.logger.error(f"❌ Invalid symbol: {symbol}")
                return False
            
            # Check side
            if side.lower() not in ['buy', 'sell']:
                self.logger.error(f"❌ Invalid side: {side}")
                return False
            
            # Check quantity
            if quantity <= 0:
                self.logger.error(f"❌ Invalid quantity: {quantity}")
                return False
            
            # Check price
            if price <= 0:
                self.logger.error(f"❌ Invalid price: {price}")
                return False
            
            # Check minimum order value
            order_value = quantity * price
            if order_value < self.config.MIN_ORDER_SIZE:
                self.logger.error(f"❌ Order value too small: ${order_value:.2f} < ${self.config.MIN_ORDER_SIZE}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Trade validation error: {e}")
            return False
    
    async def _simulate_trade_execution(self, trade_id: str, symbol: str, side: str, 
                                      quantity: float, price: float, order_type: str) -> bool:
        """Simulate realistic trade execution with slippage and fees"""
        try:
            # Simulate network delay (50-200ms)
            delay = 0.05 + (hash(trade_id) % 150) / 1000
            await asyncio.sleep(delay)
            
            # Simulate occasional failures (2% failure rate)
            failure_chance = hash(trade_id) % 50
            if failure_chance == 0:
                self.logger.warning(f"⚠️ Simulated trade failure for {trade_id} (network/exchange error)")
                return False
            
            # Calculate executed price with slippage
            executed_price = self._calculate_slippage(price, side, order_type, trade_id)
            
            # Calculate trading fees (0.1% maker/taker fee)
            fee_rate = 0.001
            if order_type == 'limit':
                fee_rate = 0.0008  # Lower fees for limit orders
            
            fee = quantity * executed_price * fee_rate
            self.total_fees += fee
            
            # Calculate actual slippage for tracking
            slippage = abs(executed_price - price) / price
            self.total_slippage += slippage
            
            # Store trade record
            trade_record = {
                'id': trade_id,
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side.lower(),
                'quantity': quantity,
                'requested_price': price,
                'executed_price': executed_price,
                'slippage': slippage,
                'fee': fee,
                'order_type': order_type,
                'status': 'FILLED',
                'execution_time_ms': delay * 1000
            }
            
            self.executed_trades.append(trade_record)
            
            # Log to trade logger with detailed info
            self.trade_logger.info(
                f"TRADE_EXECUTED | {trade_record['timestamp'].isoformat()} | "
                f"{symbol} | {side.upper()} | {quantity:.6f} | "
                f"${executed_price:.2f} | Slippage: {slippage:.4f} | Fee: ${fee:.4f}"
            )
            
            # Log execution details
            if slippage > 0.002:  # More than 0.2% slippage
                self.logger.warning(f"⚠️ High slippage on {symbol}: {slippage:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Simulated execution error: {e}")
            return False
    
    def _calculate_slippage(self, price: float, side: str, order_type: str, trade_id: str) -> float:
        """Calculate realistic slippage based on order type and market conditions"""
        try:
            if order_type == 'limit':
                # Limit orders typically have no slippage (filled at requested price)
                return price
            
            # Market order slippage (0.01% to 0.5% based on various factors)
            base_slippage = 0.0001  # 0.01% base slippage
            
            # Add random component based on trade_id (deterministic for testing)
            random_factor = (hash(trade_id) % 100) / 10000  # 0-0.01%
            
            # Simulate higher slippage during volatile periods (randomly)
            volatility_factor = 0
            if hash(trade_id) % 10 == 0:  # 10% chance of high volatility
                volatility_factor = 0.002  # Extra 0.2% slippage
            
            total_slippage = base_slippage + random_factor + volatility_factor
            
            # Apply slippage direction based on side
            if side.lower() == 'buy':
                # Buying typically gets filled at slightly higher price
                return price * (1 + total_slippage)
            else:
                # Selling typically gets filled at slightly lower price
                return price * (1 - total_slippage)
                
        except Exception as e:
            self.logger.error(f"❌ Slippage calculation error: {e}")
            return price
    
    async def _execute_real_trade(self, trade_id: str, symbol: str, side: str, 
                                quantity: float, price: float, order_type: str) -> bool:
        """Execute real trade on exchange"""
        try:
            if not self.exchange:
                self.logger.error("❌ No exchange connection available")
                return False
            
            # Execute order based on type
            order = None
            if order_type == 'market':
                order = self.exchange.create_market_order(symbol, side, quantity)
            elif order_type == 'limit':
                order = self.exchange.create_limit_order(symbol, side, quantity, price)
            else:
                self.logger.error(f"❌ Unsupported order type: {order_type}")
                return False
            
            # Check order status
            if order and order.get('status') in ['closed', 'filled']:
                # Store real trade record
                trade_record = {
                    'id': trade_id,
                    'exchange_id': order.get('id'),
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': side.lower(),
                    'quantity': quantity,
                    'requested_price': price,
                    'executed_price': order.get('average', price),
                    'fee': order.get('fee', {}).get('cost', 0),
                    'order_type': order_type,
                    'status': 'FILLED'
                }
                
                self.executed_trades.append(trade_record)
                
                # Log real trade
                self.trade_logger.info(
                    f"REAL_TRADE | {trade_record['timestamp'].isoformat()} | "
                    f"{symbol} | {side.upper()} | {quantity:.6f} | "
                    f"${trade_record['executed_price']:.2f} | Exchange ID: {order.get('id')}"
                )
                
                self.logger.info(f"✅ Real trade executed: {order['id']}")
                return True
            else:
                self.logger.error(f"❌ Real trade failed: {order}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Real execution error: {e}")
            return False
    
    def _log_successful_trade(self, trade_id: str, symbol: str, side: str, quantity: float, price: float):
        """Log successful trade details"""
        try:
            self.logger.info(f"📋 Trade Success: {trade_id}")
            self.logger.info(f"   Symbol: {symbol}")
            self.logger.info(f"   Side: {side.upper()}")
            self.logger.info(f"   Quantity: {quantity:.6f}")
            self.logger.info(f"   Price: ${price:.2f}")
            self.logger.info(f"   Value: ${quantity * price:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Trade logging error: {e}")
    
    def _log_failed_trade(self, trade_id: str, symbol: str, side: str, quantity: float, price: float):
        """Log failed trade details"""
        try:
            failure_record = {
                'id': trade_id,
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'requested_price': price,
                'reason': 'Execution failed'
            }
            
            self.failed_trades.append(failure_record)
            
            self.logger.warning(f"⚠️ Trade Failed: {trade_id}")
            self.logger.warning(f"   Symbol: {symbol}")
            self.logger.warning(f"   Side: {side.upper()}")
            self.logger.warning(f"   Quantity: {quantity:.6f}")
            self.logger.warning(f"   Price: ${price:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed trade logging error: {e}")
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a pending order"""
        try:
            if self.config.SANDBOX_MODE:
                # Simulate order cancellation
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
                    self.logger.info(f"✅ Simulated order cancelled: {order_id}")
                    return True
                else:
                    self.logger.warning(f"⚠️ Order not found for cancellation: {order_id}")
                    return False
            else:
                # Real order cancellation
                if self.exchange:
                    result = self.exchange.cancel_order(order_id, symbol)
                    self.logger.info(f"✅ Real order cancelled: {order_id}")
                    return True
                else:
                    self.logger.error("❌ No exchange connection for order cancellation")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Order cancellation error: {e}")
            return False
    
    def get_execution_stats(self) -> Dict:
        """Get comprehensive execution statistics"""
        try:
            if self.total_trades == 0:
                return {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'success_rate': 0,
                    'total_fees': 0,
                    'avg_slippage': 0,
                    'total_volume': 0
                }
            
            success_rate = (self.successful_trades / self.total_trades) * 100
            avg_slippage = (self.total_slippage / self.successful_trades) * 100 if self.successful_trades > 0 else 0
            
            # Calculate total trading volume
            total_volume = sum(trade['quantity'] * trade['executed_price'] 
                             for trade in self.executed_trades)
            
            return {
                'total_trades': self.total_trades,
                'successful_trades': self.successful_trades,
                'failed_trades': len(self.failed_trades),
                'success_rate': success_rate,
                'total_fees': self.total_fees,
                'avg_slippage': avg_slippage,
                'total_volume': total_volume,
                'trades_today': self._count_trades_today(),
                'last_trade': self.executed_trades[-1]['timestamp'] if self.executed_trades else None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating execution stats: {e}")
            return {
                'total_trades': 0, 'successful_trades': 0, 'success_rate': 0,
                'total_fees': 0, 'avg_slippage': 0, 'total_volume': 0
            }
    
    def _count_trades_today(self) -> int:
        """Count trades executed today"""
        try:
            today = datetime.now().date()
            return len([trade for trade in self.executed_trades 
                       if trade['timestamp'].date() == today])
        except:
            return 0
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trade history"""
        try:
            return self.executed_trades[-limit:] if self.executed_trades else []
        except Exception as e:
            self.logger.error(f"❌ Error getting recent trades: {e}")
            return []
    
    def get_trade_summary_by_symbol(self) -> Dict:
        """Get trading summary grouped by symbol"""
        try:
            summary = {}
            
            for trade in self.executed_trades:
                symbol = trade['symbol']
                if symbol not in summary:
                    summary[symbol] = {
                        'total_trades': 0,
                        'buy_trades': 0,
                        'sell_trades': 0,
                        'total_volume': 0,
                        'total_fees': 0,
                        'avg_slippage': 0
                    }
                
                summary[symbol]['total_trades'] += 1
                if trade['side'] == 'buy':
                    summary[symbol]['buy_trades'] += 1
                else:
                    summary[symbol]['sell_trades'] += 1
                
                summary[symbol]['total_volume'] += trade['quantity'] * trade['executed_price']
                summary[symbol]['total_fees'] += trade['fee']
                summary[symbol]['avg_slippage'] += trade.get('slippage', 0)
            
            # Calculate averages
            for symbol_data in summary.values():
                if symbol_data['total_trades'] > 0:
                    symbol_data['avg_slippage'] /= symbol_data['total_trades']
                    symbol_data['avg_slippage'] *= 100  # Convert to percentage
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error generating trade summary: {e}")
            return {}