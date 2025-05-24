# trading/grid_strategy.py - Grid trading strategy implementation
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

class GridStrategy:
    """
    Advanced grid trading strategy implementation
    Places buy orders below current price and sell orders above,
    profiting from price oscillations in both directions
    """
    
    def __init__(self, config, data_collector):
        self.config = config
        self.data_collector = data_collector
        self.logger = logging.getLogger(__name__)
        
        # Grid state for each symbol
        self.active_grids = {}      # symbol -> grid configuration
        self.grid_orders = {}       # symbol -> list of active orders
        self.filled_orders = {}     # symbol -> list of filled orders
        self.grid_profits = {}      # symbol -> total profit from grid
        
        # Grid performance tracking
        self.grid_stats = {}        # symbol -> performance statistics
        
    def has_active_grid(self, symbol: str) -> bool:
        """Check if symbol has an active grid"""
        return symbol in self.active_grids and self.active_grids[symbol]['status'] == 'active'
    
    async def create_grid(self, symbol: str, center_price: float, volatility: float) -> bool:
        """Create a new grid for a symbol"""
        try:
            self.logger.info(f"🔥 Creating grid for {symbol} at ${center_price:,.2f}")
            
            # Check if conditions are suitable for grid trading
            if not self._is_suitable_for_grid(symbol, center_price, volatility):
                self.logger.info(f"⚠️ Market conditions not suitable for grid: {symbol}")
                return False
            
            # Calculate optimal grid parameters
            grid_config = self._calculate_grid_parameters(symbol, center_price, volatility)
            if not grid_config:
                return False
            
            # Create grid orders
            orders = self._create_grid_orders(symbol, grid_config)
            if not orders:
                return False
            
            # Store grid configuration
            self.active_grids[symbol] = {
                **grid_config,
                'status': 'active',
                'created_at': datetime.now(),
                'total_invested': 0,
                'realized_profit': 0,
                'trades_completed': 0
            }
            
            self.grid_orders[symbol] = orders
            self.filled_orders[symbol] = []
            self.grid_profits[symbol] = 0
            
            # Initialize performance tracking
            self.grid_stats[symbol] = {
                'total_cycles': 0,
                'successful_cycles': 0,
                'avg_cycle_profit': 0,
                'max_drawdown': 0,
                'grid_efficiency': 0
            }
            
            self.logger.info(f"✅ Grid created for {symbol}:")
            self.logger.info(f"   📊 Spacing: {grid_config['spacing_percent']:.2f}%")
            self.logger.info(f"   📈 Buy levels: {len(grid_config['buy_levels'])}")
            self.logger.info(f"   📉 Sell levels: {len(grid_config['sell_levels'])}")
            self.logger.info(f"   💰 Investment: ${grid_config['total_investment']:,.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error creating grid for {symbol}: {e}")
            return False
    
    def _is_suitable_for_grid(self, symbol: str, price: float, volatility: float) -> bool:
        """Check if market conditions are suitable for grid trading"""
        try:
            # Get recent price data for analysis
            historical_data = self.data_collector.get_historical_data(symbol)
            if historical_data is None or len(historical_data) < 50:
                return False
            
            # Check volatility range (not too high, not too low)
            if volatility < self.config.GRID_MIN_VOLATILITY:
                self.logger.info(f"⚠️ Volatility too low for grid: {volatility:.3f}")
                return False
            
            if volatility > 0.08:  # Too volatile for stable grid
                self.logger.info(f"⚠️ Volatility too high for grid: {volatility:.3f}")
                return False
            
            # Check if price is in a suitable range (not at extreme levels)
            recent_prices = historical_data['close'].tail(50)
            price_percentile = (price - recent_prices.min()) / (recent_prices.max() - recent_prices.min())
            
            if price_percentile < 0.2 or price_percentile > 0.8:
                self.logger.info(f"⚠️ Price at extreme level for grid: {price_percentile:.2f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error checking grid suitability: {e}")
            return False
    
    def _calculate_grid_parameters(self, symbol: str, center_price: float, volatility: float) -> Optional[Dict]:
        """Calculate optimal grid parameters based on market conditions"""
        try:
            # Adjust grid spacing based on volatility
            base_spacing = self.config.GRID_SPACING_PERCENT
            
            if volatility > 0.05:
                # High volatility - wider spacing
                spacing = base_spacing * 1.5
            elif volatility < 0.02:
                # Low volatility - tighter spacing
                spacing = base_spacing * 0.7
            else:
                # Normal volatility
                spacing = base_spacing
            
            # Ensure spacing is within reasonable bounds
            spacing = max(0.3, min(2.0, spacing))
            
            # Calculate grid levels
            grid_size = self.config.GRID_SIZE
            buy_levels = []
            sell_levels = []
            
            # Buy orders (below current price)
            for i in range(1, grid_size + 1):
                price_level = center_price * (1 - (spacing / 100) * i)
                buy_levels.append(price_level)
            
            # Sell orders (above current price)
            for i in range(1, grid_size + 1):
                price_level = center_price * (1 + (spacing / 100) * i)
                sell_levels.append(price_level)
            
            # Calculate total investment required
            total_investment = len(buy_levels) * self.config.BASE_ORDER_SIZE
            
            # Check if investment is within limits
            if total_investment > self.config.MAX_GRID_INVESTMENT:
                # Reduce order size to fit within limits
                adjusted_order_size = self.config.MAX_GRID_INVESTMENT / len(buy_levels)
                self.logger.info(f"📊 Adjusting order size: ${self.config.BASE_ORDER_SIZE} -> ${adjusted_order_size:.2f}")
            else:
                adjusted_order_size = self.config.BASE_ORDER_SIZE
            
            return {
                'center_price': center_price,
                'spacing_percent': spacing,
                'buy_levels': buy_levels,
                'sell_levels': sell_levels,
                'order_size_usd': adjusted_order_size,
                'total_investment': len(buy_levels) * adjusted_order_size,
                'grid_range': {
                    'lower': min(buy_levels),
                    'upper': max(sell_levels),
                    'range_percent': ((max(sell_levels) - min(buy_levels)) / center_price) * 100
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating grid parameters: {e}")
            return None
    
    def _create_grid_orders(self, symbol: str, grid_config: Dict) -> List[Dict]:
        """Create simulated grid orders"""
        try:
            orders = []
            order_size_usd = grid_config['order_size_usd']
            
            # Create buy orders
            for i, price in enumerate(grid_config['buy_levels']):
                quantity = order_size_usd / price
                order = {
                    'id': f"grid_buy_{symbol}_{i}_{int(datetime.now().timestamp())}",
                    'symbol': symbol,
                    'side': 'buy',
                    'price': price,
                    'quantity': quantity,
                    'order_size_usd': order_size_usd,
                    'status': 'open',
                    'created_at': datetime.now(),
                    'grid_level': i + 1,
                    'type': 'grid_buy'
                }
                orders.append(order)
            
            # Create sell orders (simulate having some inventory)
            # In reality, you'd either start with existing holdings or
            # only create sell orders as buy orders get filled
            base_quantity = order_size_usd / grid_config['center_price']
            
            for i, price in enumerate(grid_config['sell_levels']):
                order = {
                    'id': f"grid_sell_{symbol}_{i}_{int(datetime.now().timestamp())}",
                    'symbol': symbol,
                    'side': 'sell',
                    'price': price,
                    'quantity': base_quantity,
                    'order_size_usd': order_size_usd,
                    'status': 'open',
                    'created_at': datetime.now(),
                    'grid_level': i + 1,
                    'type': 'grid_sell'
                }
                orders.append(order)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"❌ Error creating grid orders: {e}")
            return []
    
    async def manage_grid(self, symbol: str, current_price: float):
        """Manage existing grid orders - check for fills and create new orders"""
        try:
            if symbol not in self.grid_orders:
                return
            
            orders = self.grid_orders[symbol]
            filled_orders = []
            new_orders = []
            
            for order in orders:
                if order['status'] == 'open' and self._should_fill_order(order, current_price):
                    # Fill the order
                    order['status'] = 'filled'
                    order['filled_at'] = datetime.now()
                    order['filled_price'] = current_price
                    filled_orders.append(order)
                    
                    # Process the filled order
                    profit, new_order = await self._process_filled_order(symbol, order, current_price)
                    
                    if new_order:
                        new_orders.append(new_order)
                    
                    if profit > 0:
                        self.grid_profits[symbol] += profit
                        self.active_grids[symbol]['realized_profit'] += profit
                        self.active_grids[symbol]['trades_completed'] += 1
                        
                        self.logger.info(f"💰 Grid profit for {symbol}: ${profit:.2f} "
                                       f"(Total: ${self.grid_profits[symbol]:.2f})")
            
            # Add new orders to the grid
            if new_orders:
                orders.extend(new_orders)
            
            # Update grid statistics
            if filled_orders:
                self._update_grid_stats(symbol, filled_orders)
            
            # Remove filled orders from active list (keep them in filled_orders for tracking)
            self.grid_orders[symbol] = [o for o in orders if o['status'] == 'open']
            self.filled_orders[symbol].extend(filled_orders)
            
        except Exception as e:
            self.logger.error(f"❌ Error managing grid for {symbol}: {e}")
    
    def _should_fill_order(self, order: Dict, current_price: float) -> bool:
        """Determine if an order should be filled based on current price"""
        try:
            if order['side'] == 'buy':
                # Buy order fills when current price drops to or below order price
                return current_price <= order['price']
            else:  # sell order
                # Sell order fills when current price rises to or above order price
                return current_price >= order['price']
        except Exception as e:
            self.logger.error(f"❌ Error checking order fill: {e}")
            return False
    
    async def _process_filled_order(self, symbol: str, filled_order: Dict, current_price: float) -> tuple:
        """Process a filled order and create corresponding opposite order"""
        try:
            profit = 0
            new_order = None
            
            grid_config = self.active_grids[symbol]
            spacing_percent = grid_config['spacing_percent']
            
            if filled_order['side'] == 'buy':
                # Buy order filled - create corresponding sell order
                sell_price = filled_order['price'] * (1 + spacing_percent / 100)
                
                new_order = {
                    'id': f"grid_sell_{symbol}_{sell_price:.2f}_{int(datetime.now().timestamp())}",
                    'symbol': symbol,
                    'side': 'sell',
                    'price': sell_price,
                    'quantity': filled_order['quantity'],
                    'order_size_usd': filled_order['order_size_usd'],
                    'status': 'open',
                    'created_at': datetime.now(),
                    'paired_with': filled_order['id'],
                    'type': 'grid_sell'
                }
                
                self.logger.info(f"🟢 Grid BUY filled: {symbol} @ ${filled_order['price']:.2f}")
                
            else:  # sell order filled
                # Sell order filled - calculate profit and create buy order
                if 'paired_with' in filled_order:
                    # Find the paired buy order to calculate exact profit
                    buy_price = self._find_paired_buy_price(symbol, filled_order['paired_with'])
                    if buy_price:
                        profit = (filled_order['price'] - buy_price) * filled_order['quantity']
                
                buy_price = filled_order['price'] * (1 - spacing_percent / 100)
                
                new_order = {
                    'id': f"grid_buy_{symbol}_{buy_price:.2f}_{int(datetime.now().timestamp())}",
                    'symbol': symbol,
                    'side': 'buy',
                    'price': buy_price,
                    'quantity': filled_order['order_size_usd'] / buy_price,
                    'order_size_usd': filled_order['order_size_usd'],
                    'status': 'open',
                    'created_at': datetime.now(),
                    'type': 'grid_buy'
                }
                
                self.logger.info(f"🔴 Grid SELL filled: {symbol} @ ${filled_order['price']:.2f} "
                               f"(Profit: ${profit:.2f})")
            
            return profit, new_order
            
        except Exception as e:
            self.logger.error(f"❌ Error processing filled order: {e}")
            return 0, None
    
    def _find_paired_buy_price(self, symbol: str, paired_order_id: str) -> Optional[float]:
        """Find the buy price for a paired sell order"""
        try:
            # Search in filled orders
            filled_orders = self.filled_orders.get(symbol, [])
            for order in filled_orders:
                if order['id'] == paired_order_id:
                    return order['price']
            
            # Fallback: estimate based on grid spacing
            grid_config = self.active_grids.get(symbol)
            if grid_config:
                spacing = grid_config['spacing_percent'] / 100
                # This is an approximation
                return grid_config['center_price'] * (1 - spacing)
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error finding paired buy price: {e}")
            return None
    
    def _update_grid_stats(self, symbol: str, filled_orders: List[Dict]):
        """Update grid performance statistics"""
        try:
            stats = self.grid_stats[symbol]
            
            # Count completed cycles (buy -> sell pairs)
            for order in filled_orders:
                if order['side'] == 'sell' and 'paired_with' in order:
                    stats['total_cycles'] += 1
                    stats['successful_cycles'] += 1  # All cycles are successful for now
            
            # Update average cycle profit
            if stats['successful_cycles'] > 0:
                stats['avg_cycle_profit'] = self.grid_profits[symbol] / stats['successful_cycles']
            
            # Calculate grid efficiency (profit per dollar invested)
            total_investment = self.active_grids[symbol]['total_investment']
            if total_investment > 0:
                stats['grid_efficiency'] = (self.grid_profits[symbol] / total_investment) * 100
                
        except Exception as e:
            self.logger.error(f"❌ Error updating grid stats: {e}")
    
    def get_grid_status(self, symbol: str) -> Dict:
        """Get comprehensive grid status for a symbol"""
        try:
            if symbol not in self.active_grids:
                return {'status': 'No grid'}
            
            grid_config = self.active_grids[symbol]
            orders = self.grid_orders.get(symbol, [])
            
            # Count active orders by type
            active_buy_orders = len([o for o in orders if o['side'] == 'buy' and o['status'] == 'open'])
            active_sell_orders = len([o for o in orders if o['side'] == 'sell' and o['status'] == 'open'])
            
            # Get recent performance
            total_profit = self.grid_profits.get(symbol, 0)
            trades_completed = grid_config.get('trades_completed', 0)
            stats = self.grid_stats.get(symbol, {})
            
            # Calculate grid health
            total_orders = active_buy_orders + active_sell_orders
            expected_orders = self.config.GRID_SIZE * 2
            grid_health = (total_orders / expected_orders) * 100 if expected_orders > 0 else 0
            
            return {
                'status': 'Active',
                'active_orders': total_orders,
                'active_buy_orders': active_buy_orders,
                'active_sell_orders': active_sell_orders,
                'total_profit': total_profit,
                'trades_completed': trades_completed,
                'center_price': grid_config['center_price'],
                'spacing_percent': grid_config['spacing_percent'],
                'grid_range': grid_config['grid_range'],
                'total_investment': grid_config['total_investment'],
                'grid_health': grid_health,
                'performance': {
                    'total_cycles': stats.get('total_cycles', 0),
                    'avg_cycle_profit': stats.get('avg_cycle_profit', 0),
                    'grid_efficiency': stats.get('grid_efficiency', 0)
                },
                'created_at': grid_config['created_at'],
                'uptime': datetime.now() - grid_config['created_at']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting grid status for {symbol}: {e}")
            return {'status': 'Error', 'error': str(e)}
    
    async def stop_grid(self, symbol: str) -> bool:
        """Stop and remove grid for a symbol"""
        try:
            if symbol not in self.active_grids:
                self.logger.warning(f"⚠️ No active grid found for {symbol}")
                return False
            
            # Mark grid as stopped
            self.active_grids[symbol]['status'] = 'stopped'
            self.active_grids[symbol]['stopped_at'] = datetime.now()
            
            # In a real implementation, you'd cancel all open orders here
            active_orders = len(self.grid_orders.get(symbol, []))
            
            self.logger.info(f"🛑 Grid stopped for {symbol}")
            self.logger.info(f"   📊 Total profit: ${self.grid_profits.get(symbol, 0):.2f}")
            self.logger.info(f"   🔄 Trades completed: {self.active_grids[symbol].get('trades_completed', 0)}")
            self.logger.info(f"   📋 Active orders cancelled: {active_orders}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping grid for {symbol}: {e}")
            return False
    
    def get_all_grids_summary(self) -> Dict:
        """Get summary of all active grids"""
        try:
            summary = {
                'total_grids': len(self.active_grids),
                'active_grids': len([g for g in self.active_grids.values() if g['status'] == 'active']),
                'total_profit': sum(self.grid_profits.values()),
                'total_investment': sum(g['total_investment'] for g in self.active_grids.values()),
                'grids': {}
            }
            
            for symbol in self.active_grids:
                summary['grids'][symbol] = self.get_grid_status(symbol)
            
            # Calculate overall performance
            if summary['total_investment'] > 0:
                summary['overall_efficiency'] = (summary['total_profit'] / summary['total_investment']) * 100
            else:
                summary['overall_efficiency'] = 0
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error generating grids summary: {e}")
            return {'total_grids': 0, 'active_grids': 0, 'total_profit': 0}