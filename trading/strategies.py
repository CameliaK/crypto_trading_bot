# trading/strategies.py - Complete strategy management system
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from technical.indicators import TechnicalAnalyzer
from trading.grid_strategy import GridStrategy

class StrategyManager:
    """
    Central strategy manager that coordinates all trading strategies
    """
    
    def __init__(self, config, data_collector, risk_manager, trade_executor):
        self.config = config
        self.data_collector = data_collector
        self.risk_manager = risk_manager
        self.trade_executor = trade_executor
        self.logger = logging.getLogger(__name__)
        
        # Initialize analyzers and strategies
        self.technical_analyzer = TechnicalAnalyzer(config)
        self.grid_strategy = GridStrategy(config, data_collector)
        
        # Strategy state
        self.latest_analysis = {}      # symbol -> latest analysis
        self.active_strategies = {}    # symbol -> active strategy type
        self.strategy_performance = {} # strategy -> performance metrics
        self.total_trades = 0
        self.total_pnl = 0.0
        
        # Portfolio tracking
        self.positions = {}           # symbol -> position info
        self.portfolio_value = config.INITIAL_BALANCE
        self.available_balance = config.INITIAL_BALANCE
        
        # Strategy execution tracking
        self.last_strategy_check = {}  # symbol -> last check time
        self.strategy_cooldown = 30    # seconds between strategy changes
        
    async def run_strategy_loop(self):
        """Main strategy execution loop"""
        self.logger.info("🎯 Starting strategy execution loop...")
        
        # Wait for initial data
        while not self.data_collector.is_data_ready():
            self.logger.info("⏳ Waiting for market data...")
            await asyncio.sleep(5)
        
        self.logger.info("✅ Market data ready, starting strategies...")
        
        while True:
            try:
                # Process each symbol
                for symbol in self.config.SYMBOLS:
                    await self.process_symbol(symbol)
                
                # Update portfolio metrics
                self.update_portfolio()
                
                await asyncio.sleep(self.config.STRATEGY_UPDATE_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"❌ Error in strategy loop: {e}")
                await asyncio.sleep(10)
    
    async def process_symbol(self, symbol: str):
        """Process trading signals for a single symbol"""
        try:
            # Get market data
            price_data = self.data_collector.get_price_data(symbol)
            historical_data = self.data_collector.get_historical_data(symbol)
            
            if not price_data or historical_data is None or len(historical_data) < 50:
                return
            
            current_price = price_data['price']
            volatility = self.data_collector.get_volatility(symbol)
            
            # 1. Technical Analysis
            technical_analysis = self.technical_analyzer.analyze_symbol(symbol, historical_data)
            
            # 2. Market Regime Detection
            market_regime = self.detect_market_regime(symbol, historical_data)
            
            # 3. Strategy Selection based on market conditions
            optimal_strategy = self.select_optimal_strategy(symbol, technical_analysis, market_regime, volatility)
            
            # 4. Execute strategy (with cooldown to prevent rapid switching)
            if self._can_change_strategy(symbol):
                await self.execute_strategy(symbol, optimal_strategy, technical_analysis, current_price, volatility)
            
            # 5. Manage existing positions and strategies
            await self.manage_existing_strategies(symbol, current_price, technical_analysis)
            
            # Store latest analysis
            self.latest_analysis[symbol] = {
                **technical_analysis,
                'market_regime': market_regime,
                'selected_strategy': optimal_strategy,
                'volatility': volatility,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error processing {symbol}: {e}")
    
    def detect_market_regime(self, symbol: str, df: pd.DataFrame) -> str:
        """
        Detect current market regime (Trending Up/Down, Sideways, Volatile)
        """
        try:
            if len(df) < 50:
                return 'UNKNOWN'
            
            prices = df['close']
            returns = prices.pct_change().dropna()
            
            # Calculate metrics for regime detection
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            # Trend analysis
            sma_20 = prices.rolling(window=20).mean()
            sma_50 = prices.rolling(window=50).mean()
            
            current_price = prices.iloc[-1]
            sma_20_current = sma_20.iloc[-1]
            sma_50_current = sma_50.iloc[-1]
            
            # Trend strength
            trend_strength = abs(current_price - sma_20_current) / sma_20_current
            
            # Moving average slopes
            sma_20_slope = (sma_20.iloc[-1] - sma_20.iloc[-5]) / sma_20.iloc[-5]
            sma_50_slope = (sma_50.iloc[-1] - sma_50.iloc[-10]) / sma_50.iloc[-10]
            
            # Price position relative to moving averages
            above_sma20 = current_price > sma_20_current
            above_sma50 = current_price > sma_50_current
            sma20_above_sma50 = sma_20_current > sma_50_current
            
            # Volume analysis
            volume_data = self.data_collector.get_volume_data(symbol)
            volume_ratio = volume_data.get('ratio', 1.0) if volume_data else 1.0
            
            # Market regime logic
            if volatility > 0.06:  # Very high volatility
                return 'HIGHLY_VOLATILE'
            elif volatility > 0.04:  # High volatility
                return 'VOLATILE'
            elif abs(sma_20_slope) > 0.02 and abs(sma_50_slope) > 0.015:  # Strong trend
                if sma_20_slope > 0 and sma_50_slope > 0 and above_sma20 and above_sma50:
                    return 'TRENDING_UP'
                elif sma_20_slope < 0 and sma_50_slope < 0 and not above_sma20 and not above_sma50:
                    return 'TRENDING_DOWN'
                else:
                    return 'MIXED_TREND'
            elif volatility < 0.015 and abs(sma_20_slope) < 0.01:  # Low volatility, no clear trend
                return 'SIDEWAYS'
            else:
                return 'CONSOLIDATING'
                
        except Exception as e:
            self.logger.error(f"❌ Error detecting market regime for {symbol}: {e}")
            return 'UNKNOWN'
    
    def select_optimal_strategy(self, symbol: str, analysis: Dict, market_regime: str, volatility: float) -> str:
        """
        Select the optimal strategy based on market conditions
        """
        try:
            rsi = analysis.get('rsi', 50)
            signal = analysis.get('signal', 'HOLD')
            signal_strength = analysis.get('strength', 0)
            
            # Strategy selection logic based on market regime
            if market_regime == 'SIDEWAYS' and self.config.GRID_RSI_MIN <= rsi <= self.config.GRID_RSI_MAX:
                # Perfect conditions for grid trading
                return 'GRID'
            
            elif market_regime in ['TRENDING_UP', 'TRENDING_DOWN']:
                if signal_strength > 1.0:
                    # Strong trend with high confidence signal
                    return 'MOMENTUM'
                else:
                    # Trend but weak signal - use technical analysis
                    return 'TECHNICAL'
            
            elif market_regime == 'VOLATILE' or market_regime == 'HIGHLY_VOLATILE':
                if volatility > 0.08:
                    # Too volatile - be cautious
                    return 'CONSERVATIVE'
                else:
                    # Moderate volatility - use scalping approach
                    return 'SCALPING'
            
            elif market_regime == 'CONSOLIDATING':
                # Consolidation phase - wait for breakout
                if signal_strength > 1.2:
                    return 'BREAKOUT'
                else:
                    return 'WAIT'
            
            elif signal_strength > 1.5:
                # Very strong technical signal regardless of regime
                return 'TECHNICAL'
            
            else:
                # Default strategy
                return 'TECHNICAL'
                
        except Exception as e:
            self.logger.error(f"❌ Error selecting strategy for {symbol}: {e}")
            return 'TECHNICAL'
    
    def _can_change_strategy(self, symbol: str) -> bool:
        """Check if enough time has passed to change strategy"""
        if symbol not in self.last_strategy_check:
            self.last_strategy_check[symbol] = datetime.now()
            return True
        
        time_since_last = (datetime.now() - self.last_strategy_check[symbol]).total_seconds()
        return time_since_last >= self.strategy_cooldown
    
    async def execute_strategy(self, symbol: str, strategy: str, analysis: Dict, 
                             current_price: float, volatility: float):
        """
        Execute the selected strategy
        """
        try:
            # Check risk management first
            if not self.risk_manager.can_open_position(symbol, current_price):
                return
            
            # Update last strategy check time
            self.last_strategy_check[symbol] = datetime.now()
            
            # Log strategy selection
            if symbol not in self.active_strategies or self.active_strategies[symbol] != strategy:
                self.logger.info(f"🎯 Strategy for {symbol}: {strategy} "
                               f"(Regime: {analysis.get('market_regime', 'Unknown')}, "
                               f"RSI: {analysis.get('rsi', 50):.1f})")
            
            if strategy == 'GRID':
                await self.execute_grid_strategy(symbol, analysis, current_price, volatility)
            elif strategy == 'MOMENTUM':
                await self.execute_momentum_strategy(symbol, analysis, current_price)
            elif strategy == 'TECHNICAL':
                await self.execute_technical_strategy(symbol, analysis, current_price)
            elif strategy == 'SCALPING':
                await self.execute_scalping_strategy(symbol, analysis, current_price)
            elif strategy == 'BREAKOUT':
                await self.execute_breakout_strategy(symbol, analysis, current_price)
            elif strategy == 'CONSERVATIVE':
                await self.execute_conservative_strategy(symbol, analysis, current_price)
            elif strategy == 'WAIT':
                # Do nothing - wait for better conditions
                pass
            
            # Update active strategy
            self.active_strategies[symbol] = strategy
            
        except Exception as e:
            self.logger.error(f"❌ Error executing {strategy} strategy for {symbol}: {e}")
    
    async def execute_grid_strategy(self, symbol: str, analysis: Dict, current_price: float, volatility: float):
        """Execute grid trading strategy"""
        try:
            # Check if grid already exists
            if not self.grid_strategy.has_active_grid(symbol):
                # Create new grid
                success = await self.grid_strategy.create_grid(symbol, current_price, volatility)
                if success:
                    self.logger.info(f"🔥 Grid strategy activated for {symbol}")
            else:
                # Manage existing grid
                await self.grid_strategy.manage_grid(symbol, current_price)
                
        except Exception as e:
            self.logger.error(f"❌ Error in grid strategy for {symbol}: {e}")
    
    async def execute_momentum_strategy(self, symbol: str, analysis: Dict, current_price: float):
        """Execute momentum trading strategy"""
        try:
            signal = analysis.get('signal', 'HOLD')
            strength = analysis.get('strength', 0)
            rsi = analysis.get('rsi', 50)
            
            # Strong momentum conditions
            if signal in ['BUY', 'SELL'] and strength > 1.0:
                # Additional filters for momentum trades
                if signal == 'BUY' and rsi < 65:  # Not too overbought
                    await self._execute_trade(symbol, 'buy', current_price, 'momentum')
                elif signal == 'SELL' and rsi > 35:  # Not too oversold
                    await self._execute_trade(symbol, 'sell', current_price, 'momentum')
                        
        except Exception as e:
            self.logger.error(f"❌ Error in momentum strategy for {symbol}: {e}")
    
    async def execute_technical_strategy(self, symbol: str, analysis: Dict, current_price: float):
        """Execute pure technical analysis strategy"""
        try:
            signal = analysis.get('signal', 'HOLD')
            strength = analysis.get('strength', 0)
            rsi = analysis.get('rsi', 50)
            
            # Strong technical signals with RSI confirmation
            if (signal == 'BUY' and strength > 0.8 and rsi < 45) or \
               (signal == 'SELL' and strength > 0.8 and rsi > 55):
                
                await self._execute_trade(symbol, signal.lower(), current_price, 'technical')
                        
        except Exception as e:
            self.logger.error(f"❌ Error in technical strategy for {symbol}: {e}")
    
    async def execute_scalping_strategy(self, symbol: str, analysis: Dict, current_price: float):
        """Execute scalping strategy for volatile markets"""
        try:
            rsi = analysis.get('rsi', 50)
            
            # Scalping based on extreme RSI levels
            if rsi < 25:  # Extremely oversold
                await self._execute_trade(symbol, 'buy', current_price, 'scalping', risk_multiplier=0.5)
            elif rsi > 75:  # Extremely overbought
                await self._execute_trade(symbol, 'sell', current_price, 'scalping', risk_multiplier=0.5)
                        
        except Exception as e:
            self.logger.error(f"❌ Error in scalping strategy for {symbol}: {e}")
    
    async def execute_breakout_strategy(self, symbol: str, analysis: Dict, current_price: float):
        """Execute breakout strategy"""
        try:
            signal = analysis.get('signal', 'HOLD')
            strength = analysis.get('strength', 0)
            bb_data = analysis.get('bollinger', {})
            
            # Breakout from Bollinger Bands
            bb_position = bb_data.get('position', 0.5)
            
            if signal == 'BUY' and strength > 1.2 and bb_position > 0.95:
                # Upward breakout
                await self._execute_trade(symbol, 'buy', current_price, 'breakout')
            elif signal == 'SELL' and strength > 1.2 and bb_position < 0.05:
                # Downward breakout
                await self._execute_trade(symbol, 'sell', current_price, 'breakout')
                        
        except Exception as e:
            self.logger.error(f"❌ Error in breakout strategy for {symbol}: {e}")
    
    async def execute_conservative_strategy(self, symbol: str, analysis: Dict, current_price: float):
        """Execute conservative strategy for uncertain conditions"""
        try:
            signal = analysis.get('signal', 'HOLD')
            strength = analysis.get('strength', 0)
            rsi = analysis.get('rsi', 50)
            
            # Only trade on very strong signals with conservative position sizing
            if strength > 2.0:  # Very high threshold
                if signal == 'BUY' and rsi < 40:
                    await self._execute_trade(symbol, 'buy', current_price, 'conservative', risk_multiplier=0.3)
                elif signal == 'SELL' and rsi > 60:
                    await self._execute_trade(symbol, 'sell', current_price, 'conservative', risk_multiplier=0.3)
                        
        except Exception as e:
            self.logger.error(f"❌ Error in conservative strategy for {symbol}: {e}")
    
    async def _execute_trade(self, symbol: str, side: str, price: float, strategy: str, risk_multiplier: float = 1.0):
        """Helper method to execute a trade with proper risk management"""
        try:
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol, price, self.available_balance, risk_multiplier
            )
            
            if position_size > 0:
                # Execute trade
                success = await self.trade_executor.execute_trade(
                    symbol, side, position_size, price
                )
                
                if success:
                    # Register position with risk manager
                    self.risk_manager.register_position(symbol, price, position_size, side)
                    
                    # Record trade
                    await self.record_trade(symbol, side, position_size, price, strategy)
                    
                    self.logger.info(f"📈 {strategy.upper()} {side.upper()}: {symbol} "
                                   f"{position_size:.6f} @ ${price:.2f}")
                        
        except Exception as e:
            self.logger.error(f"❌ Error executing trade: {e}")
    
    async def manage_existing_strategies(self, symbol: str, current_price: float, analysis: Dict):
        """Manage existing positions and strategies"""
        try:
            # Update unrealized P&L for risk manager
            self.risk_manager.update_unrealized_pnl(symbol, current_price)
            
            # Check if positions should be closed (stop loss/take profit)
            if self.risk_manager.should_close_position(symbol, current_price):
                # Close position
                pnl = self.risk_manager.close_position(symbol, current_price)
                
                # Execute closing trade
                position = self.risk_manager.open_positions.get(symbol)
                if position:
                    opposite_side = 'sell' if position['side'] == 'buy' else 'buy'
                    await self.trade_executor.execute_trade(
                        symbol, opposite_side, position['quantity'], current_price
                    )
                    
                    self.logger.info(f"🔒 Position closed: {symbol} P&L: ${pnl:.2f}")
            
            # Manage grid strategy if active
            if self.grid_strategy.has_active_grid(symbol):
                await self.grid_strategy.manage_grid(symbol, current_price)
                
        except Exception as e:
            self.logger.error(f"❌ Error managing existing strategies for {symbol}: {e}")
    
    async def record_trade(self, symbol: str, side: str, quantity: float, price: float, strategy: str):
        """Record a trade for tracking"""
        try:
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'value': quantity * price,
                'strategy': strategy
            }
            
            # Update position tracking
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0,
                    'avg_price': 0,
                    'total_invested': 0,
                    'realized_pnl': 0,
                    'trades': []
                }
            
            position = self.positions[symbol]
            
            if side.upper() == 'BUY':
                # Add to position
                total_value = position['quantity'] * position['avg_price'] + quantity * price
                total_quantity = position['quantity'] + quantity
                position['avg_price'] = total_value / total_quantity if total_quantity > 0 else price
                position['quantity'] = total_quantity
                position['total_invested'] += quantity * price
                self.available_balance -= quantity * price
                
            elif side.upper() == 'SELL':
                # Reduce position
                if position['quantity'] >= quantity:
                    # Calculate realized P&L
                    realized_pnl = (price - position['avg_price']) * quantity
                    position['realized_pnl'] += realized_pnl
                    position['quantity'] -= quantity
                    self.available_balance += quantity * price
                    self.total_pnl += realized_pnl
                    
                    if realized_pnl != 0:
                        self.logger.info(f"💰 Realized P&L for {symbol}: ${realized_pnl:.2f}")
            
            position['trades'].append(trade_record)
            self.total_trades += 1
            
        except Exception as e:
            self.logger.error(f"❌ Error recording trade: {e}")
    
    def update_portfolio(self):
        """Update portfolio metrics"""
        try:
            total_value = self.available_balance
            unrealized_pnl = 0
            
            # Calculate unrealized P&L
            for symbol, position in self.positions.items():
                if position['quantity'] > 0:
                    current_price = self.data_collector.get_latest_price(symbol)
                    if current_price:
                        position_value = position['quantity'] * current_price
                        total_value += position_value
                        unrealized_pnl += (current_price - position['avg_price']) * position['quantity']
            
            self.portfolio_value = total_value
            
        except Exception as e:
            self.logger.error(f"❌ Error updating portfolio: {e}")
    
    # ================================
    # PUBLIC INTERFACE METHODS
    # ================================
    
    def get_latest_analysis(self, symbol: str) -> Dict:
        """Get latest analysis for a symbol"""
        return self.latest_analysis.get(symbol, {
            'signal': 'HOLD',
            'strength': 0,
            'rsi': 50,
            'market_regime': 'UNKNOWN',
            'selected_strategy': 'NONE'
        })
    
    def get_strategy_status(self, symbol: str) -> str:
        """Get current strategy status for a symbol"""
        strategy = self.active_strategies.get(symbol, 'NONE')
        
        if strategy == 'GRID':
            grid_status = self.grid_strategy.get_grid_status(symbol)
            active_orders = grid_status.get('active_orders', 0)
            profit = grid_status.get('total_profit', 0)
            return f"Grid({active_orders} orders, ${profit:.0f})"
        elif strategy == 'MOMENTUM':
            return "Momentum"
        elif strategy == 'TECHNICAL':
            return "Technical"
        elif strategy == 'SCALPING':
            return "Scalping"
        elif strategy == 'BREAKOUT':
            return "Breakout"
        elif strategy == 'CONSERVATIVE':
            return "Conservative"
        elif strategy == 'WAIT':
            return "Waiting"
        else:
            return "Analyzing"
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        try:
            unrealized_pnl = 0
            active_positions = 0
            
            for symbol, position in self.positions.items():
                if position['quantity'] > 0:
                    active_positions += 1
                    current_price = self.data_collector.get_latest_price(symbol)
                    if current_price:
                        unrealized_pnl += (current_price - position['avg_price']) * position['quantity']
            
            # Add grid profits
            grid_profits = sum(self.grid_strategy.grid_profits.values())
            total_pnl = self.total_pnl + unrealized_pnl + grid_profits
            total_pnl_percent = (total_pnl / self.config.INITIAL_BALANCE) * 100
            
            return {
                'total_value': self.portfolio_value,
                'available_balance': self.available_balance,
                'total_pnl': total_pnl,
                'total_pnl_percent': total_pnl_percent,
                'realized_pnl': self.total_pnl,
                'unrealized_pnl': unrealized_pnl,
                'grid_profits': grid_profits,
                'active_positions': active_positions,
                'total_trades': self.total_trades
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating portfolio summary: {e}")
            return {
                'total_value': 0, 'available_balance': 0, 'total_pnl': 0,
                'total_pnl_percent': 0, 'realized_pnl': 0, 'unrealized_pnl': 0,
                'grid_profits': 0, 'active_positions': 0, 'total_trades': 0
            }
    
    def get_performance_metrics(self) -> Dict:
        """Get detailed performance metrics"""
        try:
            portfolio = self.get_portfolio_summary()
            
            # Calculate win rate (simplified)
            winning_trades = sum(1 for pos in self.positions.values() 
                               if pos['realized_pnl'] > 0)
            total_closed_trades = sum(1 for pos in self.positions.values() 
                                    if pos['realized_pnl'] != 0)
            
            win_rate = (winning_trades / max(total_closed_trades, 1)) * 100
            
            return {
                'total_pnl': portfolio['total_pnl'],
                'total_trades': self.total_trades,
                'win_rate': win_rate,
                'portfolio_value': self.portfolio_value,
                'active_strategies': len(self.active_strategies),
                'grid_profits': portfolio['grid_profits'],
                'realized_pnl': portfolio['realized_pnl'],
                'unrealized_pnl': portfolio['unrealized_pnl']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating performance metrics: {e}")
            return {
                'total_pnl': 0, 'total_trades': 0, 'win_rate': 0,
                'portfolio_value': 0, 'active_strategies': 0
            }
    
    async def close_all_positions(self):
        """Close all open positions"""
        self.logger.info("🚨 Closing all positions...")
        
        try:
            for symbol, position in self.positions.items():
                if position['quantity'] > 0:
                    current_price = self.data_collector.get_latest_price(symbol)
                    if current_price:
                        # Close position
                        await self.trade_executor.execute_trade(
                            symbol, 'sell', position['quantity'], current_price
                        )
                        
                        # Update risk manager
                        self.risk_manager.close_position(symbol, current_price)
            
            # Stop all grids
            for symbol in list(self.grid_strategy.active_grids.keys()):
                await self.grid_strategy.stop_grid(symbol)
        
            self.logger.info("✅ All positions closed")
            
        except Exception as e:
            self.logger.error(f"❌ Error closing positions: {e}")
    
    def generate_final_report(self) -> List[str]:
        """Generate final trading report"""
        try:
            portfolio = self.get_portfolio_summary()
            performance = self.get_performance_metrics()
            
            report = [
                f"Total Trades: {self.total_trades}",
                f"Final Portfolio Value: ${portfolio['total_value']:,.2f}",
                f"Total P&L: ${portfolio['total_pnl']:,.2f} ({portfolio['total_pnl_percent']:+.2f}%)",
                f"Realized P&L: ${portfolio['realized_pnl']:,.2f}",
                f"Unrealized P&L: ${portfolio['unrealized_pnl']:,.2f}",
                f"Grid Profits: ${portfolio['grid_profits']:,.2f}",
                f"Win Rate: {performance['win_rate']:.1f}%",
                f"Active Positions: {portfolio['active_positions']}",
                f"Active Strategies: {performance['active_strategies']}"
            ]
            
            # Add grid summary
            grid_summary = self.grid_strategy.get_all_grids_summary()
            if grid_summary['total_grids'] > 0:
                report.extend([
                    f"Grid Summary:",
                    f"  Total Grids: {grid_summary['total_grids']}",
                    f"  Grid Efficiency: {grid_summary.get('overall_efficiency', 0):.2f}%"
                ])
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Error generating final report: {e}")
            return [f"Error generating report: {e}"]