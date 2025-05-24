# main.py - Main entry point for our crypto trading bot
import asyncio
import logging
from datetime import datetime

# Import our organized modules
from config.settings import Config
from utils.logger import setup_logger
from data.collectors import RealTimeCollector
from trading.strategies import StrategyManager
from trading.risk_manager import RiskManager
from trading.executor import TradeExecutor

class CryptoTradingBot:
    """
    Main trading bot that coordinates all components
    """
    
    def __init__(self):
        # Initialize configuration and logging
        self.config = Config()
        self.logger = setup_logger()
        
        # Validate configuration
        if not self.config.validate_config():
            self.logger.error("❌ Configuration validation failed!")
            return
        
        # Initialize all components
        self.data_collector = RealTimeCollector(self.config)
        self.risk_manager = RiskManager(self.config)
        self.trade_executor = TradeExecutor(self.config, self.data_collector.exchange)
        self.strategy_manager = StrategyManager(
            self.config, 
            self.data_collector, 
            self.risk_manager,
            self.trade_executor
        )
        
        # Bot state
        self.is_running = False
        self.start_time = None
        
    async def start(self):
        """Start the complete trading bot system"""
        self.start_time = datetime.now()
        self.logger.info("🚀 Starting Crypto AI Trading Bot v3.0")
        self.logger.info("=" * 60)
        
        # Display configuration summary
        config_summary = self.config.get_summary()
        for item in config_summary:
            self.logger.info(f"🔧 {item}")
        
        self.logger.info("=" * 60)
        
        self.is_running = True
        
        # Start all components concurrently
        tasks = [
            # Data collection
            self.data_collector.collect_prices(),
            self.data_collector.collect_historical_data(),
            self.data_collector.collect_order_books(),
            self.data_collector.collect_sentiment(),
            
            # Strategy execution
            self.strategy_manager.run_strategy_loop(),
            
            # Monitoring and display
            self.display_live_status(),
            self.monitor_performance(),
            
            # Risk management
            self.risk_manager.monitor_risk()
        ]
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"💥 Critical error in main loop: {e}")
            await self.shutdown()
    
    async def display_live_status(self):
        """Display live trading status"""
        while self.is_running:
            try:
                await asyncio.sleep(15)  # Update every 15 seconds
                
                if not self.data_collector.price_data:
                    continue
                
                print(f"\n🔴 LIVE STATUS - {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 90)
                
                # Display prices and signals
                for symbol in self.config.SYMBOLS:
                    if symbol in self.data_collector.price_data:
                        price_data = self.data_collector.price_data[symbol]
                        price = price_data['price']
                        change = price_data['change']
                        
                        # Get latest analysis
                        analysis = self.strategy_manager.get_latest_analysis(symbol)
                        signal = analysis.get('signal', 'HOLD')
                        rsi = analysis.get('rsi', 50)
                        
                        # Get strategy status
                        strategy_status = self.strategy_manager.get_strategy_status(symbol)
                        
                        # Color coding
                        color = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                        signal_color = "🟢" if signal == 'BUY' else "🔴" if signal == 'SELL' else "⚪"
                        
                        print(f"{color} {symbol:12} ${price:>8,.2f} ({change:>+5.1f}%) | "
                              f"{signal_color} {signal:>4} | RSI: {rsi:>5.1f} | {strategy_status}")
                
                # Display portfolio summary
                portfolio = self.strategy_manager.get_portfolio_summary()
                print("-" * 90)
                print(f"💰 Portfolio: ${portfolio['total_value']:>8,.2f} | "
                      f"P&L: ${portfolio['total_pnl']:>8.2f} ({portfolio['total_pnl_percent']:>+5.1f}%) | "
                      f"Active: {portfolio['active_positions']} | "
                      f"Trades: {portfolio['total_trades']}")
                
                print("=" * 90)
                
            except Exception as e:
                self.logger.error(f"❌ Display error: {e}")
                await asyncio.sleep(5)
    
    async def monitor_performance(self):
        """Monitor overall bot performance"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Get performance metrics
                performance = self.strategy_manager.get_performance_metrics()
                uptime = datetime.now() - self.start_time
                
                self.logger.info(f"📊 Performance Update:")
                self.logger.info(f"   ⏱️  Uptime: {uptime}")
                self.logger.info(f"   💰 Total P&L: ${performance.get('total_pnl', 0):.2f}")
                self.logger.info(f"   📈 Win Rate: {performance.get('win_rate', 0):.1f}%")
                self.logger.info(f"   🎯 Total Trades: {performance.get('total_trades', 0)}")
                
            except Exception as e:
                self.logger.error(f"❌ Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("🛑 Initiating graceful shutdown...")
        self.is_running = False
        
        try:
            # Close all positions if needed
            await self.strategy_manager.close_all_positions()
            
            # Save final report
            final_report = self.strategy_manager.generate_final_report()
            self.logger.info("📋 Final Trading Report:")
            for line in final_report:
                self.logger.info(f"   {line}")
                
        except Exception as e:
            self.logger.error(f"❌ Shutdown error: {e}")
        
        self.logger.info("✅ Shutdown complete")
    
    def stop(self):
        """Stop the bot (called from signal handlers)"""
        asyncio.create_task(self.shutdown())

async def main():
    """Main function"""
    bot = CryptoTradingBot()
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\n👋 Received shutdown signal...")
        await bot.shutdown()
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        if 'bot' in locals():
            await bot.shutdown()

if __name__ == "__main__":
    print("🤖 Crypto AI Trading Bot - Professional Architecture")
    print("🔧 Initializing all components...")
    asyncio.run(main())