#!/usr/bin/env python3
# setup_bot.py - Automated setup script for the crypto trading bot

import os
import sys

def create_directory_structure():
    """Create the complete directory structure"""
    directories = [
        'config',
        'data', 
        'trading',
        'technical',
        'utils',
        'ml',
        'tests',
        'logs'  # For log files
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create __init__.py files
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f'"""{directory.title()} module for crypto trading bot"""\n')
    
    print("✅ Directory structure created!")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = [
        "ccxt>=4.0.0",
        "pandas>=1.5.0", 
        "numpy>=1.21.0",
        "asyncio",
        "python-dotenv>=0.19.0",
        "aiohttp>=3.8.0"
    ]
    
    print("📝 Creating requirements.txt...")
    with open('requirements.txt', 'w') as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print("✅ requirements.txt created!")

def create_env_template():
    """Create .env template file"""
    env_content = """# Crypto Trading Bot Environment Variables

# Binance API Credentials (optional for paper trading)
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here

# Bot Configuration
SANDBOX_MODE=True
INITIAL_BALANCE=10000.0
LOG_LEVEL=INFO

# Risk Management
MAX_DAILY_LOSS=0.05
MAX_POSITION_SIZE=0.1

# Grid Trading
GRID_SIZE=8
GRID_SPACING_PERCENT=0.5
BASE_ORDER_SIZE=100
"""
    
    print("🔧 Creating .env template...")
    with open('.env.template', 'w') as f:
        f.write(env_content)
    
    print("✅ .env template created!")

def create_readme():
    """Create README.md file"""
    readme_content = """# Crypto AI Trading Bot

An advanced cryptocurrency trading bot with multiple strategies:

## Features
- 🔥 **Grid Trading**: Automated buy/sell orders in price ranges
- 📊 **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages  
- 🎯 **Smart Strategy Selection**: Automatically picks best strategy for market conditions
- 🛡️ **Risk Management**: Stop losses, position sizing, daily loss limits
- 📱 **Paper Trading**: Test strategies without real money
- 📈 **Real-time Data**: Live prices from Binance exchange

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the bot:**
```bash
python main.py
```

3. **Watch it trade:**
- See live crypto prices updating
- Observe technical analysis signals  
- Watch grid orders being placed and filled
- Monitor profit/loss in real-time

## Configuration

Edit `config/settings.py` to customize:
- Trading pairs to monitor
- Risk management settings
- Strategy parameters
- Grid trading configuration

## Strategies

### Grid Trading
- Places buy orders below current price
- Places sell orders above current price  
- Profits from price oscillations
- Works best in sideways markets

### Technical Analysis
- Uses RSI, MACD, Bollinger Bands
- Generates BUY/SELL signals
- Combines multiple indicators
- Works best in trending markets

### Risk Management
- Maximum 5% daily loss limit
- Position sizing based on account balance
- Stop losses on all trades
- Automatic strategy switching

## Safety

- **Paper Trading Mode**: Enabled by default (no real money)
- **Simulated Orders**: All trades are simulated initially
- **Comprehensive Logging**: All actions logged to files
- **Risk Controls**: Multiple safety mechanisms

## Next Steps

1. **Test Thoroughly**: Run in paper trading mode first
2. **Add Real API Keys**: For live data (still paper trading)
3. **Enable Live Trading**: Only after extensive testing
4. **Add More Strategies**: AI prediction, sentiment analysis

## Support

This is educational software. Use at your own risk.
Always test thoroughly before risking real money.
"""
    
    print("📖 Creating README.md...")
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("✅ README.md created!")

def main():
    """Main setup function"""
    print("🤖 Crypto Trading Bot Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.basename(os.getcwd()).endswith('trading_bot'):
        print("⚠️  Recommendation: Run this in a directory named 'crypto_trading_bot'")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("👋 Setup cancelled")
            return
    
    # Create structure
    create_directory_structure()
    create_requirements_file() 
    create_env_template()
    create_readme()
    
    print("\n" + "=" * 40)
    print("🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Copy the code files from Claude to their respective locations")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the bot: python main.py")
    print("\n🚀 Happy trading!")

if __name__ == "__main__":
    main()