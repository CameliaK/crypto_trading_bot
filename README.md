# Crypto AI Trading Bot

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
