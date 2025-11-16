"""
CONFIGURATION SETTINGS
======================
Central configuration for NullSpectre v2.0

Loads from environment variables with fallback defaults.

Author: Grim (Institutional Standards)
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class Settings:
    """Application settings loaded from environment variables"""

    # =====================================================================
    # BINANCE API CONFIGURATION
    # =====================================================================

    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "True").lower() == "true"

    # =====================================================================
    # DATABASE CONFIGURATION
    # =====================================================================

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///nullspectre.db"  # Default to SQLite
    )

    # =====================================================================
    # LOGGING CONFIGURATION
    # =====================================================================

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", "./logs"))
    LOG_MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", 10 * 1024 * 1024))  # 10MB
    LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", 5))

    # =====================================================================
    # TRADING CONFIGURATION
    # =====================================================================

    # Risk management
    MAX_POSITION_SIZE_USD: float = float(os.getenv("MAX_POSITION_SIZE_USD", 1000.0))
    MAX_LEVERAGE: int = int(os.getenv("MAX_LEVERAGE", 100))
    MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", 5.0))
    MAX_CONSECUTIVE_LOSSES: int = int(os.getenv("MAX_CONSECUTIVE_LOSSES", 3))

    # Position sizing
    DEFAULT_LEVERAGE: int = int(os.getenv("DEFAULT_LEVERAGE", 50))
    BASE_POSITION_SIZE_USD: float = float(os.getenv("BASE_POSITION_SIZE_USD", 100.0))

    # Profit targets (multipliers)
    PROFIT_TARGET_1: float = float(os.getenv("PROFIT_TARGET_1", 0.50))  # 50%
    PROFIT_TARGET_2: float = float(os.getenv("PROFIT_TARGET_2", 1.00))  # 100%
    PROFIT_TARGET_3: float = float(os.getenv("PROFIT_TARGET_3", 2.00))  # 200%

    # Exit percentages at each target
    EXIT_PCT_TARGET_1: float = float(os.getenv("EXIT_PCT_TARGET_1", 0.25))  # 25%
    EXIT_PCT_TARGET_2: float = float(os.getenv("EXIT_PCT_TARGET_2", 0.25))  # 25%
    EXIT_PCT_TARGET_3: float = float(os.getenv("EXIT_PCT_TARGET_3", 0.25))  # 25%
    # Remaining 25% is the "runner"

    # =====================================================================
    # STRATEGY CONFIGURATION
    # =====================================================================

    # New listing scanner
    MIN_LISTING_VOLUME_24H: float = float(
        os.getenv("MIN_LISTING_VOLUME_24H", 100000)
    )  # $100k
    MIN_LISTING_PRICE_CHANGE: float = float(
        os.getenv("MIN_LISTING_PRICE_CHANGE", 10.0)
    )  # 10%
    MAX_LISTING_SPREAD_BPS: float = float(
        os.getenv("MAX_LISTING_SPREAD_BPS", 50)
    )  # 50 bps

    # Rejection pattern detector
    MIN_REJECTIONS_REQUIRED: int = int(os.getenv("MIN_REJECTIONS_REQUIRED", 3))
    REJECTION_TIMEFRAME: str = os.getenv("REJECTION_TIMEFRAME", "1h")
    REJECTION_PROXIMITY_PCT: float = float(
        os.getenv("REJECTION_PROXIMITY_PCT", 0.5)
    )  # 0.5%

    # CVD divergence
    CVD_LOOKBACK_CANDLES: int = int(os.getenv("CVD_LOOKBACK_CANDLES", 20))
    CVD_DIVERGENCE_THRESHOLD: float = float(
        os.getenv("CVD_DIVERGENCE_THRESHOLD", 0.15)
    )  # 15%

    # Conviction scoring
    MIN_CONVICTION_SCORE: float = float(
        os.getenv("MIN_CONVICTION_SCORE", 60.0)
    )  # 60/100

    # =====================================================================
    # DATA COLLECTION CONFIGURATION
    # =====================================================================

    # Market data refresh rates (seconds)
    SCANNER_REFRESH_INTERVAL: int = int(
        os.getenv("SCANNER_REFRESH_INTERVAL", 300)
    )  # 5 min
    ORDERBOOK_REFRESH_INTERVAL: int = int(
        os.getenv("ORDERBOOK_REFRESH_INTERVAL", 5)
    )  # 5 sec
    FUNDING_REFRESH_INTERVAL: int = int(
        os.getenv("FUNDING_REFRESH_INTERVAL", 60)
    )  # 1 min

    # =====================================================================
    # SYSTEM CONFIGURATION
    # =====================================================================

    # Environment
    ENV: str = os.getenv("ENV", "development")  # development, staging, production

    # Paper trading mode
    PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "True").lower() == "true"

    # Feature flags
    ENABLE_TELEGRAM_NOTIFICATIONS: bool = (
        os.getenv("ENABLE_TELEGRAM_NOTIFICATIONS", "False").lower() == "true"
    )
    ENABLE_DISCORD_NOTIFICATIONS: bool = (
        os.getenv("ENABLE_DISCORD_NOTIFICATIONS", "False").lower() == "true"
    )

    # Notification tokens (if enabled)
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    DISCORD_WEBHOOK_URL: Optional[str] = os.getenv("DISCORD_WEBHOOK_URL")

    def __init__(self):
        """Validate settings on initialization"""
        self._validate_api_keys()
        self._validate_risk_parameters()
        self._create_directories()

    def _validate_api_keys(self):
        """Validate API credentials are present"""
        if not self.PAPER_TRADING and not self.BINANCE_TESTNET:
            # Only require keys for live trading
            if not self.BINANCE_API_KEY or not self.BINANCE_API_SECRET:
                raise ValueError(
                    "Binance API credentials required for live trading. "
                    "Set BINANCE_API_KEY and BINANCE_API_SECRET in .env file"
                )

    def _validate_risk_parameters(self):
        """Validate risk management parameters are sane"""
        if self.MAX_LEVERAGE > 125:
            raise ValueError("MAX_LEVERAGE cannot exceed 125 (Binance limit)")

        if self.MAX_DAILY_LOSS_PCT > 100:
            raise ValueError("MAX_DAILY_LOSS_PCT cannot exceed 100%")

        if self.MAX_POSITION_SIZE_USD <= 0:
            raise ValueError("MAX_POSITION_SIZE_USD must be positive")

        # Validate exit percentages sum to <= 100%
        total_exit_pct = (
            self.EXIT_PCT_TARGET_1 + self.EXIT_PCT_TARGET_2 + self.EXIT_PCT_TARGET_3
        )
        if total_exit_pct > 1.0:
            raise ValueError(
                f"Exit percentages sum to {total_exit_pct*100}%, must be <= 100%"
            )

    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def is_production(self) -> bool:
        """Check if running in production mode"""
        return (
            self.ENV == "production"
            and not self.PAPER_TRADING
            and not self.BINANCE_TESTNET
        )

    def is_testnet(self) -> bool:
        """Check if using testnet"""
        return self.BINANCE_TESTNET

    def __repr__(self) -> str:
        """String representation hiding sensitive data"""
        return (
            f"Settings("
            f"ENV={self.ENV}, "
            f"TESTNET={self.BINANCE_TESTNET}, "
            f"PAPER_TRADING={self.PAPER_TRADING}, "
            f"LOG_LEVEL={self.LOG_LEVEL}"
            f")"
        )


# Create global settings instance
settings = Settings()


if __name__ == "__main__":
    """Test settings configuration"""
    print("=" * 70)
    print("SETTINGS CONFIGURATION")
    print("=" * 70)

    print(f"\nEnvironment: {settings.ENV}")
    print(f"Testnet: {settings.BINANCE_TESTNET}")
    print(f"Paper Trading: {settings.PAPER_TRADING}")
    print(f"Production Mode: {settings.is_production()}")

    print(f"\nAPI Configuration:")
    print(
        f"  API Key: {'*' * 8 + settings.BINANCE_API_KEY[-4:] if settings.BINANCE_API_KEY else 'NOT SET'}"
    )
    print(
        f"  API Secret: {'*' * 8 + settings.BINANCE_API_SECRET[-4:] if settings.BINANCE_API_SECRET else 'NOT SET'}"
    )

    print(f"\nRisk Management:")
    print(f"  Max Position Size: ${settings.MAX_POSITION_SIZE_USD:,.2f}")
    print(f"  Max Leverage: {settings.MAX_LEVERAGE}X")
    print(f"  Max Daily Loss: {settings.MAX_DAILY_LOSS_PCT}%")
    print(f"  Max Consecutive Losses: {settings.MAX_CONSECUTIVE_LOSSES}")

    print(f"\nProfit Targets:")
    print(
        f"  Target 1: {settings.PROFIT_TARGET_1*100:.0f}% (exit {settings.EXIT_PCT_TARGET_1*100:.0f}%)"
    )
    print(
        f"  Target 2: {settings.PROFIT_TARGET_2*100:.0f}% (exit {settings.EXIT_PCT_TARGET_2*100:.0f}%)"
    )
    print(
        f"  Target 3: {settings.PROFIT_TARGET_3*100:.0f}% (exit {settings.EXIT_PCT_TARGET_3*100:.0f}%)"
    )

    print(f"\nStrategy Configuration:")
    print(f"  Min Listing Volume: ${settings.MIN_LISTING_VOLUME_24H:,.0f}")
    print(f"  Min Price Change: {settings.MIN_LISTING_PRICE_CHANGE}%")
    print(f"  Min Rejections: {settings.MIN_REJECTIONS_REQUIRED}")
    print(f"  Min Conviction Score: {settings.MIN_CONVICTION_SCORE}/100")

    print(f"\nLogging:")
    print(f"  Level: {settings.LOG_LEVEL}")
    print(f"  Directory: {settings.LOG_DIR}")

    print("\n" + "=" * 70)
