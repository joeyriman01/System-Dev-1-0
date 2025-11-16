import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


class LoggingConfig:
    """Configure logging for the entire system."""

    def __init__(
        self, log_level: str = "INFO", log_file: str = "data_storage/logs/trading.log"
    ):
        self.log_level = log_level.upper()
        self.log_file = Path(log_file)
        self._setup_logging()

    def _setup_logging(self):
        """Set up loguru logger with file and console handlers."""

        # Remove default handler
        logger.remove()

        # Create log directory if it doesn't exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Console handler (colored output)
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=self.log_level,
            colorize=True,
        )

        # File handler (detailed logs)
        logger.add(
            self.log_file,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            level="DEBUG",  # Always log DEBUG to file
            rotation="100 MB",  # Rotate when file reaches 100MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress rotated logs
            backtrace=True,  # Include traceback
            diagnose=True,  # Include variable values
        )

        # Separate file for errors only
        error_log = self.log_file.parent / "errors.log"
        logger.add(
            error_log,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            level="ERROR",
            rotation="50 MB",
            retention="60 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

        # Separate file for trades only
        trade_log = self.log_file.parent / "trades.log"
        logger.add(
            trade_log,
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            filter=lambda record: "TRADE" in record["extra"],
            rotation="50 MB",
            retention="1 year",  # Keep trade logs for 1 year
            compression="zip",
        )

        logger.info(f"Logging initialized - Level: {self.log_level}")
        logger.info(f"Main log: {self.log_file}")
        logger.info(f"Error log: {error_log}")
        logger.info(f"Trade log: {trade_log}")

    @staticmethod
    def log_trade(trade_type: str, symbol: str, **kwargs):
        """Log trade-specific information."""
        logger.bind(TRADE=True).info(f"[TRADE] {trade_type} | {symbol} | {kwargs}")

    @staticmethod
    def log_signal(signal_type: str, symbol: str, score: int, **kwargs):
        """Log signal detection."""
        logger.info(
            f"[SIGNAL] {signal_type} | {symbol} | Score: {score}/100 | {kwargs}"
        )

    @staticmethod
    def log_entry(
        symbol: str, side: str, price: float, size: float, leverage: int, **kwargs
    ):
        """Log trade entry."""
        LoggingConfig.log_trade(
            "ENTRY",
            symbol,
            side=side,
            price=price,
            size=size,
            leverage=leverage,
            **kwargs,
        )

    @staticmethod
    def log_exit(
        symbol: str, exit_type: str, price: float, pnl: float, pnl_pct: float, **kwargs
    ):
        """Log trade exit (partial, stop, or runner)."""
        LoggingConfig.log_trade(
            f"EXIT-{exit_type}",
            symbol,
            price=price,
            pnl=f"${pnl:.2f}",
            pnl_pct=f"{pnl_pct:.2f}%",
            **kwargs,
        )

    @staticmethod
    def log_api_call(
        exchange: str,
        endpoint: str,
        success: bool,
        response_time: Optional[float] = None,
    ):
        """Log API calls for debugging rate limits and errors."""
        status = "SUCCESS" if success else "FAILED"
        msg = f"[API] {exchange} | {endpoint} | {status}"
        if response_time:
            msg += f" | {response_time:.3f}s"

        if success:
            logger.debug(msg)
        else:
            logger.warning(msg)

    @staticmethod
    def log_error(error_type: str, message: str, **kwargs):
        """Log errors with full context."""
        logger.error(f"[ERROR] {error_type} | {message} | {kwargs}")

    @staticmethod
    def log_position_update(
        symbol: str, unrealized_pnl: float, pnl_pct: float, **kwargs
    ):
        """Log position updates (for monitoring)."""
        logger.debug(
            f"[POSITION] {symbol} | "
            f"Unrealized P&L: ${unrealized_pnl:.2f} ({pnl_pct:.2f}%) | "
            f"{kwargs}"
        )

    @staticmethod
    def log_system_event(event_type: str, message: str, **kwargs):
        """Log system events (startup, shutdown, mode changes, etc.)."""
        logger.info(f"[SYSTEM] {event_type} | {message} | {kwargs}")

    @staticmethod
    def log_risk_event(risk_type: str, message: str, **kwargs):
        """Log risk management events (limits hit, stops triggered, etc.)."""
        logger.warning(f"[RISK] {risk_type} | {message} | {kwargs}")


# Global logger instance
def setup_logging(
    log_level: str = "INFO", log_file: str = "data_storage/logs/trading.log"
):
    """Initialize logging system."""
    return LoggingConfig(log_level, log_file)


# Convenience functions for logging
def log_startup(mode: str, capital: float, testnet: bool):
    """Log system startup."""
    logger.info("=" * 60)
    logger.info("NULLSPECTRE V2.0 - SYSTEM STARTING")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Capital: ${capital:.2f}")
    logger.info(f"Testnet: {testnet}")
    logger.info("=" * 60)


def log_shutdown(reason: str = "Normal shutdown"):
    """Log system shutdown."""
    logger.info("=" * 60)
    logger.info(f"NULLSPECTRE V2.0 - SHUTTING DOWN: {reason}")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Test logging
    setup_logging("DEBUG")

    log_startup("PAPER TRADING", 100.0, True)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test specialized logging
    LoggingConfig.log_signal("REJECTION", "ETHUSDT", 85, rejections=3, cvd=-5000000)
    LoggingConfig.log_entry("ETHUSDT", "SHORT", 4901.50, 100.0, 61)
    LoggingConfig.log_exit("ETHUSDT", "PARTIAL-1", 4876.25, 25.0, 50.0)
    LoggingConfig.log_api_call("Binance", "/fapi/v1/ticker/24hr", True, 0.123)

    log_shutdown("Testing complete")

    print("\n✓ Logging test complete. Check data_storage/logs/ for log files.")
