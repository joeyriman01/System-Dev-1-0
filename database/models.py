"""
NullSpectre v2.0 - Database Models
===================================
SQLAlchemy ORM models for storing trades, signals, positions, and performance data.
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Trade(Base):
    """Complete trade record from entry to final exit."""

    __tablename__ = "trades"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Trade identification
    trade_id = Column(String(50), unique=True, nullable=False)  # UUID
    symbol = Column(String(20), nullable=False)

    # Entry details
    entry_time = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_size = Column(Float, nullable=False)  # Position size in USDT
    leverage = Column(Integer, nullable=False)
    side = Column(String(10), nullable=False)  # "SHORT" or "LONG"

    # Signal information at entry
    conviction_score = Column(Integer, nullable=False)  # 0-100
    signal_breakdown = Column(JSON)  # {"rejections": 30, "cvd": 25, ...}
    rejection_count = Column(Integer)
    cvd_divergence = Column(Float)
    funding_rate = Column(Float)
    orderbook_imbalance = Column(Float)

    # Stop management
    initial_stop = Column(Float, nullable=False)
    current_stop = Column(Float)
    stop_moved_to_be = Column(Boolean, default=False)
    stop_move_time = Column(DateTime)

    # Partial exits (stored as JSON array)
    partials = Column(
        JSON
    )  # [{"type": "PARTIAL-1", "price": 4876.25, "size": 25, ...}, ...]

    # Runner status
    runner_active = Column(Boolean, default=False)
    runner_size = Column(Float)
    runner_entry_price = Column(Float)

    # Exit details
    exit_time = Column(DateTime)
    exit_type = Column(String(20))  # "STOP", "RUNNER", "MANUAL", "EMERGENCY"
    exit_price = Column(Float)

    # Performance metrics
    gross_pnl = Column(Float)  # Before fees
    net_pnl = Column(Float)  # After fees
    pnl_pct = Column(Float)  # Percentage return
    fees = Column(Float)
    max_favorable_excursion = Column(Float)  # Best unrealized profit
    max_adverse_excursion = Column(Float)  # Worst unrealized loss

    # Duration
    duration_seconds = Column(Integer)

    # Notes and metadata
    notes = Column(Text)
    screenshot_path = Column(String(200))
    chart_link = Column(String(200))

    # Status
    status = Column(String(20))  # "OPEN", "CLOSED", "STOPPED"

    # Relationships
    position_updates = relationship(
        "PositionUpdate", back_populates="trade", cascade="all, delete-orphan"
    )

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return (
            f"<Trade(trade_id='{self.trade_id}', symbol='{self.symbol}', "
            f"entry={self.entry_price}, pnl={self.net_pnl})>"
        )


class PositionUpdate(Base):
    """Real-time position updates (P&L tracking)."""

    __tablename__ = "position_updates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), ForeignKey("trades.trade_id"), nullable=False)

    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    current_price = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    pnl_pct = Column(Float, nullable=False)

    # Relationship
    trade = relationship("Trade", back_populates="position_updates")

    def __repr__(self):
        return (
            f"<PositionUpdate(trade_id='{self.trade_id}', pnl={self.unrealized_pnl})>"
        )


class Signal(Base):
    """Detected trading signals (whether traded or not)."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(50), unique=True, nullable=False)

    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    symbol = Column(String(20), nullable=False)

    # Signal components
    conviction_score = Column(Integer, nullable=False)
    conviction_tier = Column(String(10))  # "MAX", "HIGH", "MEDIUM", "SKIP"

    rejection_count = Column(Integer)
    rejection_score = Column(Integer)

    cvd_divergence = Column(Float)
    cvd_score = Column(Integer)

    funding_rate = Column(Float)
    funding_score = Column(Integer)

    orderbook_imbalance = Column(Float)
    orderbook_score = Column(Integer)

    footprint_pressure = Column(String(20))  # "STRONG_SELL", "MODERATE_SELL", etc.
    footprint_score = Column(Integer)

    # Market context
    price_at_signal = Column(Float)
    resistance_level = Column(Float)
    volume_24h = Column(Float)

    # Action taken
    traded = Column(Boolean, default=False)
    trade_id = Column(String(50))  # Link to Trade if executed
    skip_reason = Column(String(100))  # Why signal was skipped (if applicable)

    # Metadata
    notes = Column(Text)

    def __repr__(self):
        return (
            f"<Signal(symbol='{self.symbol}', score={self.conviction_score}, "
            f"traded={self.traded})>"
        )


class DailyPerformance(Base):
    """Daily performance summary."""

    __tablename__ = "daily_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, unique=True)

    # Trade statistics
    trades_total = Column(Integer, default=0)
    trades_won = Column(Integer, default=0)
    trades_lost = Column(Integer, default=0)
    trades_stopped = Column(Integer, default=0)

    # Performance metrics
    gross_pnl = Column(Float, default=0.0)
    net_pnl = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    fees_paid = Column(Float, default=0.0)

    # Win rate
    win_rate = Column(Float)  # Percentage

    # Average trade metrics
    avg_win = Column(Float)
    avg_loss = Column(Float)
    avg_duration_seconds = Column(Integer)

    # Risk metrics
    largest_win = Column(Float)
    largest_loss = Column(Float)
    max_drawdown_pct = Column(Float)

    # Account metrics
    starting_balance = Column(Float)
    ending_balance = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return (
            f"<DailyPerformance(date='{self.date}', trades={self.trades_total}, "
            f"pnl={self.net_pnl})>"
        )


class SystemLog(Base):
    """System events and errors."""

    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    log_level = Column(String(10))  # "DEBUG", "INFO", "WARNING", "ERROR"
    log_type = Column(String(20))  # "SYSTEM", "API", "TRADE", "RISK", "ERROR"

    message = Column(Text, nullable=False)
    details = Column(JSON)  # Additional structured data

    def __repr__(self):
        return f"<SystemLog(level='{self.log_level}', type='{self.log_type}')>"


# Database session management
class DatabaseManager:
    """Manage database connections and sessions."""

    def __init__(self, database_url: str = "sqlite:///data_storage/nullspectre.db"):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(self.engine)

    def get_session(self):
        """Get a new database session."""
        return self.SessionLocal()


# Convenience functions
def init_database(database_url: str = "sqlite:///data_storage/nullspectre.db"):
    """Initialize database and create all tables."""
    db_manager = DatabaseManager(database_url)
    db_manager.create_tables()
    return db_manager


if __name__ == "__main__":
    # Test database creation
    print("Creating database and tables...")

    db_manager = init_database()
    print("✓ Database created successfully!")

    # Test session
    session = db_manager.get_session()
    print("✓ Session created successfully!")

    # Test creating a trade record
    test_trade = Trade(
        trade_id="TEST001",
        symbol="ETHUSDT",
        entry_time=datetime.utcnow(),
        entry_price=4901.50,
        entry_size=100.0,
        leverage=61,
        side="SHORT",
        conviction_score=85,
        initial_stop=4976.00,
        status="OPEN",
    )

    session.add(test_trade)
    session.commit()
    print("✓ Test trade record created!")

    # Query it back
    trades = session.query(Trade).all()
    print(f"✓ Found {len(trades)} trade(s) in database")

    # Clean up test data
    session.query(Trade).filter_by(trade_id="TEST001").delete()
    session.commit()
    session.close()

    print("\n✓ Database test complete!")
