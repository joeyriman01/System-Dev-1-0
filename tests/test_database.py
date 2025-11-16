<<<<<<< HEAD
#!/usr/bin/env python3
"""Test database module."""

from database.models import init_database, Trade, Signal
from datetime import datetime

if __name__ == "__main__":
    print("Testing database...")
    
    # Initialize database
    db_manager = init_database("sqlite:///data_storage/nullspectre.db")
    print("✓ Database created")
    
    # Test session
    session = db_manager.get_session()
    print("✓ Session created")
    
    # Create test trade
    test_trade = Trade(
        trade_id="TEST_001",
        symbol="ETHUSDT",
        entry_time=datetime.utcnow(),
        entry_price=4901.50,
        entry_size=100.0,
        leverage=61,
        side="SHORT",
        conviction_score=85,
        initial_stop=4976.00,
        status="OPEN"
    )
    
    session.add(test_trade)
    session.commit()
    print("✓ Test trade created")
    
    # Query it back
    trades = session.query(Trade).all()
    print(f"✓ Found {len(trades)} trade(s)")
    
    # Clean up
    session.query(Trade).delete()
    session.commit()
    session.close()
    print("✓ Test data cleaned up")
    
    print("\n✓ Database test complete!")
=======
#!/usr/bin/env python3
"""Test database module."""

from database.models import init_database, Trade, Signal
from datetime import datetime

if __name__ == "__main__":
    print("Testing database...")
    
    # Initialize database
    db_manager = init_database("sqlite:///data_storage/nullspectre.db")
    print("✓ Database created")
    
    # Test session
    session = db_manager.get_session()
    print("✓ Session created")
    
    # Create test trade
    test_trade = Trade(
        trade_id="TEST_001",
        symbol="ETHUSDT",
        entry_time=datetime.utcnow(),
        entry_price=4901.50,
        entry_size=100.0,
        leverage=61,
        side="SHORT",
        conviction_score=85,
        initial_stop=4976.00,
        status="OPEN"
    )
    
    session.add(test_trade)
    session.commit()
    print("✓ Test trade created")
    
    # Query it back
    trades = session.query(Trade).all()
    print(f"✓ Found {len(trades)} trade(s)")
    
    # Clean up
    session.query(Trade).delete()
    session.commit()
    session.close()
    print("✓ Test data cleaned up")
    
    print("\n✓ Database test complete!")
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
    print("✓ Database file: data_storage/nullspectre.db")