<<<<<<< HEAD
#!/usr/bin/env python3
"""Test configuration module."""

from config.settings import settings, validate_settings, print_settings_summary

if __name__ == "__main__":
    print("Testing configuration...")
    
    if validate_settings():
        print("✓ Configuration valid!")
        print_settings_summary()
    else:
=======
#!/usr/bin/env python3
"""Test configuration module."""

from config.settings import settings, validate_settings, print_settings_summary

if __name__ == "__main__":
    print("Testing configuration...")
    
    if validate_settings():
        print("✓ Configuration valid!")
        print_settings_summary()
    else:
>>>>>>> 9af756b73b89b9b4c38d9e9973d524d2cefc95bb
        print("✗ Configuration has errors.")