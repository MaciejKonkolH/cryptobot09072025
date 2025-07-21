import pathlib
import sys
import logging

# Configure basic logging to see Freqtrade's output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("--- Freqtrade File Access Spy v2.0 ---")

# Store the original functions we want to spy on
original_glob = pathlib.Path.glob
original_exists = pathlib.Path.exists

# --- Spy Functions ---
def spy_glob(self, pattern):
    """Spy on the glob function to see search patterns."""
    print(f"\n🕵️‍♂️ SPY: Freqtrade is SEARCHING in directory '{self}' with pattern '{pattern}'\n")
    return original_glob(self, pattern)

def spy_exists(self):
    """Spy on the exists function to see exact file checks."""
    # We only log this if it's a file check to avoid spam from directory checks
    if self.is_file():
        print(f"🕵️‍♂️ SPY: Freqtrade is CHECKING FOR file: '{self}'")
    return original_exists(self)

# --- Activate the Spies (Monkey-Patching) ---
pathlib.Path.glob = spy_glob
pathlib.Path.exists = spy_exists
print("--- Spy activated. Preparing to run Freqtrade... ---\n")

# --- Run Freqtrade ---
try:
    # We must import main AFTER patching
    from freqtrade.__main__ import main

    # Prepare sys.argv for Freqtrade with the exact backtest command
    sys.argv = [
        'freqtrade',  # The script name, as expected by argparse
        'backtesting',
        '--config', 'user_data/config.json',
        '--strategy', 'Enhanced_ML_MA43200_Buffer_Strategy',
        '--timerange', '20240101-20240102',
        '--dry-run-wallet', '1000',
        '--datadir', 'user_data/strategies/inputs'  # <-- DODANE: wskazuje gdzie szukać danych
    ]

    print(f"--- Executing Freqtrade with command: {' '.join(sys.argv)} ---\n")
    # Run Freqtrade's main entry point
    main()

except SystemExit as e:
    # Freqtrade often exits with SystemExit, which is normal.
    print(f"\n--- Freqtrade exited cleanly with code: {e.code} ---")
except ImportError as e:
    print(f"\n--- CRITICAL ERROR ---")
    print(f"An import error occurred: {e}")
    print("This might mean the script needs to be run from a different directory")
    print("or the environment is not activated correctly.")
except Exception as e:
    print(f"\n--- Freqtrade crashed with an unhandled exception: {e} ---")

print("\n--- Spy script finished. ---") 