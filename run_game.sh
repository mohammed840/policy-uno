#!/bin/bash
# Run the UNO TUI game
cd "$(dirname "$0")"
exec python3 -c "from tui.app import main; main()"
