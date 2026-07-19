#!/usr/bin/env python3
"""
Beta test harness — injects synthetic sensor events into the running daemon
to verify the full companion pipeline end-to-end.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_terminal_error():
    """Write a synthetic terminal error and watch for companion response."""
    exit_file = Path("/tmp/bubby_last_exit")
    exit_file.write_text("exit_code=1\nerror=ModuleNotFoundError: No module named 'flask'\n")
    print(f"[TEST] Wrote terminal error to {exit_file}")
    print("[TEST] Check logs for companion response within 60 seconds...")
    print("[TEST] Expected: 'Terminal error detected' → synthesis → critic → output")

def test_calendar_deadline():
    """Write a synthetic .ics file and watch for calendar alert."""
    ics_dir = Path.home() / "Documents"
    ics_dir.mkdir(exist_ok=True)
    from datetime import datetime, timedelta
    now = datetime.now()
    event_start = now + timedelta(minutes=30)
    event_end = event_start + timedelta(hours=1)
    
    ics_file = ics_dir / f"bubby_test_{now.strftime('%H%M')}.ics"
    ics_file.write_text(f"""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
DTSTART:{event_start.strftime('%Y%m%dT%H%M%S')}
DTEND:{event_end.strftime('%Y%m%dT%H%M%S')}
SUMMARY:Bubby Beta Test — Meeting Review
DESCRIPTION:Review beta test results with the team
LOCATION:Virtual
END:VEVENT
END:VCALENDAR
""")
    print(f"[TEST] Wrote calendar event to {ics_file}")
    print("[TEST] Event: 'Bubby Beta Test — Meeting Review' in 30 minutes")
    print("[TEST] Check logs for companion response within 120 seconds...")

def watch_logs():
    """Print relevant log lines."""
    log_file = Path(__file__).parent.parent / "logs" / "session_output.log"
    if not log_file.exists():
        log_file = Path(__file__).parent.parent / "logs" / "mission_control.log"
    if log_file.exists():
        print(f"\n[TEST] Last 10 lines from {log_file}:")
        lines = log_file.read_text().splitlines()[-10:]
        for line in lines:
            print(f"  {line}")
    else:
        print("\n[TEST] No log file found at logs/session_output.log")
        print("[TEST] Is the daemon running?")

if __name__ == "__main__":
    print("=" * 60)
    print("BUBBY BETA TEST HARNESS")
    print("=" * 60)
    print()
    
    if "--terminal" in sys.argv:
        test_terminal_error()
    elif "--calendar" in sys.argv:
        test_calendar_deadline()
    elif "--logs" in sys.argv:
        watch_logs()
    elif "--all" in sys.argv:
        test_terminal_error()
        time.sleep(2)
        test_calendar_deadline()
        time.sleep(2)
        watch_logs()
    else:
        print("Usage:")
        print("  python scripts/beta_test.py --terminal   # Simulate terminal error")
        print("  python scripts/beta_test.py --calendar   # Simulate calendar deadline")
        print("  python scripts/beta_test.py --all        # Run both + show logs")
        print("  python scripts/beta_test.py --logs       # Show recent log output")