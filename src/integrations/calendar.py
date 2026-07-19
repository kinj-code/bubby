"""Calendar Sensor — parses local .ics files for proactive deadline alerts.

Polls .ics (iCalendar) files in user's documents folder. When an event
is within 60 minutes, triggers a ProactivityContext with SYSTEM_ALERT urgency.

RAM: ~3MB (ics parsing + in-memory event cache).
"""

import logging
import time
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CalendarEvent:
    """A parsed calendar event."""
    title: str
    start: datetime
    end: Optional[datetime] = None
    location: str = ""
    description: str = ""
    source_file: str = ""
    minutes_until: float = float('inf')
    is_imminent: bool = False
    urgency: float = 0.0


class CalendarSensor:
    """Monitors local iCalendar files for upcoming deadlines."""

    IMMINENT_MINUTES = 60       # Events within this window are "imminent"
    WARNING_MINUTES = 120       # Events within this window get elevated urgency
    CHECK_INTERVAL = 120        # Poll interval (seconds)

    def __init__(self, calendar_dir: Optional[str] = None) -> None:
        self._calendar_dir = Path(calendar_dir) if calendar_dir else Path.home() / "Documents"
        self._last_check = 0.0
        self._events_found = 0
        logger.info(f"CalendarSensor initialized (dir={self._calendar_dir})")

    def poll(self) -> List[CalendarEvent]:
        """Poll for imminent events. Returns list of urgent events."""
        now = time.time()
        if now - self._last_check < self.CHECK_INTERVAL:
            return []
        self._last_check = now

        events = self._parse_all()
        imminent = []
        now_dt = datetime.now()

        for event in events:
            delta = event.start - now_dt
            event.minutes_until = delta.total_seconds() / 60.0

            if 0 <= event.minutes_until <= self.IMMINENT_MINUTES:
                event.is_imminent = True
                event.urgency = 0.7 + (1.0 - event.minutes_until / self.IMMINENT_MINUTES) * 0.3
                imminent.append(event)

        if imminent:
            logger.info(f"Calendar: {len(imminent)} imminent events detected")
        return imminent

    def _parse_all(self) -> List[CalendarEvent]:
        events = []
        if not self._calendar_dir.exists():
            return events
        for ics_path in self._calendar_dir.rglob("*.ics"):
            parsed = self._parse_ics(ics_path)
            events.extend(parsed)
        self._events_found = len(events)
        return events

    def _parse_ics(self, path: Path) -> List[CalendarEvent]:
        """Parse a single .ics file."""
        try:
            text = path.read_text(errors='replace')
        except Exception:
            return []
        events = []
        blocks = text.split("BEGIN:VEVENT")
        for block in blocks[1:]:
            block = block.split("END:VEVENT")[0] if "END:VEVENT" in block else block
            title = self._extract_field(block, "SUMMARY")
            start_str = self._extract_field(block, "DTSTART")
            end_str = self._extract_field(block, "DTEND")
            location = self._extract_field(block, "LOCATION")
            desc = self._extract_field(block, "DESCRIPTION")

            start = self._parse_dt(start_str)
            if not start or not title:
                continue
            end = self._parse_dt(end_str) if end_str else None

            events.append(CalendarEvent(
                title=title, start=start, end=end,
                location=location, description=desc,
                source_file=str(path),
            ))
        return events

    def _extract_field(self, block: str, field: str) -> str:
        for line in block.split("\n"):
            if line.startswith(field):
                val = line.split(":", 1)[-1] if ":" in line else line.split(";", 1)[-1]
                return val.replace("\\,", ",").replace("\\n", "\n").strip()
        return ""

    def _parse_dt(self, dt_str: str) -> Optional[datetime]:
        if not dt_str:
            return None
        dt_clean = dt_str.replace("T", "").replace("Z", "").replace("-", "").replace(":", "")
        try:
            return datetime.strptime(dt_clean[:14], "%Y%m%d%H%M%S")
        except (ValueError, IndexError):
            try:
                return datetime.strptime(dt_clean[:8], "%Y%m%d")
            except ValueError:
                return None

    def get_stats(self) -> Dict[str, Any]:
        return {"events_found": self._events_found, "calendar_dir": str(self._calendar_dir)}


if __name__ == "__main__":
    import tempfile, os
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    logger.info("=" * 60); logger.info("CALENDAR SENSOR TEST")

    tmpdir = Path(tempfile.mkdtemp())
    now = datetime.now()
    soon = now + timedelta(minutes=30)
    later = now + timedelta(hours=3)

    ics = tmpdir / "test.ics"
    ics.write_text(f"""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
DTSTART:{soon.strftime('%Y%m%dT%H%M%S')}
SUMMARY:Company Law Exam
LOCATION:Room 302
END:VEVENT
BEGIN:VEVENT
DTSTART:{later.strftime('%Y%m%dT%H%M%S')}
SUMMARY:Study Group
END:VEVENT
END:VCALENDAR""")

    sensor = CalendarSensor(calendar_dir=str(tmpdir))
    events = sensor.poll()
    assert len(events) >= 1, f"Expected imminent events, got {len(events)}"
    assert any("Company Law" in e.title for e in events)
    imminent = [e for e in events if e.is_imminent]
    assert len(imminent) >= 1
    logger.info(f"✓ Imminent events: {len(imminent)} (urgency={imminent[0].urgency:.2f})")
    logger.info(f"✓ Stats: {sensor.get_stats()}")

    import shutil; shutil.rmtree(tmpdir)
    logger.info("ALL CALENDAR TESTS PASSED")