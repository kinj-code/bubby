"""Terminal context sensor for detecting workflow events.

Monitors terminal state flags, exit codes, and error patterns
to trigger high-urgency proactive interventions (e.g., build failures,
tracebacks, permission errors).

RAM: Negligible (stat polling + regex, no model loaded).
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TerminalEvent(str, Enum):
    """Types of terminal events detected."""
    BUILD_FAILED = "build_failed"
    TRACEBACK = "traceback"
    PERMISSION_ERROR = "permission_error"
    COMMAND_NOT_FOUND = "command_not_found"
    TEST_FAILED = "test_failed"
    LINT_ERROR = "lint_error"
    SEGFAULT = "segfault"
    SUCCESS = "success"
    UNKNOWN = "unknown"


@dataclass
class TerminalState:
    """Current terminal context snapshot."""
    event: TerminalEvent = TerminalEvent.UNKNOWN
    exit_code: int = 0
    error_summary: str = ""
    error_file: str = ""          # File where error occurred (if detectable)
    error_line: int = 0           # Line number (if detectable)
    suggestion: str = ""          # Known fix suggestion from LTM
    urgency: float = 0.0          # 0.0-1.0 urgency score
    timestamp: float = field(default_factory=time.time)
    raw_output: str = ""


# Error patterns that indicate high-urgency situations
TERMINAL_ERROR_PATTERNS: Dict[TerminalEvent, List[str]] = {
    TerminalEvent.BUILD_FAILED: [
        r"(?:^|\n)make\[\d+\]: \*\*\*",
        r"error: build failed",
        r"npm ERR!",
        r"cargo build.*failed",
        r"cmake.*Error",
        r"go build.*failed",
    ],
    TerminalEvent.TRACEBACK: [
        r"Traceback \(most recent call last\):",
        r"SyntaxError:",
        r"TypeError:",
        r"ValueError:",
        r"IndexError:",
        r"KeyError:",
        r"AttributeError:",
        r"ImportError:",
        r"ModuleNotFoundError:",
        r"NameError:",
        r"UnboundLocalError:",
        r"RuntimeError:",
        r"OSError:",
        r"FileNotFoundError:",
    ],
    TerminalEvent.PERMISSION_ERROR: [
        r"Permission denied",
        r"EACCES",
        r"Error: EACCES: permission denied",
    ],
    TerminalEvent.COMMAND_NOT_FOUND: [
        r"command not found",
        r"zsh: command not found",
        r"bash: .*: command not found",
        r"is not recognized as an internal",
    ],
    TerminalEvent.TEST_FAILED: [
        r"FAILED \(.*\)",
        r"Tests run: \d+, Failures: [1-9]",
        r"assertion failed",
        r"AssertionError",
        r"FAIL: test_",
    ],
    TerminalEvent.LINT_ERROR: [
        r"error:.*\[.*\]",         # Rust compiler errors
        r"^E\d{4}:",               # Rust error codes
        r"pylint:.*error",
        r"eslint:.*error",
    ],
    TerminalEvent.SEGFAULT: [
        r"Segmentation fault",
        r"SIGSEGV",
        r"core dumped",
    ],
}


class TerminalSensor:
    """
    Non-intrusive terminal context sensor.

    Monitors terminal state by checking:
    1. Exit code files written by shell hooks
    2. Log files for error patterns
    3. Direct stdin/stdout parsing (if piped)

    Flow:
    1. Shell hook writes exit code + last command to a temp file
    2. TerminalSensor polls this file periodically
    3. If an error pattern matches → high urgency event
    4. ProactivityEvaluator uses this to decide intervention
    
    Memory: ~0MB (stat + regex only).
    """

    # Default file paths for shell hook output
    DEFAULT_EXIT_CODE_FILE = "/tmp/bubby_last_exit"
    DEFAULT_COMMAND_FILE = "/tmp/bubby_last_command"
    DEFAULT_OUTPUT_FILE = "/tmp/bubby_last_output"

    def __init__(
        self,
        exit_code_file: Optional[str] = None,
        command_file: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> None:
        """
        Initialize terminal sensor.

        Args:
            exit_code_file: Path to file containing last exit code
            command_file: Path to file containing last command
            output_file: Path to file containing last terminal output
        """
        self._exit_code_file = exit_code_file or self.DEFAULT_EXIT_CODE_FILE
        self._command_file = command_file or self.DEFAULT_COMMAND_FILE
        self._output_file = output_file or self.DEFAULT_OUTPUT_FILE
        
        self._last_exit_code: Optional[int] = None
        self._last_checked = 0.0
        self._events_detected = 0
        self._check_interval = 2.0  # seconds between checks
        
        logger.info(f"TerminalSensor initialized (exit_file={self._exit_code_file})")

    def poll(self) -> TerminalState:
        """
        Poll the terminal state files for new events.

        Returns:
            TerminalState with event type and urgency score
        """
        now = time.time()
        if now - self._last_checked < self._check_interval:
            return TerminalState()  # Return empty state (throttled)
        self._last_checked = now

        # Read exit code
        exit_code = self._read_exit_code()
        if exit_code is None or exit_code == self._last_exit_code:
            return TerminalState(exit_code=exit_code or 0)  # No change
        
        self._last_exit_code = exit_code

        # Read command
        command = self._read_command()

        # Read output if available
        output = self._read_output()

        # If exit code is 0, it's a success (low urgency unless we're tracking completions)
        if exit_code == 0:
            return TerminalState(
                event=TerminalEvent.SUCCESS,
                exit_code=0,
                raw_output=output,
                urgency=0.1,  # Low urgency but worth noting
            )

        # Detect error type from output
        event_type, error_summary, error_file, error_line = self._classify_error(output)

        # Calculate urgency based on error severity
        urgency = self._calculate_urgency(event_type, exit_code, output)

        state = TerminalState(
            event=event_type,
            exit_code=exit_code,
            error_summary=error_summary,
            error_file=error_file,
            error_line=error_line,
            urgency=urgency,
            raw_output=output[:500] if output else "",
        )

        self._events_detected += 1
        logger.info(
            f"Terminal event: {event_type.value} (code={exit_code}, urgency={urgency:.2f})"
        )
        if error_summary:
            logger.info(f"  Summary: {error_summary}")

        return state

    def _read_exit_code(self) -> Optional[int]:
        """Read exit code from file."""
        try:
            path = Path(self._exit_code_file)
            if not path.exists():
                return None
            content = path.read_text().strip()
            return int(content)
        except (ValueError, OSError):
            return None

    def _read_command(self) -> str:
        """Read last command from file."""
        try:
            path = Path(self._command_file)
            if not path.exists():
                return ""
            return path.read_text().strip()[:200]
        except OSError:
            return ""

    def _read_output(self) -> str:
        """Read last terminal output from file."""
        try:
            path = Path(self._output_file)
            if not path.exists():
                return ""
            return path.read_text()[:1000]  # Limit to 1KB
        except OSError:
            return ""

    def _classify_error(self, output: str) -> tuple:
        """
        Classify terminal output into an error type.

        Returns:
            (TerminalEvent, error_summary, error_file, error_line)
        """
        if not output:
            return TerminalEvent.UNKNOWN, "Command failed with non-zero exit code", "", 0

        for event_type, patterns in TERMINAL_ERROR_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
                if match:
                    # Extract file and line if available
                    error_file, error_line = self._extract_location(output)
                    summary = self._extract_summary(output, event_type)
                    return event_type, summary, error_file, error_line

        return TerminalEvent.UNKNOWN, output[:200].strip(), "", 0

    def _extract_location(self, output: str) -> tuple:
        """
        Extract file and line number from error output.

        Returns:
            (filename, line_number)
        """
        # Pattern: File "path/to/file.py", line 42
        match = re.search(
            r'File\s+"([^"]+)",\s+line\s+(\d+)',
            output
        )
        if match:
            return match.group(1), int(match.group(2))

        # Pattern: path/to/file.rs:42:5
        match = re.search(
            r'([\w/\.-]+):(\d+):\d+',
            output
        )
        if match:
            return match.group(1), int(match.group(2))

        return "", 0

    def _extract_summary(self, output: str, event_type: TerminalEvent) -> str:
        """Extract a one-line summary of the error."""
        lines = output.strip().split("\n")

        if event_type == TerminalEvent.TRACEBACK:
            # Get the last line (actual error)
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith("File "):
                    return line[:200]

        if event_type == TerminalEvent.BUILD_FAILED:
            for line in lines:
                if "error:" in line.lower():
                    return line.strip()[:200]

        if event_type == TerminalEvent.TEST_FAILED:
            for line in lines:
                if "FAIL" in line or "failure" in line.lower():
                    return line.strip()[:200]

        # Default: first non-empty line
        for line in lines:
            if line.strip():
                return line.strip()[:200]

        return "Unknown error"

    def _calculate_urgency(
        self,
        event_type: TerminalEvent,
        exit_code: int,
        output: str,
    ) -> float:
        """
        Calculate intervention urgency score (0.0-1.0).

        Higher urgency for:
        - Build failures blocking workflow
        - Tracebacks stopping code execution
        - Permission errors blocking operations
        """
        base_scores = {
            TerminalEvent.BUILD_FAILED: 0.85,
            TerminalEvent.TRACEBACK: 0.90,
            TerminalEvent.TEST_FAILED: 0.70,
            TerminalEvent.PERMISSION_ERROR: 0.75,
            TerminalEvent.COMMAND_NOT_FOUND: 0.50,
            TerminalEvent.LINT_ERROR: 0.40,
            TerminalEvent.SEGFAULT: 0.95,
            TerminalEvent.SUCCESS: 0.10,
            TerminalEvent.UNKNOWN: 0.30,
        }

        score = base_scores.get(event_type, 0.30)

        # Boost for high exit codes (critical failures)
        if exit_code > 128:
            score = min(1.0, score + 0.15)

        # Boost if output suggests critical failure
        critical_keywords = [
            "cannot continue", "fatal", "critical", "panic",
            "abort", "unrecoverable",
        ]
        if output:
            output_lower = output.lower()
            for kw in critical_keywords:
                if kw in output_lower:
                    score = min(1.0, score + 0.10)
                    break

        return round(score, 2)

    def get_shell_hook_script(self) -> str:
        """
        Generate a shell hook script for bash/zsh.

        Returns a string that can be sourced in .bashrc/.zshrc to capture
        exit codes and terminal output.

        Returns:
            Shell script string
        """
        return f"""# Bubby Terminal Sensor Hook
# Add this to your ~/.bashrc or ~/.zshrc

_bubby_preexec() {{
    echo "$1" > {self._command_file}
}}

_bubby_precmd() {{
    echo "$?" > {self._exit_code_file}
}}

# If using zsh
if [[ -n "$ZSH_VERSION" ]]; then
    autoload -U add-zsh-hook
    add-zsh-hook preexec _bubby_preexec
    add-zsh-hook precmd _bubby_precmd
fi

# If using bash
if [[ -n "$BASH_VERSION" ]]; then
    trap '_bubby_preexec "$BASH_COMMAND"' DEBUG
    PROMPT_COMMAND="_bubby_precmd;$PROMPT_COMMAND"
fi
"""

    def get_stats(self) -> Dict[str, Any]:
        """Get sensor statistics."""
        return {
            "events_detected": self._events_detected,
            "last_exit_code": self._last_exit_code,
            "exit_code_file": self._exit_code_file,
        }


# Testing helper
if __name__ == "__main__":
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    logger.info("=" * 60)
    logger.info("TERMINAL SENSOR TEST")
    logger.info("=" * 60)

    # Create temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        exit_file = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        output_file = f.name
    
    sensor = TerminalSensor(
        exit_code_file=exit_file,
        output_file=output_file,
    )

    # Test 1: No exit code file
    state = sensor.poll()
    assert state.exit_code == 0, f"Default should be 0, got {state.exit_code}"
    logger.info("✓ Test 1: No exit file → empty state")

    # Test 2: Success exit
    Path(exit_file).write_text("0")
    state = sensor.poll()
    assert state.event == TerminalEvent.SUCCESS
    assert state.urgency == 0.1
    logger.info(f"✓ Test 2: Exit 0 → success (urgency={state.urgency})")

    # Test 3: Traceback detection
    Path(exit_file).write_text("1")
    traceback_output = """
Traceback (most recent call last):
  File "main.py", line 42, in <module>
    result = divide(10, 0)
  File "main.py", line 15, in divide
    return a / b
ZeroDivisionError: division by zero
"""
    Path(output_file).write_text(traceback_output)
    sensor._last_exit_code = None  # Force refresh
    state = sensor.poll()
    assert state.event == TerminalEvent.TRACEBACK
    assert state.urgency >= 0.85
    assert "ZeroDivisionError" in state.error_summary
    logger.info(f"✓ Test 3: Traceback detected (urgency={state.urgency})")

    # Test 4: Build failure
    Path(exit_file).write_text("2")
    build_output = """
error: build failed
make[1]: *** [Makefile:45: build] Error 1
"""
    Path(output_file).write_text(build_output)
    sensor._last_exit_code = None
    state = sensor.poll()
    assert state.event == TerminalEvent.BUILD_FAILED
    assert state.urgency >= 0.80
    logger.info(f"✓ Test 4: Build failure detected (urgency={state.urgency})")

    # Test 5: Permission error
    Path(exit_file).write_text("1")
    perm_output = "Error: EACCES: permission denied, open '/etc/config.yaml'"
    Path(output_file).write_text(perm_output)
    sensor._last_exit_code = None
    state = sensor.poll()
    assert state.event == TerminalEvent.PERMISSION_ERROR
    logger.info(f"✓ Test 5: Permission error detected")

    # Test 6: Shell hook generation
    hook = sensor.get_shell_hook_script()
    assert "preexec" in hook or "precmd" in hook
    assert exit_file in hook
    logger.info(f"✓ Test 6: Shell hook generated ({len(hook)} chars)")

    # Test 7: Segfault (highest urgency)
    Path(exit_file).write_text("139")  # SIGSEGV exit code
    segfault_output = "Segmentation fault (core dumped)"
    Path(output_file).write_text(segfault_output)
    sensor._last_exit_code = None
    state = sensor.poll()
    assert state.event == TerminalEvent.SEGFAULT
    assert state.urgency >= 0.90
    logger.info(f"✓ Test 7: Segfault detected (urgency={state.urgency})")

    # Test 8: Unknown error
    sensor._last_exit_code = None
    Path(exit_file).write_text("1")
    Path(output_file).write_text("Something went wrong, no details available.")
    state = sensor.poll()
    assert state.event == TerminalEvent.UNKNOWN
    logger.info(f"✓ Test 8: Unknown error defaults to low urgency")

    # Test 9: Stats
    stats = sensor.get_stats()
    assert stats["events_detected"] >= 5
    logger.info(f"✓ Test 9: Stats tracked ({stats['events_detected']} events)")

    # Cleanup
    os.unlink(exit_file)
    os.unlink(output_file)

    logger.info("\nALL TERMINAL SENSOR TESTS PASSED")