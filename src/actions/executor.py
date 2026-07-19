"""Secure system action executor with strict command whitelisting.

The SystemExecutor is the ONLY path through which Bubby can execute
system commands. Every action is validated against a hardcoded whitelist
before execution. Untrusted or hallucinated commands are logged and rejected.

RAM: Negligible (subprocess delegation uses OS memory).
"""

import logging
import subprocess
import shlex
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ActionCategory(str, Enum):
    """Categories of permitted system actions."""
    SYSTEM_INFO = "system_info"      # Read-only system queries
    UTILITY = "utility"              # Launch apps, open files
    POWER = "power"                  # Lock screen, sleep, etc.
    FILE_OPS = "file_ops"            # Read/summarize files (read-only)
    NOTIFICATION = "notification"    # Send desktop notifications
    DISPLAY = "display"             # Brightness, display settings


@dataclass(frozen=True)
class WhitelistedCommand:
    """A single whitelisted command with metadata."""
    name: str                        # Friendly name for logging
    command_template: str            # Command with optional {param} placeholders
    category: ActionCategory         # Category for grouping
    description: str                 # What this command does
    requires_approval: bool = False  # Whether user confirmation is needed
    max_params: int = 0              # Max number of additional parameters
    allowed_params: Optional[List[str]] = None  # Allowed param values (None = any)


# The strict whitelist — every executable command must be registered here
COMMAND_WHITELIST: Dict[str, WhitelistedCommand] = {
    # ── SYSTEM INFO ──
    "check_battery": WhitelistedCommand(
        name="Check Battery",
        command_template="upower -i /org/freedesktop/UPower/devices/battery_BAT0 2>/dev/null | grep -E 'percentage|state' || echo 'Battery info not available'",
        category=ActionCategory.SYSTEM_INFO,
        description="Get battery percentage and status",
    ),
    "check_disk": WhitelistedCommand(
        name="Check Disk Space",
        command_template="df -h / | tail -1 | awk '{print \"Disk: \" $3 \" used / \" $2 \" total (\" $5 \")\"}'",
        category=ActionCategory.SYSTEM_INFO,
        description="Get disk usage for root partition",
    ),
    "check_memory": WhitelistedCommand(
        name="Check Memory",
        command_template="free -h | awk '/^Mem:/ {printf \"Memory: %s used / %s total (%.0f%%)\\n\", $3, $2, ($3/$2)*100}'",
        category=ActionCategory.SYSTEM_INFO,
        description="Get RAM usage statistics",
    ),
    "check_uptime": WhitelistedCommand(
        name="Check Uptime",
        command_template="uptime -p | sed 's/up //'",
        category=ActionCategory.SYSTEM_INFO,
        description="Get system uptime",
    ),
    "check_cpu": WhitelistedCommand(
        name="Check CPU",
        command_template="top -bn1 | grep 'Cpu(s)' | awk '{print \"CPU: \" 100-$8 \"% used\"}'",
        category=ActionCategory.SYSTEM_INFO,
        description="Get CPU usage percentage",
    ),
    "check_date": WhitelistedCommand(
        name="Check Date",
        command_template="date '+%A, %B %d, %Y — %I:%M %p'",
        category=ActionCategory.SYSTEM_INFO,
        description="Get current date and time",
    ),
    "check_weather": WhitelistedCommand(
        name="Check Weather",
        command_template="curl -s 'wttr.in/?format=%C+%t' 2>/dev/null || echo 'Weather unavailable offline'",
        category=ActionCategory.SYSTEM_INFO,
        description="Get current weather (requires internet)",
    ),

    # ── UTILITY ──
    "open_terminal": WhitelistedCommand(
        name="Open Terminal",
        command_template="gnome-terminal -- bash -c 'echo Welcome back!; exec bash' &",
        category=ActionCategory.UTILITY,
        description="Open a new terminal window",
    ),
    "open_calculator": WhitelistedCommand(
        name="Open Calculator",
        command_template="gnome-calculator &",
        category=ActionCategory.UTILITY,
        description="Open the calculator app",
    ),
    "open_files": WhitelistedCommand(
        name="Open File Manager",
        command_template="nautilus --new-window &",
        category=ActionCategory.UTILITY,
        description="Open the file manager",
    ),
    "open_browser": WhitelistedCommand(
        name="Open Browser",
        command_template="xdg-open https://www.google.com &",
        category=ActionCategory.UTILITY,
        description="Open the default web browser",
    ),
    "open_settings": WhitelistedCommand(
        name="Open Settings",
        command_template="gnome-control-center &",
        category=ActionCategory.UTILITY,
        description="Open system settings",
    ),
    "open_vscode": WhitelistedCommand(
        name="Open VS Code",
        command_template="code . &",
        category=ActionCategory.UTILITY,
        description="Open VS Code in current directory",
    ),
    "take_screenshot": WhitelistedCommand(
        name="Take Screenshot",
        command_template="gnome-screenshot &",
        category=ActionCategory.UTILITY,
        description="Take a screenshot",
    ),

    # ── POWER ──
    "lock_screen": WhitelistedCommand(
        name="Lock Screen",
        command_template="xdg-screensaver lock 2>/dev/null || loginctl lock-session",
        category=ActionCategory.POWER,
        description="Lock the screen",
        requires_approval=True,
    ),
    "sleep_system": WhitelistedCommand(
        name="Sleep System",
        command_template="systemctl suspend",
        category=ActionCategory.POWER,
        description="Put the system to sleep",
        requires_approval=True,
    ),

    # ── NOTIFICATIONS ──
    "send_notification": WhitelistedCommand(
        name="Send Notification",
        command_template="notify-send '{param1}' '{param2}'",
        category=ActionCategory.NOTIFICATION,
        description="Send a desktop notification",
        max_params=2,
    ),

    # ── DISPLAY ──
    "brightness_up": WhitelistedCommand(
        name="Brightness Up",
        command_template="brightnessctl set +10% 2>/dev/null || echo 'brightnessctl not available'",
        category=ActionCategory.DISPLAY,
        description="Increase screen brightness by 10%",
    ),
    "brightness_down": WhitelistedCommand(
        name="Brightness Down",
        command_template="brightnessctl set 10%- 2>/dev/null || echo 'brightnessctl not available'",
        category=ActionCategory.DISPLAY,
        description="Decrease screen brightness by 10%",
    ),
    "volume_up": WhitelistedCommand(
        name="Volume Up",
        command_template="pactl set-sink-volume @DEFAULT_SINK@ +5%",
        category=ActionCategory.DISPLAY,
        description="Increase volume by 5%",
    ),
    "volume_down": WhitelistedCommand(
        name="Volume Down",
        command_template="pactl set-sink-volume @DEFAULT_SINK@ -5%",
        category=ActionCategory.DISPLAY,
        description="Decrease volume by 5%",
    ),
    "volume_mute": WhitelistedCommand(
        name="Mute/Unmute",
        command_template="pactl set-sink-mute @DEFAULT_SINK@ toggle",
        category=ActionCategory.DISPLAY,
        description="Toggle mute on/off",
    ),
}


@dataclass(frozen=True)
class ActionRequest:
    """A validated action request from the LLM."""
    action: str
    params: List[str] = field(default_factory=list)
    is_valid: bool = False
    command: Optional[WhitelistedCommand] = None


@dataclass(frozen=True)
class ActionResult:
    """Result of executing a system action."""
    action: str
    success: bool
    output: str = ""
    error: str = ""
    exit_code: int = 0
    requires_approval: bool = False


class SystemExecutor:
    """
    Secure executor for whitelisted system commands.

    Flow:
    1. LLM generates {"action": "check_battery", "speech": "...", "animation": "..."}
    2. InteractionHandler routes action to SystemExecutor.validate()
    3. If valid and not requiring approval → execute()
    4. If requiring approval → callback to user for confirmation
    5. If invalid → log and ignore (never execute unknown commands)
    """

    def __init__(self, whitelist: Optional[Dict[str, WhitelistedCommand]] = None) -> None:
        """
        Initialize the system executor.

        Args:
            whitelist: Custom whitelist (defaults to COMMAND_WHITELIST)
        """
        self._whitelist = whitelist or COMMAND_WHITELIST
        self._executed_count = 0
        self._rejected_count = 0
        self._errors = 0
        logger.info(
            f"SystemExecutor initialized with {len(self._whitelist)} whitelisted commands "
            f"across {len(set(c.category for c in self._whitelist.values()))} categories"
        )

    def validate(self, action_name: str, params: Optional[List[str]] = None) -> ActionRequest:
        """
        Validate an action against the whitelist.

        Args:
            action_name: The action string from the LLM
            params: Optional parameters for parameterized commands

        Returns:
            ActionRequest with validation result
        """
        if not action_name:
            return ActionRequest(action="", is_valid=False)

        action_name = action_name.strip().lower()
        params = params or []

        # Check whitelist
        command = self._whitelist.get(action_name)
        if not command:
            logger.warning(f"Rejected unknown action: '{action_name}'")
            self._rejected_count += 1
            return ActionRequest(action=action_name, params=params, is_valid=False)

        # Validate parameter count
        if len(params) > command.max_params:
            logger.warning(
                f"Rejected action '{action_name}': too many params "
                f"({len(params)} > {command.max_params})"
            )
            self._rejected_count += 1
            return ActionRequest(action=action_name, params=params, is_valid=False)

        # Validate parameter values if restricted
        if command.allowed_params is not None and params:
            for param in params:
                if param not in command.allowed_params:
                    logger.warning(
                        f"Rejected action '{action_name}': param '{param}' not in "
                        f"allowed values {command.allowed_params}"
                    )
                    self._rejected_count += 1
                    return ActionRequest(action=action_name, params=params, is_valid=False)

        logger.debug(f"Validated action: '{action_name}' (category={command.category})")
        return ActionRequest(
            action=action_name,
            params=params,
            is_valid=True,
            command=command,
        )

    def execute(self, request: ActionRequest) -> ActionResult:
        """
        Execute a validated action request.

        Only executes if the action is validated AND does not require
        user approval. Actions requiring approval must be routed through
        the confirmation flow first.

        Args:
            request: Validated ActionRequest

        Returns:
            ActionResult with execution output
        """
        if not request.is_valid or not request.command:
            return ActionResult(
                action=request.action,
                success=False,
                error="Action not validated — rejected",
            )

        command = request.command

        # Check approval requirement
        if command.requires_approval:
            logger.info(
                f"Action '{request.action}' requires user approval — "
                f"deferring execution"
            )
            return ActionResult(
                action=request.action,
                success=False,
                requires_approval=True,
                error="User approval required",
            )

        # Build command string
        cmd_str = command.command_template

        # Substitute parameters
        for i, param in enumerate(request.params[:command.max_params]):
            placeholder = f"{{param{i + 1}}}"
            # Basic sanitization: strip shell metacharacters
            safe_param = param.replace("'", "").replace('"', "").replace("`", "").replace("$", "")
            cmd_str = cmd_str.replace(placeholder, safe_param)

        logger.info(f"Executing: [{command.name}] {cmd_str[:80]}")

        try:
            # Execute with timeout and security constraints
            result = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True,
                timeout=15,  # 15 second timeout
                env={},      # Empty env for security (system commands don't need env)
                cwd="/",     # Root dir for safety
            )

            output = result.stdout.strip()
            error = result.stderr.strip()

            if result.returncode == 0:
                self._executed_count += 1
                logger.info(f"✓ [{command.name}] succeeded: {output[:80]}")
                return ActionResult(
                    action=request.action,
                    success=True,
                    output=output,
                    exit_code=0,
                )
            else:
                self._errors += 1
                logger.warning(f"✗ [{command.name}] failed (exit={result.returncode}): {error[:80]}")
                return ActionResult(
                    action=request.action,
                    success=False,
                    output=output,
                    error=error,
                    exit_code=result.returncode,
                )

        except subprocess.TimeoutExpired:
            self._errors += 1
            logger.error(f"Timeout executing '{request.action}'")
            return ActionResult(
                action=request.action,
                success=False,
                error="Command timed out after 15 seconds",
                exit_code=-1,
            )
        except Exception as e:
            self._errors += 1
            logger.error(f"Error executing '{request.action}': {e}")
            return ActionResult(
                action=request.action,
                success=False,
                error=str(e),
                exit_code=-1,
            )

    def get_available_actions(self) -> List[str]:
        """Get list of all available action names."""
        return list(self._whitelist.keys())

    def get_actions_by_category(self, category: ActionCategory) -> List[str]:
        """Get action names filtered by category."""
        return [
            name for name, cmd in self._whitelist.items()
            if cmd.category == category
        ]

    def get_whitelist_for_prompt(self) -> str:
        """
        Build a formatted whitelist summary for inclusion in the LLM system prompt.

        Returns:
            Formatted string listing available actions grouped by category
        """
        lines = ["Available system actions (only these may be used):"]
        by_category: Dict[ActionCategory, List[str]] = {}
        for name, cmd in self._whitelist.items():
            by_category.setdefault(cmd.category, []).append(f"    - {name}: {cmd.description}")

        for category in ActionCategory:
            if category in by_category:
                lines.append(f"\n  {category.value}:")
                lines.extend(by_category[category])

        lines.append("\n  Leave 'action' as empty string '' if no system action is needed.")
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "whitelisted_commands": len(self._whitelist),
            "executed": self._executed_count,
            "rejected": self._rejected_count,
            "errors": self._errors,
        }


# Testing helper
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    logger.info("=" * 60)
    logger.info("SYSTEM EXECUTOR TEST")
    logger.info("=" * 60)

    executor = SystemExecutor()

    # Test 1: Valid action
    result = executor.validate("check_battery")
    assert result.is_valid
    logger.info(f"✓ Validated: {result.action} → {result.command.name}")

    # Test 2: Invalid action (hallucinated)
    result = executor.validate("delete_all_files")
    assert not result.is_valid
    logger.info(f"✓ Rejected hallucinated: {result.action}")

    # Test 3: Action requiring approval
    result = executor.validate("lock_screen")
    assert result.is_valid
    assert result.command.requires_approval
    logger.info(f"✓ Requires approval: {result.command.name}")

    # Test 4: Valid action with params
    result = executor.validate("send_notification", params=["Hello!", "Bubby here!"])
    assert result.is_valid
    logger.info(f"✓ Validated with params: {result.action}")

    # Test 5: Too many params
    result = executor.validate("send_notification", params=["a", "b", "c"])
    assert not result.is_valid
    logger.info(f"✓ Rejected too many params")

    # Test 6: Info-only execution (safe)
    result = executor.validate("check_date")
    assert result.is_valid
    exec_result = executor.execute(result)
    logger.info(f"✓ Executed: success={exec_result.success}, output='{exec_result.output}'")

    # Test 7: Approval-required execution
    result = executor.validate("lock_screen")
    exec_result = executor.execute(result)
    assert exec_result.requires_approval
    logger.info(f"✓ Approval required correctly flagged")

    # Test 8: Whitelist for prompt
    prompt_text = executor.get_whitelist_for_prompt()
    assert "check_battery" in prompt_text
    assert "lock_screen" in prompt_text
    logger.info(f"✓ Whitelist prompt generated ({len(prompt_text)} chars)")

    # Test 9: Get by category
    system_info_actions = executor.get_actions_by_category(ActionCategory.SYSTEM_INFO)
    assert len(system_info_actions) >= 5
    logger.info(f"✓ System info actions: {len(system_info_actions)}")

    # Test 10: Stats
    stats = executor.get_stats()
    assert stats["rejected"] == 2
    assert stats["executed"] == 1
    logger.info(f"✓ Stats: {stats}")

    logger.info("\n" + "=" * 60)
    logger.info("ALL EXECUTOR TESTS PASSED")
    logger.info("=" * 60)