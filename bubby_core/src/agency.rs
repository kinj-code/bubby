//! Deterministic Finite State Machine for LLM agency governance.
//!
//! The LLM is a logic engine, not a state machine. This FSM wraps
//! all tool-use logic and enforces strict transition rules. The LLM
//! can *suggest* state changes; the FSM validates and gates them.
//!
//! States:
//!   IDLE        — waiting for user input or trigger
//!   RESEARCHING — RAG active, retrieving memories
//!   PLANNING    — LLM generating a plan / tool calls
//!   EXECUTING   — running a validated system command
//!   ERROR       — anomaly detected; reverts to IDLE after logging
//!
//! Transitions (LLM-suggested → FSM-gated):
//!   IDLE        → RESEARCHING  (user query received)
//!   RESEARCHING → PLANNING     (RAG complete)
//!   PLANNING    → EXECUTING    (tool call validated)
//!   PLANNING    → IDLE         (no action needed)
//!   EXECUTING   → IDLE         (command finished or blocked)
//!   EXECUTING   → ERROR        (forbidden command attempted)
//!   *           → ERROR        (any illegal transition)
//!   ERROR       → IDLE         (after recording anomaly)

use serde::{Deserialize, Serialize};
use std::fmt;

// ── State definitions ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum State {
    Idle,
    Researching,
    Planning,
    Executing,
    Error,
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ── LLM action request ────────────────────────────────────────────

/// What the LLM wants to do. The FSM validates this before acting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRequest {
    /// Requested next state, e.g. "EXECUTING"
    pub transition: String,
    /// Tool to call, e.g. "bash", "file_write"
    pub tool: String,
    /// Command string (for bash/shell tools)
    pub cmd: String,
    /// Human-readable reason (logged on rejection)
    pub reason: String,
}

/// FSM response to an action request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResponse {
    /// Did the FSM allow the request?
    pub allowed: bool,
    /// The resulting state after the request
    pub new_state: State,
    /// Output to send back to the LLM context
    pub message: String,
    /// The executed command output (only for EXECUTING → Idle)
    pub command_output: Option<String>,
}

// ── Forbidden command detection ───────────────────────────────────

/// Check if a shell command contains known destructive patterns.
/// This is a **hardcoded safety net** — independent of the Python whitelist.
pub fn is_forbidden_command(cmd: &str) -> bool {
    let lower = cmd.to_lowercase();

    // Pattern matching — exact substrings that are never safe
    let forbidden_patterns = [
        // Destructive file operations
        "rm -rf", "rm -r", "rmdir",
        // Privilege escalation
        "sudo ", "su ", "doas ",
        // Filesystem formatting
        "mkfs.", "mkswap", "dd if=",
        // Fork bombs and resource exhaustion
        ":(){ :|:& };:", "fork bomb",
        // System takeover
        "chmod 777", "chown -R",
        // Network exfiltration
        "nc -l", "ncat -l",
        // Kernel module loading
        "modprobe", "insmod",
        // Raw device manipulation
        "/dev/sda", "/dev/nvme",
        // Kill critical processes
        "kill -9 1", "killall init",
    ];

    for pattern in &forbidden_patterns {
        if lower.contains(pattern) {
            return true;
        }
    }

    // Also block any command starting with a dangerous binary
    let dangerous_prefixes = ["rm ", "shred ", "mkfs", "dd ", "chroot "];
    for prefix in &dangerous_prefixes {
        if lower.starts_with(prefix) {
            return true;
        }
    }

    false
}

// ── FSM ───────────────────────────────────────────────────────────

pub struct AgencyFSM {
    state: State,
    /// Log of rejected requests for anomaly tracking
    rejection_log: Vec<String>,
}

impl AgencyFSM {
    pub fn new() -> Self {
        Self {
            state: State::Idle,
            rejection_log: Vec::new(),
        }
    }

    /// Current state.
    pub fn state(&self) -> State {
        self.state
    }

    /// Process a state transition suggestion from the LLM.
    /// Returns true if the transition is valid.
    pub fn validate_transition(&self, target: State) -> Result<State, String> {
        match (self.state, target) {
            // ── Permitted transitions ─────────────────────────
            (State::Idle, State::Researching) |
            (State::Researching, State::Planning) |
            (State::Planning, State::Executing) |
            (State::Planning, State::Idle) |
            (State::Executing, State::Idle) |
            (State::Error, State::Idle) => Ok(target),

            // ── Self-transitions (no-op) ──────────────────────
            _ if self.state == target => Ok(self.state),

            // ── Everything else — REJECTED ────────────────────
            _ => Err(format!(
                "Illegal transition: {} → {} is not permitted",
                self.state, target
            )),
        }
    }

    /// Transition to a new state if valid. Records the request
    /// in the rejection log if invalid and goes to Error state.
    pub fn apply_transition(&mut self, target: State) -> State {
        match self.validate_transition(target) {
            Ok(new_state) => {
                self.state = new_state;
                new_state
            }
            Err(msg) => {
                self.rejection_log.push(msg.clone());
                self.state = State::Error;
                State::Error
            }
        }
    }

    /// Process a full action request from the LLM.
    ///
    /// This is the main entry point. It:
    /// 1. Parses the suggested transition
    /// 2. Validates it against the current state
    /// 3. If EXECUTING, checks the command against the forbidden list
    /// 4. Returns an ActionResponse with the outcome
    pub fn process_request(&mut self, req: &ActionRequest) -> ActionResponse {
        let target = match req.transition.to_uppercase().as_str() {
            "IDLE" => State::Idle,
            "RESEARCHING" => State::Researching,
            "PLANNING" => State::Planning,
            "EXECUTING" => State::Executing,
            "ERROR" => State::Error,
            _ => {
                // Unrecognized transition name — reject
                let msg = format!(
                    "Unknown transition '{}' — rejected",
                    req.transition
                );
                self.rejection_log.push(msg.clone());
                return ActionResponse {
                    allowed: false,
                    new_state: self.state,
                    message: msg,
                    command_output: None,
                };
            }
        };

        // Validate transition from current state
        let new_state = match self.validate_transition(target) {
            Ok(s) => s,
            Err(msg) => {
                self.rejection_log.push(msg.clone());
                let _ = self.apply_transition(State::Error);
                return ActionResponse {
                    allowed: false,
                    new_state: State::Error,
                    message: format!("FSM BLOCKED: {}", msg),
                    command_output: None,
                };
            }
        };

        // ── EXECUTING state — extra permission gate ────────────
        if target == State::Executing {
            if self.state == State::Planning {
                // Check capability-based permissions

                // 1. Forbidden command check
                if is_forbidden_command(&req.cmd) {
                    let msg = format!(
                        "PERMISSION DENIED: command '{}' contains forbidden patterns. \
                         Transition blocked; entering ERROR state.",
                        &req.cmd[..req.cmd.len().min(80)]
                    );
                    self.rejection_log.push(msg.clone());
                    self.state = State::Error;
                    return ActionResponse {
                        allowed: false,
                        new_state: State::Error,
                        message: msg,
                        command_output: None,
                    };
                }

                // 2. Empty command check
                if req.cmd.trim().is_empty() {
                    let msg = "EXECUTING rejected: empty command".to_string();
                    self.rejection_log.push(msg.clone());
                    self.state = State::Planning;
                    return ActionResponse {
                        allowed: false,
                        new_state: State::Planning,
                        message: msg,
                        command_output: None,
                    };
                }

                // ── All checks passed — allow execution ───────
                self.state = State::Executing;
                return ActionResponse {
                    allowed: true,
                    new_state: State::Executing,
                    message: format!(
                        "Executing '{}' via tool '{}'",
                        &req.cmd[..req.cmd.len().min(60)],
                        req.tool
                    ),
                    command_output: None,
                };
            }
        }

        // For non-EXECUTING transitions, simply apply
        self.state = new_state;
        ActionResponse {
            allowed: true,
            new_state: self.state,
            message: format!(
                "Transitioned to {} (reason: {})",
                self.state,
                &req.reason[..req.reason.len().min(100)]
            ),
            command_output: None,
        }
    }

    /// Complete execution and return to Idle.
    pub fn complete_execution(&mut self, output: String) -> ActionResponse {
        let prev_state = self.state;

        // Only valid from EXECUTING or ERROR
        if prev_state != State::Executing && prev_state != State::Error {
            let msg = format!(
                "Cannot complete execution from {} state",
                prev_state
            );
            self.rejection_log.push(msg.clone());
            return ActionResponse {
                allowed: false,
                new_state: prev_state,
                message: msg,
                command_output: None,
            };
        }

        self.state = State::Idle;
        ActionResponse {
            allowed: true,
            new_state: State::Idle,
            message: "Execution complete; returned to IDLE".into(),
            command_output: Some(output),
        }
    }

    /// Force a transition to ERROR state (e.g., external watchdog).
    pub fn force_error(&mut self, reason: &str) {
        self.rejection_log.push(reason.to_string());
        self.state = State::Error;
    }

    /// Get rejection log for audit purposes.
    pub fn rejection_log(&self) -> &[String] {
        &self.rejection_log
    }

    /// Clear the rejection log.
    pub fn clear_log(&mut self) {
        self.rejection_log.clear();
    }
}

impl Default for AgencyFSM {
    fn default() -> Self {
        Self::new()
    }
}

// ── Safe command wrapper ──────────────────────────────────────────

/// Sanitize a command before execution (shell injection prevention).
/// This is a last-resort filter; the primary defense is the whitelist.
pub fn sanitize_command(cmd: &str) -> String {
    // Strip shell metacharacters that could chain commands
    cmd.replace(';', "")
        .replace("&&", "")
        .replace("||", "")
        .replace('`', "")
        .replace('$', "")
        .replace('\n', " ")
        .trim()
        .to_string()
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── FSM transition tests ──────────────────────────────────

    #[test]
    fn test_valid_transitions() {
        let mut fsm = AgencyFSM::new();

        // Idle → Researching → Planning → Executing → Idle
        assert_eq!(fsm.apply_transition(State::Researching), State::Researching);
        assert_eq!(fsm.apply_transition(State::Planning), State::Planning);
        assert_eq!(fsm.apply_transition(State::Executing), State::Executing);
        assert_eq!(fsm.apply_transition(State::Idle), State::Idle);

        // Rejection log should be empty
        assert!(fsm.rejection_log().is_empty());
    }

    #[test]
    fn test_planning_to_idle_direct() {
        let mut fsm = AgencyFSM::new();
        fsm.apply_transition(State::Researching);
        fsm.apply_transition(State::Planning);
        // LLM decides no action needed → back to IDLE
        assert_eq!(fsm.apply_transition(State::Idle), State::Idle);
        assert!(fsm.rejection_log().is_empty());
    }

    #[test]
    fn test_illegal_idle_to_executing_blocked() {
        let mut fsm = AgencyFSM::new();
        // Jump from IDLE directly to EXECUTING — MUST be blocked
        let result = fsm.apply_transition(State::Executing);
        assert_eq!(result, State::Error);
        assert_eq!(fsm.rejection_log().len(), 1);
        assert!(
            fsm.rejection_log()[0].contains("Illegal transition"),
            "Log should mention illegal transition, got: {}",
            fsm.rejection_log()[0]
        );
    }

    #[test]
    fn test_illegal_error_to_executing_blocked() {
        let mut fsm = AgencyFSM::new();
        fsm.apply_transition(State::Executing); // IDLE → ERROR
        assert_eq!(fsm.state(), State::Error);
        // From ERROR, can only go to IDLE
        let result = fsm.apply_transition(State::Executing);
        assert_eq!(result, State::Error);
        assert!(fsm.rejection_log().len() >= 2);
    }

    // ── Forbidden command tests ────────────────────────────────

    #[test]
    fn test_forbidden_rm_rf() {
        assert!(is_forbidden_command("rm -rf /tmp/test"));
        assert!(is_forbidden_command("rm -r /var/log"));
        assert!(is_forbidden_command("sudo rm -rf /"));
    }

    #[test]
    fn test_forbidden_sudo() {
        assert!(is_forbidden_command("sudo systemctl stop"));
        assert!(is_forbidden_command("su root"));
    }

    #[test]
    fn test_allowed_commands() {
        assert!(!is_forbidden_command("ls -la"));
        assert!(!is_forbidden_command("echo hello"));
        assert!(!is_forbidden_command("df -h"));
        assert!(!is_forbidden_command("uptime"));
        assert!(!is_forbidden_command("brightnessctl set +10%"));
        assert!(!is_forbidden_command("notify-send 'hello' 'world'"));
    }

    #[test]
    fn test_forbidden_mkfs() {
        assert!(is_forbidden_command("mkfs.ext4 /dev/sda1"));
        assert!(is_forbidden_command("mkswap /dev/sda2"));
    }

    #[test]
    fn test_forbidden_dd() {
        assert!(is_forbidden_command("dd if=/dev/zero of=/dev/sda"));
    }

    // ── Full ActionRequest processing tests ─────────────────────

    #[test]
    fn test_process_valid_request() {
        let mut fsm = AgencyFSM::new();

        // Step 1: RESEARCHING
        let req = ActionRequest {
            transition: "RESEARCHING".into(),
            tool: "rag".into(),
            cmd: "search memories".into(),
            reason: "user asked a question".into(),
        };
        let resp = fsm.process_request(&req);
        assert!(resp.allowed);
        assert_eq!(resp.new_state, State::Researching);

        // Step 2: PLANNING
        let req = ActionRequest {
            transition: "PLANNING".into(),
            tool: "llm".into(),
            cmd: "generate response".into(),
            reason: "RAG results ready".into(),
        };
        let resp = fsm.process_request(&req);
        assert!(resp.allowed);
        assert_eq!(resp.new_state, State::Planning);

        // Step 3: EXECUTING with safe command
        let req = ActionRequest {
            transition: "EXECUTING".into(),
            tool: "bash".into(),
            cmd: "notify-send 'Bubby' 'Hello!'".into(),
            reason: "send greeting notification".into(),
        };
        let resp = fsm.process_request(&req);
        assert!(resp.allowed, "Safe command should be allowed, got: {}", resp.message);
        assert_eq!(resp.new_state, State::Executing);
    }

    #[test]
    fn test_process_destructive_command_blocked() {
        let mut fsm = AgencyFSM::new();
        // Navigate to PLANNING first
        fsm.apply_transition(State::Researching);
        fsm.apply_transition(State::Planning);

        // LLM tries to execute rm -rf
        let req = ActionRequest {
            transition: "EXECUTING".into(),
            tool: "bash".into(),
            cmd: "rm -rf /tmp/test".into(),
            reason: "LLM hallucinated cleanup".into(),
        };
        let resp = fsm.process_request(&req);

        assert!(!resp.allowed, "Destructive command MUST be blocked");
        assert_eq!(resp.new_state, State::Error, "Should enter ERROR state");
        assert!(
            resp.message.contains("PERMISSION DENIED"),
            "Message should mention permission denied, got: {}",
            resp.message
        );
    }

    #[test]
    fn test_process_idle_to_executing_blocked() {
        let mut fsm = AgencyFSM::new();
        // Try to jump from IDLE directly to EXECUTING
        let req = ActionRequest {
            transition: "EXECUTING".into(),
            tool: "bash".into(),
            cmd: "echo hello".into(),
            reason: "trying to skip planning".into(),
        };
        let resp = fsm.process_request(&req);
        assert!(!resp.allowed, "IDLE → EXECUTING must be blocked");
        assert_eq!(resp.new_state, State::Error);
        assert!(
            resp.message.contains("FSM BLOCKED"),
            "Should mention FSM blocked, got: {}",
            resp.message
        );
    }

    #[test]
    fn test_complete_execution() {
        let mut fsm = AgencyFSM::new();
        fsm.apply_transition(State::Researching);
        fsm.apply_transition(State::Planning);
        fsm.apply_transition(State::Executing);

        let resp = fsm.complete_execution("Command output: success".into());
        assert!(resp.allowed);
        assert_eq!(resp.new_state, State::Idle);
        assert_eq!(resp.command_output, Some("Command output: success".into()));
    }

    #[test]
    fn test_sanitize_command() {
        // "ls; rm -rf /" → semicolon replaced with "", "ls rm -rf /"
        assert_eq!(sanitize_command("ls; rm -rf /"), "ls rm -rf /");
        // "echo hello && evil" → "&&" replaced with "", "echo hello  evil" (double space)
        assert_eq!(sanitize_command("echo hello && evil"), "echo hello  evil");
        assert_eq!(sanitize_command("uptime"), "uptime");
    }
}