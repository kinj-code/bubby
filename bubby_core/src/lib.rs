//! Bubby Core — Rust kernel for the autonomous desktop companion.
//!
//! Exports:
//! - `memory`: Persistent memory store (SQLite + WAL)
//! - `vector`: HNSW approximate nearest neighbor search
//! - `python`: PyO3 bindings for import from Python

pub mod memory;
pub mod vector;
pub mod ipc;
pub mod security;
pub mod mtls;
pub mod agency;

/// Core version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
