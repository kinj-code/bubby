//! Bubby Core — Rust kernel for the autonomous desktop companion.
//!
//! Exports:
//! - `memory`: Persistent memory store (SQLite + WAL)
//! - `vector`: HNSW approximate nearest neighbor search
//! - `python`: PyO3 bindings for import from Python

pub mod memory;
pub mod vector;

// Python bindings in a future phase (requires proper PyO3 setup)
// #[cfg(feature = "python-bindings")]
// pub mod python;

/// Core version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
