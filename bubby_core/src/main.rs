//! Bubby Core — standalone binary entry point.
//!
//! Starts the IPC server (for Python↔Rust communication) and the
//! broadcast server (for UI state updates). The binary runs
//! independently; the Python UI connects via local TCP sockets.
//!
//! Usage:
//!   cargo run --release -- \
//!     --db data/memory.db \
//!     --ipc 127.0.0.1:9500 \
//!     --broadcast 127.0.0.1:9501 \
//!     --encrypt-key 0a1b2c...   (64-char hex, optional)

use std::path::PathBuf;
use std::sync::Arc;

use bubby_core::broadcast::{BroadcastServer, StateEvent};
use bubby_core::ipc;
use bubby_core::memory::MemoryStore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Parse command-line arguments ──────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut db_path = PathBuf::from("data/memory.db");
    let mut ipc_addr = "127.0.0.1:9500".to_string();
    let mut broadcast_addr = "127.0.0.1:9501".to_string();
    let mut hex_key: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--db" => { i += 1; db_path = PathBuf::from(&args[i]); }
            "--ipc" => { i += 1; ipc_addr = args[i].clone(); }
            "--broadcast" => { i += 1; broadcast_addr = args[i].clone(); }
            "--encrypt-key" => { i += 1; hex_key = Some(args[i].clone()); }
            _ => eprintln!("Unknown argument: {}", args[i]),
        }
        i += 1;
    }

    // ── Open memory store ────────────────────────────────────
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let store = if let Some(ref key) = hex_key {
        if key.len() != 64 {
            eprintln!("Encryption key must be 64 hex characters (256 bits)");
            std::process::exit(1);
        }
        MemoryStore::open_encrypted(&db_path, key)?
    } else {
        MemoryStore::open(&db_path)?
    };

    eprintln!("[core] Memory store opened: {}", db_path.display());
    eprintln!("[core] {} records loaded", store.count());

    let store = Arc::new(store);

    // ── Start broadcast server ───────────────────────────────
    let broadcast = Arc::new(BroadcastServer::bind(&broadcast_addr)?);
    eprintln!("[core] Broadcast server listening on {}", broadcast_addr);

    broadcast.broadcast(&StateEvent::state_change("IDLE", "Bubby core started"));

    // ── Start IPC server (blocks) ────────────────────────────
    eprintln!("[core] IPC server listening on {}", ipc_addr);
    eprintln!("[core] Bubby Core ready — waiting for connections...");

    // The IPC server runs in a loop; we also periodically accept
    // new broadcast clients.
    let store_ipc = Arc::clone(&store);
    let broadcast_clone = Arc::clone(&broadcast);

    // Spawn a thread for periodic broadcast client acceptance
    std::thread::spawn(move || {
        loop {
            broadcast_clone.accept_new_clients();
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    });

    // Run the IPC server on the main thread
    ipc::run_server(store_ipc, &ipc_addr)?;

    Ok(())
}