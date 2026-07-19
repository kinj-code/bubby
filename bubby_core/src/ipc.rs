//! IPC Bridge — high-speed binary protocol over TCP localhost.
//!
//! Uses **MessagePack** (rmp-serde in Rust, msgpack in Python) for
//! cross-language binary serialisation. No JSON overhead.
//!
//! Protocol:
//!   [4 bytes LE: payload length] [MessagePack-encoded RequestV1 / ResponseV1]
//!
//! Supported commands mirror `memory::MemoryStore`:
//!   add, add_with_embedding, search_by_text, search_by_embedding,
//!   batch_search_by_embedding, get_recent, get_by_importance,
//!   count, clear, stats, get, ping

use crate::memory::{MemoryRecord, SearchResult};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

// ── Wire protocol types ───────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct RequestV1 {
    pub id: u64,
    pub method: String,
    /// Already-encoded MessagePack bytes for the method-specific payload.
    /// Python will pack the inner dict and put the raw bytes here.
    /// The Rust server deserialises it with rmp_serde.
    pub payload: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseV1 {
    pub id: u64,
    pub status: String,
    /// MessagePack-encoded payload (type depends on method).
    pub payload: Vec<u8>,
    pub error: String,
}

// ── Per-method payloads ───────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct AddParams {
    pub text: String,
    pub timestamp: f64,
    pub importance: f64,
    pub metadata: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddWithEmbeddingParams {
    pub text: String,
    pub timestamp: f64,
    pub importance: f64,
    pub metadata: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchByTextParams {
    pub query: String,
    pub k: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchByEmbeddingParams {
    pub embedding: Vec<f32>,
    pub k: usize,
    pub min_score: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchSearchParams {
    pub queries: Vec<SearchByEmbeddingParams>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetByImportanceParams {
    pub n: usize,
    pub min_importance: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetParams {
    pub id: i64,
}

// ── Helper: MessagePack encode / decode ───────────────────────────

fn mp_serialize<T: Serialize>(value: &T) -> Vec<u8> {
    rmp_serde::to_vec(value).unwrap_or_default()
}

fn mp_deserialize<'a, T: Deserialize<'a>>(data: &'a [u8]) -> Result<T, String> {
    rmp_serde::from_slice(data).map_err(|e| e.to_string())
}

// ── TCP server ────────────────────────────────────────────────────

pub fn run_server(store: Arc<crate::memory::MemoryStore>, addr: &str) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    eprintln!("[ipc] Listening on {}", addr);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let store = Arc::clone(&store);
                std::thread::spawn(move || {
                    handle_client(store, stream);
                });
            }
            Err(e) => {
                eprintln!("[ipc] accept error: {}", e);
            }
        }
    }
    Ok(())
}

fn handle_client(store: Arc<crate::memory::MemoryStore>, mut stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    let mut len_buf = [0u8; 4];

    loop {
        if stream.read_exact(&mut len_buf).is_err() {
            break;
        }
        let payload_len = u32::from_le_bytes(len_buf) as usize;
        let mut payload = vec![0u8; payload_len];
        if stream.read_exact(&mut payload).is_err() {
            break;
        }

        let req: RequestV1 = match mp_deserialize(&payload) {
            Ok(r) => r,
            Err(e) => {
                let resp = ResponseV1 {
                    id: 0,
                    status: "error".into(),
                    payload: vec![],
                    error: format!("deserialise failed: {}", e),
                };
                let _ = send_response(&mut stream, &resp);
                continue;
            }
        };

        let resp = dispatch(&store, &req);
        if send_response(&mut stream, &resp).is_err() {
            break;
        }
    }
}

fn send_response(stream: &mut TcpStream, resp: &ResponseV1) -> std::io::Result<()> {
    let data = mp_serialize(resp);
    let len = (data.len() as u32).to_le_bytes();
    stream.write_all(&len)?;
    stream.write_all(&data)?;
    stream.flush()?;
    Ok(())
}

fn dispatch(store: &crate::memory::MemoryStore, req: &RequestV1) -> ResponseV1 {
    let id = req.id;
    match req.method.as_str() {
        "ping" => ok(id, &serde_json::json!({"version": crate::VERSION})),

        "add" => match mp_deserialize::<AddParams>(&req.payload) {
            Ok(p) => match store.add(&p.text, p.timestamp, p.importance, &p.metadata) {
                Ok(rid) => ok(id, &serde_json::json!({"id": rid})),
                Err(e) => err(id, &e.to_string()),
            },
            Err(e) => err(id, &format!("bad params: {}", e)),
        },

        "add_with_embedding" => {
            match mp_deserialize::<AddWithEmbeddingParams>(&req.payload) {
                Ok(p) => match store.add_with_embedding(
                    &p.text, p.timestamp, p.importance, &p.metadata, &p.embedding,
                ) {
                    Ok(rid) => ok(id, &serde_json::json!({"id": rid})),
                    Err(e) => err(id, &e.to_string()),
                },
                Err(e) => err(id, &format!("bad params: {}", e)),
            }
        }

        "search_by_text" => {
            match mp_deserialize::<SearchByTextParams>(&req.payload) {
                Ok(p) => match store.search_by_text(&p.query, p.k) {
                    Ok(records) => ok(id, &records),
                    Err(e) => err(id, &e.to_string()),
                },
                Err(e) => err(id, &format!("bad params: {}", e)),
            }
        }

        "search_by_embedding" => {
            match mp_deserialize::<SearchByEmbeddingParams>(&req.payload) {
                Ok(p) => match store.search_by_embedding(&p.embedding, p.k, p.min_score) {
                    Ok(results) => ok(id, &results),
                    Err(e) => err(id, &e.to_string()),
                },
                Err(e) => err(id, &format!("bad params: {}", e)),
            }
        }

        "batch_search_by_embedding" => {
            match mp_deserialize::<BatchSearchParams>(&req.payload) {
                Ok(p) => {
                    let mut batch: Vec<serde_json::Value> = Vec::with_capacity(p.queries.len());
                    for q in &p.queries {
                        match store.search_by_embedding(&q.embedding, q.k, q.min_score) {
                            Ok(results) => {
                                batch.push(serde_json::json!(results));
                            }
                            Err(e) => {
                                batch.push(serde_json::json!({"error": e.to_string()}));
                            }
                        }
                    }
                    ok(id, &batch)
                }
                Err(e) => err(id, &format!("bad params: {}", e)),
            }
        }

        "search_by_metadata" => {
            // Accept a generic JSON object with key/value/k
            match serde_json::from_slice::<serde_json::Value>(&req.payload) {
                Ok(v) => {
                    let key = v["key"].as_str().unwrap_or("");
                    let value = v["value"].as_str().unwrap_or("");
                    let k = v["k"].as_u64().unwrap_or(5) as usize;
                    match store.search_by_metadata(key, value, k) {
                        Ok(records) => ok(id, &records),
                        Err(e) => err(id, &e.to_string()),
                    }
                }
                Err(e) => err(id, &format!("bad params: {}", e)),
            }
        }

        "get_recent" => {
            let n: usize = mp_deserialize::<u64>(&req.payload).unwrap_or(10) as usize;
            match store.get_recent(n) {
                Ok(records) => ok(id, &records),
                Err(e) => err(id, &e.to_string()),
            }
        }

        "get_by_importance" => {
            match mp_deserialize::<GetByImportanceParams>(&req.payload) {
                Ok(p) => match store.get_by_importance(p.n, p.min_importance) {
                    Ok(records) => ok(id, &records),
                    Err(e) => err(id, &e.to_string()),
                },
                Err(e) => err(id, &format!("bad params: {}", e)),
            }
        }

        "get" => match mp_deserialize::<GetParams>(&req.payload) {
            Ok(p) => match store.get(p.id) {
                Ok(record) => ok(id, &record),
                Err(e) => err(id, &e.to_string()),
            },
            Err(e) => err(id, &format!("bad params: {}", e)),
        },

        "update_importance" => {
            match serde_json::from_slice::<serde_json::Value>(&req.payload) {
                Ok(v) => {
                    let mid = v["id"].as_i64().unwrap_or(-1);
                    let imp = v["importance"].as_f64().unwrap_or(0.5);
                    match store.update_importance(mid, imp) {
                        Ok(changed) => ok(id, &serde_json::json!({"ok": changed})),
                        Err(e) => err(id, &e.to_string()),
                    }
                }
                Err(e) => err(id, &format!("bad params: {}", e)),
            }
        }

        "delete" => match mp_deserialize::<GetParams>(&req.payload) {
            Ok(p) => match store.delete(p.id) {
                Ok(found) => ok(id, &serde_json::json!({"ok": found})),
                Err(e) => err(id, &e.to_string()),
            },
            Err(e) => err(id, &format!("bad params: {}", e)),
        },

        "count" => ok(id, &serde_json::json!({"count": store.count()})),

        "stats" => ok(id, &store.stats()),

        "clear" => match store.clear() {
            Ok(()) => ok(id, &serde_json::json!({"ok": true})),
            Err(e) => err(id, &e.to_string()),
        },

        _ => err(id, &format!("unknown method: {}", req.method)),
    }
}

fn ok<T: Serialize>(id: u64, value: &T) -> ResponseV1 {
    // Serialise inner value with rmp_serde, or fall back to serde_json
    let payload = rmp_serde::to_vec(value).unwrap_or_else(|_| {
        serde_json::to_vec(value).unwrap_or_default()
    });
    ResponseV1 {
        id,
        status: "ok".into(),
        payload,
        error: String::new(),
    }
}

fn err(id: u64, msg: &str) -> ResponseV1 {
    ResponseV1 {
        id,
        status: "error".into(),
        payload: vec![],
        error: msg.to_string(),
    }
}

/// Send one request over an already-open stream, return the response.
pub fn send_one(stream: &mut TcpStream, req: &RequestV1) -> std::io::Result<ResponseV1> {
    let data = mp_serialize(req);
    let len = (data.len() as u32).to_le_bytes();
    stream.write_all(&len)?;
    stream.write_all(&data)?;
    stream.flush()?;

    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let payload_len = u32::from_le_bytes(len_buf) as usize;
    let mut payload = vec![0u8; payload_len];
    stream.read_exact(&mut payload)?;

    mp_deserialize(&payload).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, format!("decode: {}", e))
    })
}

// ── tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryStore;
    use std::thread;
    use std::time::Duration;

    fn start_test_server() -> (String, Arc<MemoryStore>) {
        let store = Arc::new(MemoryStore::open_in_memory().unwrap());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
        let store_clone = Arc::clone(&store);
        thread::spawn(move || {
            for stream in listener.incoming() {
                if let Ok(stream) = stream {
                    handle_client(Arc::clone(&store_clone), stream);
                }
            }
        });
        thread::sleep(Duration::from_millis(10));
        (addr, store)
    }

    #[test]
    fn test_ipc_ping() {
        let (addr, _) = start_test_server();
        let mut stream = TcpStream::connect(&addr).unwrap();
        stream.set_nodelay(true).unwrap();

        let req = RequestV1 { id: 1, method: "ping".into(), payload: vec![] };
        let resp = send_one(&mut stream, &req).unwrap();
        assert_eq!(resp.status, "ok");
    }

    #[test]
    fn test_ipc_add_and_search() {
        let (addr, _) = start_test_server();
        let mut stream = TcpStream::connect(&addr).unwrap();
        stream.set_nodelay(true).unwrap();

        for i in 0..5 {
            let inner = mp_serialize(&AddParams {
                text: format!("Record {}", i),
                timestamp: i as f64,
                importance: 0.5 + i as f64 * 0.1,
                metadata: "{}".into(),
            });
            let req = RequestV1 { id: i, method: "add".into(), payload: inner };
            let resp = send_one(&mut stream, &req).unwrap();
            assert_eq!(resp.status, "ok");
        }

        let inner = mp_serialize(&SearchByTextParams { query: "Record".into(), k: 5 });
        let req = RequestV1 { id: 100, method: "search_by_text".into(), payload: inner };
        let resp = send_one(&mut stream, &req).unwrap();
        assert_eq!(resp.status, "ok");
        let records: Vec<MemoryRecord> = mp_deserialize(&resp.payload).unwrap();
        assert_eq!(records.len(), 5);
    }

    #[test]
    fn test_ipc_latency_1000_calls() {
        let (addr, _) = start_test_server();

        // Populate
        {
            let mut stream = TcpStream::connect(&addr).unwrap();
            stream.set_nodelay(true).unwrap();
            for i in 0..10 {
                let inner = mp_serialize(&AddWithEmbeddingParams {
                    text: format!("item_{}", i),
                    timestamp: i as f64,
                    importance: 0.5,
                    metadata: "{}".into(),
                    embedding: (0..4).map(|_| rand::random::<f32>()).collect(),
                });
                let req = RequestV1 { id: i, method: "add_with_embedding".into(), payload: inner };
                send_one(&mut stream, &req).unwrap();
            }
        }

        let batch = BatchSearchParams {
            queries: (0..1000)
                .map(|_| SearchByEmbeddingParams {
                    embedding: vec![1.0, 0.0, 0.0, 0.0],
                    k: 3,
                    min_score: 0.0,
                })
                .collect(),
        };

        let mut stream = TcpStream::connect(&addr).unwrap();
        stream.set_nodelay(true).unwrap();
        let req = RequestV1 {
            id: 42,
            method: "batch_search_by_embedding".into(),
            payload: mp_serialize(&batch),
        };

        let start = std::time::Instant::now();
        let resp = send_one(&mut stream, &req).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(resp.status, "ok", "batch failed: {}", resp.error);
        let results: Vec<serde_json::Value> = mp_deserialize(&resp.payload).unwrap();
        assert_eq!(results.len(), 1000);

        eprintln!(
            "IPC (msgpack) latency: 1000 batch calls in {:.3}ms ({:.3}µs/call)",
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() / 1000.0 * 1_000_000.0
        );
        assert!(elapsed.as_millis() < 50, "1000 calls took {}ms", elapsed.as_millis());
    }
}