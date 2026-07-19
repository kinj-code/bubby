//! mTLS Mobile Bridge — mutual TLS with certificate pinning.
//!
//! Generates self-signed Ed25519 server + client certs via rcgen.
//! The server requires client authentication; only clients presenting
//! the exact pinned client certificate complete the TLS handshake.
//!
//! Dependencies: rcgen 0.13, rustls 0.23, rustls-pemfile 2

use rcgen::{CertificateParams, KeyPair};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use rustls::server::WebPkiClientVerifier;
use rustls::{ClientConfig, RootCertStore, ServerConfig};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::Arc;

// ── Certificate bundle ────────────────────────────────────────────

#[derive(Clone)]
pub struct CertBundle {
    pub server_cert_pem: Vec<u8>,
    pub server_key_pem: Vec<u8>,
    pub client_cert_pem: Vec<u8>,
    pub client_key_pem: Vec<u8>,
}

impl CertBundle {
    pub fn load_or_generate(dir: &Path) -> Result<Self, String> {
        let (sc, sk) = (dir.join("server.crt"), dir.join("server.key"));
        let (cc, ck) = (dir.join("client.crt"), dir.join("client.key"));
        if sc.exists() && cc.exists() {
            return Ok(Self {
                server_cert_pem: std::fs::read(&sc).map_err(|e| e.to_string())?,
                server_key_pem: std::fs::read(&sk).map_err(|e| e.to_string())?,
                client_cert_pem: std::fs::read(&cc).map_err(|e| e.to_string())?,
                client_key_pem: std::fs::read(&ck).map_err(|e| e.to_string())?,
            });
        }
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let b = Self::generate()?;
        let write = |p: &Path, d: &[u8]| -> Result<(), String> {
            std::fs::write(p, d).map_err(|e| e.to_string())?;
            #[cfg(unix)] {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o600)).ok();
            }
            Ok(())
        };
        write(&sc, &b.server_cert_pem)?; write(&sk, &b.server_key_pem)?;
        write(&cc, &b.client_cert_pem)?; write(&ck, &b.client_key_pem)?;
        Ok(b)
    }

    pub fn generate() -> Result<Self, String> {
        let s = Self::gen("Bubby Server", &["bubby.local".into(), "localhost".into()])?;
        let c = Self::gen("Bubby Client", &["bubby-client".into()])?;
        Ok(Self { server_cert_pem: s.0, server_key_pem: s.1, client_cert_pem: c.0, client_key_pem: c.1 })
    }

    fn gen(cn: &str, sans: &[String]) -> Result<(Vec<u8>, Vec<u8>), String> {
        let key = KeyPair::generate_for(&rcgen::PKCS_ED25519).map_err(|e| e.to_string())?;
        let key_pem: Vec<u8> = key.serialize_pem().into_bytes();
        let mut params = CertificateParams::new(sans.to_vec()).map_err(|e| e.to_string())?;
        params.distinguished_name.push(rcgen::DnType::CommonName, cn);
        // `self_signed` uses `key` as BOTH subject key and signer.
        // We serialized the key before signing, so we own the PEM.
        let cert = params.self_signed(&key).map_err(|e| e.to_string())?;
        Ok((cert.pem().into_bytes(), key_pem))
    }
}

// ── PEM helpers ───────────────────────────────────────────────────

fn parse_certs(pem: &[u8]) -> Result<Vec<CertificateDer<'static>>, String> {
    rustls_pemfile::certs(&mut pem.as_ref())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

fn parse_key(pem: &[u8]) -> Result<PrivateKeyDer<'static>, String> {
    rustls_pemfile::private_key(&mut pem.as_ref())
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "no private key found".into())
}

// ── mTLS Server ───────────────────────────────────────────────────

pub struct MtlsServer {
    listener: TcpListener,
    cfg: Arc<ServerConfig>,
}

impl MtlsServer {
    pub fn bind(addr: &str, b: &CertBundle) -> Result<Self, String> {
        let listener = TcpListener::bind(addr).map_err(|e| e.to_string())?;

        // Pin the client certificate: add it as the only trusted root
        let mut root = RootCertStore::empty();
        for c in parse_certs(&b.client_cert_pem)? {
            root.add(c).map_err(|e| e.to_string())?;
        }
        let verifier = WebPkiClientVerifier::builder(Arc::new(root))
            .build()
            .map_err(|e| e.to_string())?;

        let certs = parse_certs(&b.server_cert_pem)?;
        let key = parse_key(&b.server_key_pem)?;
        let cfg = ServerConfig::builder()
            .with_client_cert_verifier(verifier)
            .with_single_cert(certs, key)
            .map_err(|e| e.to_string())?;

        Ok(Self { listener, cfg: Arc::new(cfg) })
    }

    pub fn accept(
        &self,
    ) -> Result<rustls::StreamOwned<rustls::ServerConnection, TcpStream>, String> {
        let (tcp, _) = self.listener.accept().map_err(|e| e.to_string())?;
        let conn = rustls::ServerConnection::new(Arc::clone(&self.cfg))
            .map_err(|e| e.to_string())?;
        Ok(rustls::StreamOwned::new(conn, tcp))
    }
}

// ── mTLS Client ───────────────────────────────────────────────────

pub struct MtlsClient {
    cfg: Arc<ClientConfig>,
}

impl MtlsClient {
    pub fn new(b: &CertBundle) -> Result<Self, String> {
        let mut root = RootCertStore::empty();
        for c in parse_certs(&b.server_cert_pem)? {
            root.add(c).map_err(|e| e.to_string())?;
        }
        let cc = parse_certs(&b.client_cert_pem)?;
        let ck = parse_key(&b.client_key_pem)?;
        let cfg = ClientConfig::builder()
            .with_root_certificates(root)
            .with_client_auth_cert(cc, ck)
            .map_err(|e| e.to_string())?;
        Ok(Self { cfg: Arc::new(cfg) })
    }

    pub fn connect(
        &self,
        addr: &str,
    ) -> Result<rustls::StreamOwned<rustls::ClientConnection, TcpStream>, String> {
        let sn = ServerName::try_from("localhost").map_err(|e| e.to_string())?;
        let tcp = TcpStream::connect(addr).map_err(|e| e.to_string())?;
        let conn = rustls::ClientConnection::new(Arc::clone(&self.cfg), sn)
            .map_err(|e| e.to_string())?;
        Ok(rustls::StreamOwned::new(conn, tcp))
    }
}

// ── Tests: 5-point intrusion verification ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::thread;
    use std::time::Duration;

    /// Start an echo mTLS server on a random port, return its address.
    fn start_echo(b: &CertBundle) -> String {
        let b = b.clone();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
        // The server binds directly on this single-use listener.
        // We don't need MtlsServer::bind — we already own the socket.
        let b = b.clone();
        let server_addr = addr.clone();
        thread::spawn(move || {
            // Build the TLS config manually since we already have a listener
            let mut root = RootCertStore::empty();
            for c in parse_certs(&b.client_cert_pem).unwrap() { root.add(c).unwrap(); }
            let verifier = WebPkiClientVerifier::builder(Arc::new(root)).build().unwrap();
            let sc = parse_certs(&b.server_cert_pem).unwrap();
            let sk = parse_key(&b.server_key_pem).unwrap();
            let cfg = Arc::new(ServerConfig::builder().with_client_cert_verifier(verifier).with_single_cert(sc, sk).unwrap());

            for stream in listener.incoming() {
                let tcp = match stream { Ok(s) => s, Err(_) => break };
                let conn = rustls::ServerConnection::new(Arc::clone(&cfg)).unwrap();
                let mut s = rustls::StreamOwned::new(conn, tcp);
                let mut buf = [0u8; 256];
                if let Ok(n) = s.read(&mut buf) {
                    if std::str::from_utf8(&buf[..n]).unwrap_or("").trim() == "ping" {
                        let _ = s.write_all(b"pong");
                        let _ = s.flush();
                    }
                }
            }
        });
        thread::sleep(Duration::from_millis(50));
        server_addr
    }

    fn test_bundle() -> CertBundle {
        CertBundle::generate().unwrap()
    }

    // ── Test 1: Certificate generation ─────────────────────────

    #[test]
    fn test_cert_generation() {
        let b = test_bundle();
        assert!(String::from_utf8_lossy(&b.server_cert_pem).contains("CERTIFICATE"));
        assert!(String::from_utf8_lossy(&b.server_key_pem).contains("PRIVATE KEY"));
        assert!(String::from_utf8_lossy(&b.client_cert_pem).contains("CERTIFICATE"));
        assert!(String::from_utf8_lossy(&b.client_key_pem).contains("PRIVATE KEY"));
    }

    // ── Test 2: Happy path — pinned client connects ────────────

    #[test]
    fn test_mtls_happy_path() {
        let b = test_bundle();
        let addr = start_echo(&b);

        let client = MtlsClient::new(&b).unwrap();
        let mut s = client.connect(&addr).unwrap();

        s.write_all(b"ping").unwrap();
        s.flush().unwrap();

        let mut buf = [0u8; 16];
        let n = s.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"pong");
    }

    // ── Test 3: Plain TCP without TLS → rejected ───────────────

    #[test]
    fn test_intrusion_no_tls_rejected() {
        let addr = start_echo(&test_bundle());

        let mut sock = TcpStream::connect(&addr).unwrap();
        sock.write_all(b"ping\n").unwrap();

        let mut buf = [0u8; 32];
        match sock.read(&mut buf) {
            Ok(0) | Err(_) => {} // connection closed or reset
            Ok(n) => {
                // May receive a TLS Alert record (server rejects plaintext).
                // Verify it's NOT the expected "pong" response.
                let data = String::from_utf8_lossy(&buf[..n]);
                assert!(!data.contains("pong"),
                    "non-TLS client should never receive 'pong', got: {}", data);
            }
        }
    }

    // ── Test 4: TLS without client cert → rejected ─────────────

    #[test]
    fn test_intrusion_no_client_cert_rejected() {
        let b = test_bundle();
        let addr = start_echo(&b);

        // Build a TLS client that trusts the server but has no client cert
        let mut root = RootCertStore::empty();
        for c in parse_certs(&b.server_cert_pem).unwrap() {
            root.add(c).unwrap();
        }
        let cfg = ClientConfig::builder()
            .with_root_certificates(root)
            .with_no_client_auth();

        let sn = ServerName::try_from("localhost").unwrap();
        let tcp = TcpStream::connect(&addr).unwrap();
        let conn = rustls::ClientConnection::new(Arc::new(cfg), sn).unwrap();
        let mut s = rustls::StreamOwned::new(conn, tcp);

        // Handshake happens lazily on first I/O.
        // write_all may buffer locally; flush forces the handshake.
        let _ = s.write_all(b"ping");
        let result = s.flush();
        // Also try a read — either should surface the handshake error
        let read_err = s.read(&mut [0u8; 16]).is_err();
        assert!(result.is_err() || read_err,
            "TLS without client cert must be rejected on write/read");
    }

    // ── Test 5: Wrong client cert → rejected ───────────────────

    #[test]
    fn test_intrusion_wrong_cert_rejected() {
        let addr = start_echo(&test_bundle());

        // Generate an entirely different cert bundle
        let rogue = test_bundle();
        let mut s = MtlsClient::new(&rogue).unwrap().connect(&addr).unwrap();

        // First I/O must fail — server rejects the rogue cert
        assert!(s.write_all(b"ping").is_err(),
            "rogue client cert must be rejected");
    }
}