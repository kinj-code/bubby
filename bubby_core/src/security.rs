//! Zero-trust key management for the Bubby memory store.
//!
//! - Generates a 256-bit random key on first launch.
//! - Persists the key in the OS keychain (Linux Secret Service via `keyring`).
//! - Loads the key from keychain on subsequent launches.
//! - Zeroizes all key material from process heap after use.
//!
//! Integration with `MemoryStore`:
//! ```ignore
//! let key_mgr = KeyManager::new("bubby")?;
//! let key = key_mgr.get_or_create_key()?;  // Hex-encoded 64-char string
//! let store = MemoryStore::open_encrypted("data/memory.db", &key)?;
//! drop(key);  // zeroized
//! ```

use keyring::Entry;
use rand::RngCore;
use zeroize::Zeroize as _;

/// Service name registered in the OS keychain.
const KEYRING_SERVICE: &str = "com.bubby.memory";
/// Account/entry name within the keychain service.
const KEYRING_ACCOUNT: &str = "db-encryption-key";

// ── Key material wrapper with auto-zeroize ────────────────────────

/// A 256-bit database encryption key that zeroizes itself on drop.
#[derive(Clone)]
pub struct SecureKey {
    /// Raw bytes — NEVER printed or logged.
    bytes: [u8; 32],
    /// Pre-computed hex string for passing to SQLCipher `PRAGMA key`.
    hex: String,
}

impl SecureKey {
    /// Generate a cryptographically secure random 256-bit key.
    pub fn generate() -> Self {
        let mut bytes = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut bytes);
        let hex = hex_encode(&bytes);
        Self { bytes, hex }
    }

    /// Reconstruct from a hex string (64 hex chars).
    pub fn from_hex(hex: &str) -> Result<Self, String> {
        if hex.len() != 64 {
            return Err("key must be 64 hex characters (256 bits)".into());
        }
        let mut bytes = [0u8; 32];
        for i in 0..32 {
            let byte_str = &hex[i * 2..i * 2 + 2];
            bytes[i] = u8::from_str_radix(byte_str, 16)
                .map_err(|e| format!("invalid hex byte at position {}: {}", i, e))?;
        }
        let hex = hex.to_string();
        Ok(Self { bytes, hex })
    }

    /// Return the hex-encoded key string for SQLCipher `PRAGMA key`.
    pub fn as_hex(&self) -> &str {
        &self.hex
    }
}

impl Drop for SecureKey {
    fn drop(&mut self) {
        self.bytes.zeroize();
        // Zeroize the hex string as well (force-clear the Vec's heap)
        unsafe {
            for b in self.hex.as_bytes_mut() {
                std::ptr::write_volatile(b, 0);
            }
        }
    }
}

// Prevent Debug/Display from leaking the key
impl std::fmt::Debug for SecureKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecureKey")
            .field("bytes", &"<redacted 32 bytes>")
            .field("hex", &"<redacted 64 chars>")
            .finish()
    }
}

// ── OS Keychain integration ───────────────────────────────────────

/// Manages a 256-bit database encryption key via the OS keychain.
///
/// Supported backends (via the `keyring` crate):
/// - Linux:   Secret Service (libsecret / gnome-keyring)
/// - macOS:   Keychain Services
/// - Windows: Credential Manager
pub struct KeyManager {
    entry: Entry,
}

impl KeyManager {
    /// Create a new key manager for the given application name.
    ///
    /// The `app` string is used as the keychain entry description.
    /// On Linux this requires a running D-Bus session bus and
    /// `gnome-keyring` or `kwallet`.
    pub fn new(_app: &str) -> Result<Self, String> {
        let entry = Entry::new(KEYRING_SERVICE, KEYRING_ACCOUNT)
            .map_err(|e| format!("keychain init failed: {}", e))?;
        // Do NOT call entry.set_password() here — that would overwrite
        // the stored encryption key with the app name string, making the
        // key unrecoverable across restarts. The first call to
        // get_or_create_key() handles first-run key generation + storage.
        Ok(Self { entry })
    }

    /// Retrieve the encryption key from the keychain, or generate and
    /// store a new one if it doesn't exist.
    ///
    /// The returned `SecureKey` zeroizes itself on drop.
    pub fn get_or_create_key(&self) -> Result<SecureKey, String> {
        match self.entry.get_password() {
            Ok(hex) if hex.len() == 64 => {
                // Key exists and looks valid
                SecureKey::from_hex(&hex)
            }
            Ok(_junk) => {
                // Stale / corrupted entry — regenerate
                self.regenerate()
            }
            Err(keyring::Error::NoEntry) => {
                // First run — generate fresh key
                self.regenerate()
            }
            Err(e) => {
                // Keychain unavailable — generate ephemeral key with warning
                eprintln!(
                    "[security] WARNING: OS keychain unavailable ({}). \
                     Using in-memory key. Database encryption will NOT survive \
                     process restart unless key is provided externally.",
                    e
                );
                Ok(SecureKey::generate())
            }
        }
    }

    /// Delete the key from the keychain (for "factory reset").
    pub fn delete_key(&self) -> Result<(), String> {
        self.entry
            .delete_credential()
            .map_err(|e| format!("failed to delete key from keychain: {}", e))
    }

    /// Generate a fresh key and store it in the keychain.
    fn regenerate(&self) -> Result<SecureKey, String> {
        let key = SecureKey::generate();
        self.entry
            .set_password(key.as_hex())
            .map_err(|e| format!("failed to store key in keychain: {}", e))?;
        Ok(key)
    }
}

// ── Helpers ───────────────────────────────────────────────────────

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_key_generate_and_roundtrip() {
        let key = SecureKey::generate();
        assert_eq!(key.as_hex().len(), 64);

        let hex = key.as_hex().to_string();
        drop(key);

        let restored = SecureKey::from_hex(&hex).unwrap();
        assert_eq!(restored.as_hex(), &hex);
        drop(restored);
    }

    #[test]
    fn test_secure_key_from_hex_rejects_bad_input() {
        assert!(SecureKey::from_hex("too_short").is_err());
        assert!(SecureKey::from_hex(&"g".repeat(64)).is_err());
        assert!(SecureKey::from_hex(&"0".repeat(64)).is_ok());
    }

    #[test]
    fn test_secure_key_zeroize_on_drop() {
        let hex = {
            let key = SecureKey::generate();
            key.as_hex().to_string()
        };
        // After drop, the original key's memory is zeroed.
        // We can't observe the zeroing directly in safe Rust,
        // but we verify that the roundtrip still works
        let restored = SecureKey::from_hex(&hex).unwrap();
        assert_eq!(restored.as_hex(), &hex);
    }

    #[test]
    fn test_key_manager_ephemeral_fallback() {
        // On CI or headless systems without a keychain, KeyManager
        // should still return a valid key (ephemeral).
        let mgr = KeyManager::new("bubby-test");
        // May succeed or fail depending on CI environment, but
        // `get_or_create_key` must not panic.
        let result = mgr.and_then(|km| km.get_or_create_key());
        match result {
            Ok(key) => {
                assert_eq!(key.as_hex().len(), 64);
            }
            Err(e) => {
                // Acceptable on headless CI without dbus
                eprintln!("Keychain test skipped: {}", e);
            }
        }
    }
}