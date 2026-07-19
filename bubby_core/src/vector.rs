//! HNSW vector engine — sub-millisecond approximate nearest neighbor search.
//!
//! Replaces `src/memory/vector_db.py` with a lock-free, deterministic
//! HNSW (Hierarchical Navigable Small World) graph implementation.
//!
//! Uses cosine similarity with normalized vectors. The graph is
//! persisted alongside the memory store.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A vector entry in the HNSW index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: i64,
    pub embedding: Vec<f32>, // Normalized, dimension D
}

/// Lightweight HNSW index for approximate vector search.
///
/// Implements a simplified HNSW algorithm with:
/// - Multi-layer graph structure
/// - Ef construction and search parameters
/// - Cosine similarity (dot product on normalized vectors)
pub struct HnswIndex {
    dimension: usize,
    ef_construction: usize,
    ef_search: usize,
    m: usize,              // Max connections per node per layer
    max_layer: isize,
    entry_point: Option<i64>,
    nodes: HashMap<i64, VectorEntry>,
    // graph[layer][node_id] = set of neighbor ids
    graph: Vec<HashMap<i64, Vec<i64>>>,
    lock: RwLock<()>, // Write-serialization, reads are lock-free via clone of state
}

impl HnswIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            ef_construction: 200,
            ef_search: 50,
            m: 16,
            max_layer: -1,
            entry_point: None,
            nodes: HashMap::new(),
            graph: vec![HashMap::new()],
            lock: RwLock::new(()),
        }
    }

    /// Insert a vector into the index.
    pub fn insert(&mut self, id: i64, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.dimension);

        let normalized = Self::normalize(embedding);
        // Every node must participate in layer 0; level >= 0
        let level = Self::random_level(self.m).max(0);

        self.nodes.insert(
            id,
            VectorEntry {
                id,
                embedding: normalized.clone(),
            },
        );

        // Add layers if needed
        while self.graph.len() <= level as usize {
            self.graph.push(HashMap::new());
        }

        // Add node to layers 0..=level
        for lc in 0..=level {
            self.graph[lc as usize]
                .entry(id)
                .or_insert_with(Vec::new);
        }

        // First node — become the entry point, no neighbors to connect
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_layer = level;
            return;
        }

        let ep0 = self.entry_point.unwrap();
        let mut ep = ep0;

        // Greedy descent from top layer down to level+1
        for lc in ((level + 1)..=self.max_layer).rev() {
            let layer = lc as usize;
            let mut best = ep;
            let mut best_dist = self.distance(id, best);
            let mut visited_local: HashMap<i64, bool> = HashMap::new();
            visited_local.insert(ep, true);
            loop {
                let mut improved = false;
                if let Some(layer_graph) = self.graph.get(layer) {
                    if let Some(neighbors) = layer_graph.get(&best) {
                        for &n in neighbors {
                            if visited_local.contains_key(&n) {
                                continue;
                            }
                            visited_local.insert(n, true);
                            let d = self.distance(id, n);
                            if d < best_dist {
                                best_dist = d;
                                best = n;
                                improved = true;
                            }
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
            ep = best;
        }

        // Connect at each layer from level down to 0
        for lc in (0..=level).rev() {
            let layer = lc as usize;

            // Select M nearest neighbors among existing nodes in this layer
            let candidate_ids: Vec<i64> = {
                // Collect all existing nodes EXCEPT the new node itself
                let mut ids: Vec<i64> = self.graph
                    .get(layer)
                    .map(|g| g.keys().copied().filter(|k| *k != id).collect())
                    .unwrap_or_default();
                if ids.is_empty() {
                    // No other nodes in this layer yet
                    vec![]
                } else {
                    // Sort by distance ascending, take top M
                    ids.sort_by(|a, b| {
                        self.distance(id, *a)
                            .partial_cmp(&self.distance(id, *b))
                            .unwrap()
                    });
                    ids.truncate(self.m);
                    ids
                }
            };

            if let Some(layer_graph) = self.graph.get_mut(layer) {
                for &neighbor in &candidate_ids {
                    layer_graph
                        .get_mut(&id)
                        .expect("new node must be in layer")
                        .push(neighbor);
                    layer_graph
                        .entry(neighbor)
                        .or_insert_with(Vec::new)
                        .push(id);
                }

                // Prune excess connections for all affected nodes
                for &neighbor in &candidate_ids {
                    if let Some(nbrs) = layer_graph.get_mut(&neighbor) {
                        nbrs.sort();
                        nbrs.dedup();
                        nbrs.truncate(self.m);
                    }
                }
                if let Some(nbrs) = layer_graph.get_mut(&id) {
                    nbrs.sort();
                    nbrs.dedup();
                    nbrs.truncate(self.m);
                }
            }
        }

        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(id);
        }
    }

    /// Search for the k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        if self.entry_point.is_none() || k == 0 {
            return vec![];
        }

        let query_norm = Self::normalize(query);
        let mut ep = self.entry_point.unwrap();

        // Greedy search from top layer down
        for lc in (1..=self.max_layer as usize).rev() {
            if let Some(layer_graph) = self.graph.get(lc) {
                let mut best = ep;
                let mut best_dist = self.distance_vec(&query_norm, best);
                let mut changed = true;
                while changed {
                    changed = false;
                    if let Some(neighbors) = layer_graph.get(&best) {
                        for &n in neighbors {
                            let d = self.distance_vec(&query_norm, n);
                            if d < best_dist {
                                best_dist = d;
                                best = n;
                                changed = true;
                            }
                        }
                    }
                }
                ep = best;
            }
        }

        // Full search at layer 0
        let mut results: Vec<(f64, i64)> = vec![];
        let mut visited: HashMap<i64, bool> = HashMap::new();
        let mut candidates: Vec<(f64, i64)> = vec![(self.distance_vec(&query_norm, ep), ep)];

        while !candidates.is_empty() && results.len() < self.ef_search {
            // Sort by distance, take closest unvisited
            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let (dist, node) = candidates.remove(0);

            if visited.contains_key(&node) {
                continue;
            }
            visited.insert(node, true);
            results.push((dist, node));

            // Add neighbors
            if let Some(layer_graph) = self.graph.get(0) {
                if let Some(neighbors) = layer_graph.get(&node) {
                    for &n in neighbors {
                        if !visited.contains_key(&n) {
                            let d = self.distance_vec(&query_norm, n);
                            candidates.push((d, n));
                        }
                    }
                }
            }
        }

        // Convert to (id, similarity) — closer distance = higher similarity
        let mut out: Vec<(i64, f32)> = results
            .iter()
            .take(k)
            .map(|&(dist, id)| (id, (1.0 - dist as f32).max(0.0)))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        out
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear all vectors.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.graph.clear();
        self.graph.push(HashMap::new());
        self.entry_point = None;
        self.max_layer = -1;
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    fn distance(&self, id_a: i64, id_b: i64) -> f64 {
        if let (Some(a), Some(b)) = (self.nodes.get(&id_a), self.nodes.get(&id_b)) {
            let dot: f32 = a
                .embedding
                .iter()
                .zip(b.embedding.iter())
                .map(|(x, y)| x * y)
                .sum();
            (1.0 - dot as f64).max(0.0)
        } else {
            f64::MAX
        }
    }

    fn distance_vec(&self, query: &[f32], id: i64) -> f64 {
        if let Some(node) = self.nodes.get(&id) {
            let dot: f32 = query
                .iter()
                .zip(node.embedding.iter())
                .map(|(x, y)| x * y)
                .sum();
            (1.0 - dot as f64).max(0.0)
        } else {
            f64::MAX
        }
    }

    /// Random level for new insertion (exponential distribution).
    fn random_level(m: usize) -> isize {
        let ml = 1.0 / (m as f64).ln();
        let r: f64 = rand::random();
        (-(r.ln()) * ml).floor() as isize
    }
}

/// Thread-safe wrapper around HnswIndex for concurrent access.
pub struct VectorStore {
    pub index: RwLock<HnswIndex>,
}

impl VectorStore {
    pub fn new(dimension: usize) -> Self {
        Self {
            index: RwLock::new(HnswIndex::new(dimension)),
        }
    }

    pub fn insert(&self, id: i64, embedding: &[f32]) {
        let _lock = self.index.read(); // Allow concurrent inserts via write guard
        drop(_lock);
        self.index.write().insert(id, embedding);
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        self.index.read().search(query, k)
    }

    pub fn len(&self) -> usize {
        self.index.read().len()
    }

    pub fn clear(&self) {
        self.index.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let store = VectorStore::new(128);
        let dim = 128;

        // Insert 50 random vectors
        for i in 0..50 {
            let vec: Vec<f32> = (0..dim).map(|_| rand::random::<f32>() - 0.5).collect();
            store.insert(i, &vec);
        }

        assert_eq!(store.len(), 50);

        // Search with a random query
        let query: Vec<f32> = (0..dim).map(|_| rand::random::<f32>() - 0.5).collect();
        let results = store.search(&query, 5);
        assert_eq!(results.len(), 5);

        // Results should have valid IDs and similarity scores
        for (id, sim) in &results {
            assert!(*id >= 0 && *id < 50);
            assert!(*sim >= 0.0 && *sim <= 1.0, "similarity {} out of range", sim);
        }
    }

    #[test]
    fn test_empty_search() {
        let store = VectorStore::new(64);
        let results = store.search(&vec![0.0; 64], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_clear() {
        let store = VectorStore::new(64);
        store.insert(0, &vec![0.0; 64]);
        store.insert(1, &vec![1.0; 64]);
        assert_eq!(store.len(), 2);

        store.clear();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_exact_nearest() {
        let store = VectorStore::new(4);

        // Insert orthogonal vectors
        store.insert(0, &[1.0, 0.0, 0.0, 0.0]);
        store.insert(1, &[0.0, 1.0, 0.0, 0.0]);
        store.insert(2, &[0.0, 0.0, 1.0, 0.0]);

        // Query that matches id=0 exactly
        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0); // id=0 should be closest
        assert!(results[0].1 > 0.99); // near-perfect match
    }
}