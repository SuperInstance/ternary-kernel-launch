//! # ternary-kernel-launch
//!
//! CPU-side orchestration abstractions for launching ternary-valued GPU kernels.
//! This crate provides the launch configuration, stream management, event tracking,
//! and builder patterns needed to dispatch ternary neural network kernels onto a GPU
//! — without depending on any actual GPU runtime.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// KernelConfig
// ---------------------------------------------------------------------------

/// Configuration for a single kernel launch.
///
/// Encodes the grid (total blocks), block (threads per block), shared-memory
/// budget, and an optional symbolic kernel name for profiling/debugging.
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Number of thread blocks in the grid (x-dimension).
    pub grid_size: u32,
    /// Number of threads per block (x-dimension).
    pub block_size: u32,
    /// Dynamic shared memory per block, in bytes.
    pub shared_mem_bytes: u32,
    /// Optional symbolic name for the kernel.
    pub name: Option<String>,
}

impl KernelConfig {
    /// Create a minimal config with grid and block sizes.
    pub fn new(grid_size: u32, block_size: u32) -> Self {
        Self {
            grid_size,
            block_size,
            shared_mem_bytes: 0,
            name: None,
        }
    }

    /// Set shared memory.
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Set symbolic kernel name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Total threads launched = grid_size × block_size.
    pub fn total_threads(&self) -> u64 {
        self.grid_size as u64 * self.block_size as u64
    }

    /// Check whether this config is well-formed.
    pub fn validate(&self) -> Result<(), LaunchError> {
        if self.block_size == 0 {
            return Err(LaunchError::InvalidConfig("block_size must be > 0".into()));
        }
        if self.grid_size == 0 {
            return Err(LaunchError::InvalidConfig("grid_size must be > 0".into()));
        }
        // Typical GPU limit is 1024 threads per block; we enforce a configurable max.
        const MAX_BLOCK: u32 = 1024;
        if self.block_size > MAX_BLOCK {
            return Err(LaunchError::InvalidConfig(format!(
                "block_size {} exceeds max {}",
                self.block_size, MAX_BLOCK
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LaunchParams
// ---------------------------------------------------------------------------

/// Computes launch dimensions from a data size (element count).
///
/// Given a total number of elements and a preferred block size, it calculates
/// the grid size (ceiling division) and provides utilities for multi-dimensional
/// launches.
pub struct LaunchParams;

impl LaunchParams {
    /// The conventional maximum threads per block for most GPU architectures.
    pub const MAX_BLOCK_SIZE: u32 = 1024;

    /// Compute 1-D launch parameters.
    ///
    /// Returns `(grid_size, block_size)` where `block_size` is clamped to
    /// `preferred_block` (or `MAX_BLOCK_SIZE` if larger), and `grid_size` is
    /// `ceil(elements / block_size)`.
    pub fn compute_1d(elements: u64, preferred_block: u32) -> (u32, u32) {
        let block = preferred_block.min(Self::MAX_BLOCK_SIZE).max(1);
        let grid = ((elements + block as u64 - 1) / block as u64) as u32;
        (grid.max(1), block)
    }

    /// Compute 2-D launch parameters.
    ///
    /// Distributes `width × height` elements across a 2-D grid, using a
    /// square-ish block (tx × ty) and computing grid dimensions accordingly.
    pub fn compute_2d(width: u64, height: u64, preferred_block: u32) -> LaunchParams2d {
        let block_1d = preferred_block.min(Self::MAX_BLOCK_SIZE).max(1);
        let tx = (block_1d as f64).sqrt().floor() as u32;
        let ty = block_1d / tx.max(1);
        let grid_x = ((width + tx as u64 - 1) / tx as u64) as u32;
        let grid_y = ((height + ty as u64 - 1) / ty as u64) as u32;
        LaunchParams2d {
            grid_x: grid_x.max(1),
            grid_y: grid_y.max(1),
            block_x: tx.max(1),
            block_y: ty.max(1),
        }
    }

    /// Compute launch params for ternary-aligned data.
    ///
    /// Ternary values are packed so that each element occupies ~1.58 bits
    /// (log₂3).  Here we treat each group of 16 trits as one 32-bit word.
    pub fn compute_ternary_1d(trit_count: u64, preferred_block: u32) -> (u32, u32) {
        // 16 trits per u32 word
        let words = (trit_count + 15) / 16;
        Self::compute_1d(words, preferred_block)
    }
}

/// 2-D launch dimensions.
#[derive(Debug, Clone, Copy)]
pub struct LaunchParams2d {
    pub grid_x: u32,
    pub grid_y: u32,
    pub block_x: u32,
    pub block_y: u32,
}

impl LaunchParams2d {
    pub fn total_blocks(&self) -> u64 {
        self.grid_x as u64 * self.grid_y as u64
    }
    pub fn total_threads(&self) -> u64 {
        self.total_blocks() * self.block_x as u64 * self.block_y as u64
    }
}

// ---------------------------------------------------------------------------
// StreamQueue
// ---------------------------------------------------------------------------

/// A FIFO queue of kernel launches on a logical stream.
///
/// In a real GPU runtime, streams allow concurrent kernel execution.  Here we
/// simulate the ordering semantics: kernels are enqueued and "executed"
/// sequentially.  Each kernel records wall-clock duration for testing.
#[derive(Debug)]
pub struct StreamQueue {
    id: u32,
    queue: Arc<Mutex<VecDeque<QueuedKernel>>>,
    completed: Arc<Mutex<Vec<CompletedKernel>>>,
    next_seq: Arc<AtomicU64>,
}

#[derive(Debug, Clone)]
struct QueuedKernel {
    seq: u64,
    config: KernelConfig,
    payload_size: u64,
}

#[derive(Debug, Clone)]
struct CompletedKernel {
    seq: u64,
    config: KernelConfig,
    duration: Duration,
}

impl StreamQueue {
    /// Create a new stream with the given ID.
    pub fn new(id: u32) -> Self {
        Self {
            id,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            completed: Arc::new(Mutex::new(Vec::new())),
            next_seq: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Stream identifier.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Enqueue a kernel launch.
    ///
    /// Returns the sequence number assigned to this launch.
    pub fn enqueue(&self, config: KernelConfig, payload_size: u64) -> u64 {
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
        let mut q = self.queue.lock().unwrap();
        q.push_back(QueuedKernel {
            seq,
            config,
            payload_size,
        });
        seq
    }

    /// Number of kernels waiting in the queue.
    pub fn pending_count(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Simulate executing the next kernel in the queue.
    ///
    /// Duration is estimated as `payload_size / throughput`, where throughput
    /// defaults to 1 GiB/s (simulated).  Returns the sequence number of the
    /// completed kernel, or `None` if the queue was empty.
    pub fn step(&self, throughput_bps: u64) -> Option<u64> {
        let qk = self.queue.lock().unwrap().pop_front()?;
        let duration = if throughput_bps == 0 {
            Duration::from_micros(1)
        } else {
            Duration::from_nanos((qk.payload_size * 1_000_000_000 / throughput_bps.max(1)).max(1))
        };
        let seq = qk.seq;
        self.completed.lock().unwrap().push(CompletedKernel {
            seq,
            config: qk.config,
            duration,
        });
        Some(seq)
    }

    /// Drain all pending kernels (simulate full execution).
    pub fn flush(&self, throughput_bps: u64) -> usize {
        let mut count = 0;
        while self.step(throughput_bps).is_some() {
            count += 1;
        }
        count
    }

    /// Number of completed kernels.
    pub fn completed_count(&self) -> usize {
        self.completed.lock().unwrap().len()
    }

    /// Get the sequence numbers of completed kernels, in order.
    pub fn completed_seqs(&self) -> Vec<u64> {
        self.completed
            .lock()
            .unwrap()
            .iter()
            .map(|c| c.seq)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// EventPool
// ---------------------------------------------------------------------------

/// A pool of completion events for tracking kernel progress.
///
/// Events are lightweight markers.  Recording an event snapshots the current
/// stream sequence number.  Querying an event tells you whether all kernels
/// up to that point have completed.
#[derive(Debug)]
pub struct EventPool {
    events: Arc<Mutex<Vec<Event>>>,
    next_id: Arc<AtomicU64>,
    stream_completed_seq: Arc<AtomicU64>,
}

#[derive(Debug, Clone)]
struct Event {
    id: u64,
    recorded_seq: u64,
    name: Option<String>,
}

impl EventPool {
    /// Create a new event pool, tracking a stream's completed sequence.
    pub fn new(stream_completed_seq: Arc<AtomicU64>) -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            next_id: Arc::new(AtomicU64::new(0)),
            stream_completed_seq,
        }
    }

    /// Record a new event at the given stream sequence number.
    pub fn record(&self, seq: u64, name: Option<String>) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.events.lock().unwrap().push(Event {
            id,
            recorded_seq: seq,
            name,
        });
        id
    }

    /// Check whether all kernels up to event `id` have completed.
    pub fn is_completed(&self, event_id: u64) -> Option<bool> {
        let events = self.events.lock().unwrap();
        let ev = events.iter().find(|e| e.id == event_id)?;
        let done_seq = self.stream_completed_seq.load(Ordering::SeqCst);
        Some(done_seq >= ev.recorded_seq + 1)
    }

    /// Wait (spin) until event completes, with timeout.
    ///
    /// Returns `true` if the event completed before the timeout.
    pub fn wait_until(&self, event_id: u64, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            match self.is_completed(event_id) {
                Some(true) => return true,
                Some(false) | None => {
                    if Instant::now() >= deadline {
                        return false;
                    }
                    std::thread::yield_now();
                }
            }
        }
    }

    /// Number of events in the pool.
    pub fn len(&self) -> usize {
        self.events.lock().unwrap().len()
    }

    /// Is the pool empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// LaunchBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing and enqueueing a kernel launch.
pub struct LaunchBuilder<'a> {
    stream: &'a StreamQueue,
    name: Option<String>,
    block_size: u32,
    shared_mem: u32,
    elements: u64,
    payload_size: u64,
}

impl<'a> LaunchBuilder<'a> {
    /// Start building a launch for `stream`.
    pub fn new(stream: &'a StreamQueue) -> Self {
        Self {
            stream,
            name: None,
            block_size: 256,
            shared_mem: 0,
            elements: 0,
            payload_size: 0,
        }
    }

    /// Set kernel name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set preferred block (thread) size.
    pub fn block_size(mut self, bs: u32) -> Self {
        self.block_size = bs;
        self
    }

    /// Set shared memory per block.
    pub fn shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem = bytes;
        self
    }

    /// Set number of data elements (used to compute grid).
    pub fn elements(mut self, n: u64) -> Self {
        self.elements = n;
        self.payload_size = n * std::mem::size_of::<f32>() as u64;
        self
    }

    /// Override payload size (bytes transferred).
    pub fn payload_size(mut self, bytes: u64) -> Self {
        self.payload_size = bytes;
        self
    }

    /// Build the `KernelConfig` without enqueuing.
    pub fn build_config(&self) -> KernelConfig {
        let (grid, block) = if self.elements > 0 {
            LaunchParams::compute_1d(self.elements, self.block_size)
        } else {
            (1, self.block_size)
        };
        let mut cfg = KernelConfig::new(grid, block).with_shared_mem(self.shared_mem);
        if let Some(ref n) = self.name {
            cfg = cfg.with_name(n);
        }
        cfg
    }

    /// Build, validate, and enqueue onto the stream.
    ///
    /// Returns `(seq, KernelConfig)` on success.
    pub fn launch(self) -> Result<(u64, KernelConfig), LaunchError> {
        let cfg = self.build_config();
        cfg.validate()?;
        let seq = self.stream.enqueue(cfg.clone(), self.payload_size);
        Ok((seq, cfg))
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during kernel launch preparation.
#[derive(Debug, Clone)]
pub enum LaunchError {
    InvalidConfig(String),
    QueueFull,
    StreamDestroyed,
}

impl std::fmt::Display for LaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaunchError::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
            LaunchError::QueueFull => write!(f, "stream queue is full"),
            LaunchError::StreamDestroyed => write!(f, "stream has been destroyed"),
        }
    }
}

impl std::error::Error for LaunchError {}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- KernelConfig ----

    #[test]
    fn config_total_threads() {
        let c = KernelConfig::new(10, 256);
        assert_eq!(c.total_threads(), 2560);
    }

    #[test]
    fn config_validate_ok() {
        KernelConfig::new(1, 1).validate().unwrap();
        KernelConfig::new(1000, 1024).validate().unwrap();
    }

    #[test]
    fn config_validate_zero_block() {
        let err = KernelConfig::new(10, 0).validate().unwrap_err();
        match err {
            LaunchError::InvalidConfig(msg) => assert!(msg.contains("block_size")),
            _ => panic!("wrong error variant"),
        }
    }

    #[test]
    fn config_validate_zero_grid() {
        let err = KernelConfig::new(0, 64).validate().unwrap_err();
        match err {
            LaunchError::InvalidConfig(msg) => assert!(msg.contains("grid_size")),
            _ => panic!("wrong error variant"),
        }
    }

    #[test]
    fn config_validate_exceeds_max_block() {
        let err = KernelConfig::new(1, 2048).validate().unwrap_err();
        match err {
            LaunchError::InvalidConfig(msg) => assert!(msg.contains("exceeds max")),
            _ => panic!("wrong error variant"),
        }
    }

    #[test]
    fn config_builder_chaining() {
        let c = KernelConfig::new(4, 128)
            .with_shared_mem(4096)
            .with_name("ternary_matmul");
        assert_eq!(c.shared_mem_bytes, 4096);
        assert_eq!(c.name.as_deref(), Some("ternary_matmul"));
    }

    // ---- LaunchParams 1-D ----

    #[test]
    fn launch_1d_exact_division() {
        let (grid, block) = LaunchParams::compute_1d(1024, 256);
        assert_eq!(grid, 4);
        assert_eq!(block, 256);
    }

    #[test]
    fn launch_1d_remainder() {
        let (grid, block) = LaunchParams::compute_1d(1000, 256);
        assert_eq!(grid, 4); // ceil(1000/256) = 4
        assert_eq!(block, 256);
    }

    #[test]
    fn launch_1d_single_element() {
        let (grid, block) = LaunchParams::compute_1d(1, 256);
        assert_eq!(grid, 1);
        assert_eq!(block, 256);
    }

    #[test]
    fn launch_1d_zero_elements() {
        // Should still produce at least 1 grid
        let (grid, _) = LaunchParams::compute_1d(0, 256);
        assert_eq!(grid, 1);
    }

    #[test]
    fn launch_1d_max_block_clamp() {
        let (grid, block) = LaunchParams::compute_1d(2048, 4096);
        assert_eq!(block, 1024);
        assert_eq!(grid, 2);
    }

    #[test]
    fn launch_1d_large_count() {
        let (grid, block) = LaunchParams::compute_1d(1_000_000, 256);
        assert_eq!(block, 256);
        assert_eq!(grid, (1_000_000 + 255) / 256);
    }

    // ---- LaunchParams 2-D ----

    #[test]
    fn launch_2d_basic() {
        let p = LaunchParams::compute_2d(1024, 1024, 256);
        assert!(p.block_x * p.block_y <= 256);
        assert!(p.grid_x >= 1);
        assert!(p.grid_y >= 1);
        assert!(p.total_threads() >= 1024 * 1024);
    }

    #[test]
    fn launch_2d_small() {
        let p = LaunchParams::compute_2d(1, 1, 64);
        assert_eq!(p.grid_x, 1);
        assert_eq!(p.grid_y, 1);
    }

    // ---- LaunchParams ternary ----

    #[test]
    fn launch_ternary_1d_basic() {
        // 16 trits = 1 word → 1 grid
        let (grid, block) = LaunchParams::compute_ternary_1d(16, 256);
        assert_eq!(grid, 1);
        assert_eq!(block, 256);
    }

    #[test]
    fn launch_ternary_1d_large() {
        let (grid, block) = LaunchParams::compute_ternary_1d(1_000_000, 256);
        let expected_words = (1_000_000 + 15) / 16;
        assert_eq!(grid, ((expected_words + 255) / 256) as u32);
    }

    // ---- StreamQueue ----

    #[test]
    fn stream_enqueue_and_count() {
        let s = StreamQueue::new(0);
        assert_eq!(s.pending_count(), 0);
        s.enqueue(KernelConfig::new(1, 64), 1024);
        s.enqueue(KernelConfig::new(2, 128), 2048);
        assert_eq!(s.pending_count(), 2);
    }

    #[test]
    fn stream_step_ordering() {
        let s = StreamQueue::new(0);
        let seq0 = s.enqueue(KernelConfig::new(1, 64).with_name("first"), 1024);
        let seq1 = s.enqueue(KernelConfig::new(2, 64).with_name("second"), 2048);
        let done0 = s.step(1_000_000_000);
        assert_eq!(done0, Some(seq0));
        let done1 = s.step(1_000_000_000);
        assert_eq!(done1, Some(seq1));
        assert_eq!(s.completed_seqs(), vec![seq0, seq1]);
    }

    #[test]
    fn stream_flush_all() {
        let s = StreamQueue::new(1);
        for i in 0..10 {
            s.enqueue(KernelConfig::new(i + 1, 64), 512);
        }
        let count = s.flush(1_000_000_000);
        assert_eq!(count, 10);
        assert_eq!(s.pending_count(), 0);
        assert_eq!(s.completed_count(), 10);
    }

    #[test]
    fn stream_step_empty_returns_none() {
        let s = StreamQueue::new(2);
        assert!(s.step(1_000).is_none());
    }

    #[test]
    fn stream_preserves_fifo_order() {
        let s = StreamQueue::new(3);
        let seqs: Vec<u64> = (0..5)
            .map(|i| s.enqueue(KernelConfig::new(i + 1, 32).with_name(format!("k{}", i)), 256))
            .collect();
        s.flush(1_000_000_000);
        assert_eq!(s.completed_seqs(), seqs);
    }

    // ---- EventPool ----

    #[test]
    fn event_record_and_check() {
        let completed_seq = Arc::new(AtomicU64::new(0));
        let pool = EventPool::new(completed_seq.clone());

        let ev0 = pool.record(0, Some("start".into()));
        let ev1 = pool.record(2, Some("mid".into()));
        let ev2 = pool.record(4, None);

        // Nothing done yet
        assert_eq!(pool.is_completed(ev0), Some(false));
        assert_eq!(pool.is_completed(ev1), Some(false));

        // Advance stream completion to seq 3
        completed_seq.store(3, Ordering::SeqCst);
        assert_eq!(pool.is_completed(ev0), Some(true)); // recorded at 0, needs ≥1
        assert_eq!(pool.is_completed(ev1), Some(true)); // recorded at 2, needs ≥3
        assert_eq!(pool.is_completed(ev2), Some(false)); // recorded at 4, needs ≥5
    }

    #[test]
    fn event_unknown_returns_none() {
        let completed_seq = Arc::new(AtomicU64::new(0));
        let pool = EventPool::new(completed_seq);
        assert_eq!(pool.is_completed(999), None);
    }

    #[test]
    fn event_wait_completes() {
        let completed_seq = Arc::new(AtomicU64::new(0));
        let pool = EventPool::new(completed_seq.clone());

        let ev = pool.record(0, None);

        // Spawn a thread that will mark completion after a tiny delay
        let cs = completed_seq.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            cs.store(1, Ordering::SeqCst);
        });

        let ok = pool.wait_until(ev, Duration::from_secs(2));
        assert!(ok);
    }

    #[test]
    fn event_wait_timeout() {
        let completed_seq = Arc::new(AtomicU64::new(0));
        let pool = EventPool::new(completed_seq);
        let ev = pool.record(5, None);
        let ok = pool.wait_until(ev, Duration::from_millis(10));
        assert!(!ok);
    }

    #[test]
    fn event_pool_len() {
        let completed_seq = Arc::new(AtomicU64::new(0));
        let pool = EventPool::new(completed_seq);
        assert!(pool.is_empty());
        pool.record(0, None);
        pool.record(1, None);
        assert_eq!(pool.len(), 2);
    }

    // ---- LaunchBuilder ----

    #[test]
    fn builder_basic_launch() {
        let s = StreamQueue::new(0);
        let (seq, cfg) = LaunchBuilder::new(&s)
            .name("ternary_gemm")
            .block_size(128)
            .shared_mem(8192)
            .elements(4096)
            .launch()
            .unwrap();

        assert_eq!(cfg.block_size, 128);
        assert_eq!(cfg.grid_size, 32); // 4096 / 128
        assert_eq!(cfg.shared_mem_bytes, 8192);
        assert_eq!(cfg.name.as_deref(), Some("ternary_gemm"));
        assert_eq!(s.pending_count(), 1);
        let _ = seq;
    }

    #[test]
    fn builder_single_element() {
        let s = StreamQueue::new(0);
        let (_, cfg) = LaunchBuilder::new(&s)
            .elements(1)
            .block_size(256)
            .launch()
            .unwrap();
        assert_eq!(cfg.grid_size, 1);
        assert_eq!(cfg.block_size, 256);
    }

    #[test]
    fn builder_zero_elements_still_launches() {
        let s = StreamQueue::new(0);
        let (_, cfg) = LaunchBuilder::new(&s).launch().unwrap();
        assert_eq!(cfg.grid_size, 1); // defaults to 1 grid
    }

    #[test]
    fn builder_validates_bad_block() {
        // KernelConfig::validate rejects block_size > 1024 directly
        let cfg = KernelConfig::new(1, 2048);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn builder_max_block_size() {
        let s = StreamQueue::new(0);
        let (_, cfg) = LaunchBuilder::new(&s)
            .block_size(1024)
            .elements(1024)
            .launch()
            .unwrap();
        assert_eq!(cfg.block_size, 1024);
        assert_eq!(cfg.grid_size, 1);
    }

    #[test]
    fn builder_preserves_stream_fifo() {
        let s = StreamQueue::new(0);
        LaunchBuilder::new(&s).name("a").elements(100).launch().unwrap();
        LaunchBuilder::new(&s).name("b").elements(200).launch().unwrap();
        LaunchBuilder::new(&s).name("c").elements(300).launch().unwrap();
        s.flush(1_000_000_000);
        let completed = s.completed.lock().unwrap();
        let names: Vec<&str> = completed
            .iter()
            .filter_map(|c| c.config.name.as_deref())
            .collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    // ---- Integration: stream + events ----

    #[test]
    fn stream_event_integration() {
        let s = StreamQueue::new(0);
        let completed_seq = Arc::new(AtomicU64::new(0));
        let pool = EventPool::new(completed_seq.clone());

        // Enqueue 3 kernels
        let seq0 = s.enqueue(KernelConfig::new(1, 64), 1024);
        let seq1 = s.enqueue(KernelConfig::new(2, 64), 2048);
        let _seq2 = s.enqueue(KernelConfig::new(3, 64), 4096);

        // Record event after second kernel
        let ev = pool.record(seq1, Some("after_second".into()));

        // Complete first two
        s.step(1_000_000_000);
        completed_seq.store(seq0 + 1, Ordering::SeqCst);
        assert_eq!(pool.is_completed(ev), Some(false));

        s.step(1_000_000_000);
        completed_seq.store(seq1 + 1, Ordering::SeqCst);
        assert_eq!(pool.is_completed(ev), Some(true));
    }
}
