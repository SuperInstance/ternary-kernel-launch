# ternary-kernel-launch

**The CPU-side launch infrastructure for dispatching ternary kernels to a GPU — without depending on one.**

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Why This Exists

GPU programming is split into two worlds: the *kernel* (code that runs on the GPU) and the *launch* (the CPU-side code that configures and dispatches the kernel). The launch is surprisingly complex — grid dimensions, block sizes, shared memory budgets, stream ordering, event synchronization. It's infrastructure, not algorithm.

This crate provides that infrastructure for ternary neural network kernels. It models GPU launch abstractions (streams, events, FIFO ordering) in pure Rust, with no GPU runtime dependency. You can design, test, and iterate on your scheduling logic on a laptop, then swap in a real GPU backend when you're ready.

## The Key Insight

Ternary kernels have unique launch requirements. A 1024×1024 ternary matmul operates on matrices that are 16× denser than float32 (packed trits). The launch dimensions, shared memory budgets, and data transfer sizes all change when your basic unit is a packed trit word instead of a float. This crate bakes those calculations in — `compute_ternary_1d` knows that 16 trits pack into one u32 and sizes the grid accordingly.

## Quick Start

```toml
[dependencies]
ternary-kernel-launch = "0.1"
```

```rust
use ternary_kernel_launch::*;

// Configure a kernel launch
let config = KernelConfig::new(256, 128)       // 256 blocks × 128 threads
    .with_shared_mem(8192)                      // 8 KB shared memory per block
    .with_name("ternary_matmul_4x4");           // symbolic name for profiling

config.validate()?;  // checks: block_size ≤ 1024, grid > 0, etc.
println!("Total threads: {}", config.total_threads()); // 32768

// Compute launch dimensions from data size
let (grid, block) = LaunchParams::compute_1d(1_000_000, 256);
// grid = 3907, block = 256

// Ternary-specific: packed trits (16 per u32)
let (grid, block) = LaunchParams::compute_ternary_1d(1_000_000, 256);
// operates on (1_000_000 + 15) / 16 = 62500 words

// Stream queue for ordered execution
let stream = StreamQueue::new(0);
stream.enqueue(KernelConfig::new(10, 256).with_name("layer_0"), 102400);
stream.enqueue(KernelConfig::new(20, 128).with_name("layer_1"), 204800);
stream.flush(1_000_000_000); // simulate at 1 GiB/s

// Event-based completion tracking
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
let completed = Arc::new(AtomicU64::new(0));
let pool = EventPool::new(completed.clone());
let ev = pool.record(5, Some("checkpoint".into()));
completed.store(6, Ordering::SeqCst);
assert_eq!(pool.is_completed(ev), Some(true));

// Fluent builder
let stream = StreamQueue::new(0);
let (seq, config) = LaunchBuilder::new(&stream)
    .name("ternary_attention")
    .block_size(256)
    .shared_mem(4096)
    .elements(16384)
    .launch()?;
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   LaunchBuilder                       │
│  (fluent API: name, block_size, elements, shared_mem)│
└─────────────────────────┬───────────────────────────┘
                          │ builds
                          ▼
┌─────────────────────────────────────────────────────┐
│                   KernelConfig                        │
│  (grid_size, block_size, shared_mem_bytes, name)      │
└─────────────────────────┬───────────────────────────┘
                          │ enqueued into
                          ▼
┌─────────────────────────────────────────────────────┐
│                   StreamQueue                         │
│  (FIFO kernel ordering, simulated execution)          │
│  enqueue() → step() → flush()                        │
└─────────────────────────┬───────────────────────────┘
                          │ tracked by
                          ▼
┌─────────────────────────────────────────────────────┐
│                   EventPool                           │
│  (record events, query completion, wait with timeout) │
└─────────────────────────────────────────────────────┘
```

## Component Guide

### KernelConfig — The Launch Descriptor

Encodes everything needed to dispatch one kernel:

| Field | Type | Meaning |
|-------|------|---------|
| `grid_size` | u32 | Number of thread blocks |
| `block_size` | u32 | Threads per block (max 1024) |
| `shared_mem_bytes` | u32 | Dynamic shared memory per block |
| `name` | Option<String> | Symbolic name for profiling |

```rust
let config = KernelConfig::new(100, 256)
    .with_shared_mem(4096)
    .with_name("ternary_softmax");
config.validate()?; // validates constraints
```

### LaunchParams — Grid/Block Calculator

Computes launch dimensions from data sizes:

```rust
// Standard 1D
let (grid, block) = LaunchParams::compute_1d(elements, preferred_block);

// 2D (square-ish blocks)
let params = LaunchParams::compute_2d(width, height, preferred_block);
// params.grid_x, params.grid_y, params.block_x, params.block_y

// Ternary-packed 1D (16 trits per u32)
let (grid, block) = LaunchParams::compute_ternary_1d(trit_count, preferred_block);
```

### StreamQueue — Execution Ordering

A FIFO queue that simulates GPU stream semantics:

```rust
let stream = StreamQueue::new(0);              // stream ID 0
stream.enqueue(config1, payload_bytes_1);       // queued
stream.enqueue(config2, payload_bytes_2);       // queued after config1
stream.flush(throughput_bps);                   // execute all in order
stream.completed_seqs();                        // [0, 1] — FIFO order
```

Kernels complete in the order they were enqueued. The `step()` method simulates one kernel's execution time based on payload size and throughput.

### EventPool — Completion Tracking

Record events and query whether all work up to that point has completed:

```rust
let ev = pool.record(seq_number, Some("name"));
pool.is_completed(ev);           // Some(true) or Some(false)
pool.wait_until(ev, timeout);    // spin-wait with deadline
```

### LaunchBuilder — Fluent API

Chain configuration and launch in one expression:

```rust
LaunchBuilder::new(&stream)
    .name("ternary_layer")
    .block_size(256)
    .shared_mem(8192)
    .elements(65536)
    .launch()?; // validates, computes grid, enqueues
```

## API Reference

### Core Types

```rust
struct KernelConfig {
    pub grid_size: u32,
    pub block_size: u32,
    pub shared_mem_bytes: u32,
    pub name: Option<String>,
}

impl KernelConfig {
    fn new(grid_size: u32, block_size: u32) -> Self;
    fn with_shared_mem(self, bytes: u32) -> Self;
    fn with_name(self, name: impl Into<String>) -> Self;
    fn total_threads(&self) -> u64;
    fn validate(&self) -> Result<(), LaunchError>;
}
```

### Launch Parameters

```rust
struct LaunchParams;
impl LaunchParams {
    const MAX_BLOCK_SIZE: u32 = 1024;
    fn compute_1d(elements: u64, preferred_block: u32) -> (u32, u32);
    fn compute_2d(width: u64, height: u64, preferred_block: u32) -> LaunchParams2d;
    fn compute_ternary_1d(trit_count: u64, preferred_block: u32) -> (u32, u32);
}

struct LaunchParams2d { pub grid_x: u32, pub grid_y: u32, pub block_x: u32, pub block_y: u32 }
```

### Stream and Events

```rust
struct StreamQueue { /* ... */ }
impl StreamQueue {
    fn new(id: u32) -> Self;
    fn enqueue(&self, config: KernelConfig, payload_size: u64) -> u64;
    fn step(&self, throughput_bps: u64) -> Option<u64>;
    fn flush(&self, throughput_bps: u64) -> usize;
    fn pending_count(&self) -> usize;
    fn completed_count(&self) -> usize;
}

struct EventPool { /* ... */ }
impl EventPool {
    fn new(stream_completed_seq: Arc<AtomicU64>) -> Self;
    fn record(&self, seq: u64, name: Option<String>) -> u64;
    fn is_completed(&self, event_id: u64) -> Option<bool>;
    fn wait_until(&self, event_id: u64, timeout: Duration) -> bool;
}

struct LaunchBuilder<'a> { /* ... */ }
impl LaunchBuilder<'_> {
    fn name(self, name: impl Into<String>) -> Self;
    fn block_size(self, bs: u32) -> Self;
    fn shared_mem(self, bytes: u32) -> Self;
    fn elements(self, n: u64) -> Self;
    fn payload_size(self, bytes: u64) -> Self;
    fn launch(self) -> Result<(u64, KernelConfig), LaunchError>;
}
```

### Errors

```rust
enum LaunchError {
    InvalidConfig(String),
    QueueFull,
    StreamDestroyed,
}
```

## Real-World Example: Dispatching a Ternary Transformer Layer

```rust
let stream = StreamQueue::new(0);
let completed_seq = Arc::new(AtomicU64::new(0));
let events = EventPool::new(completed_seq.clone());

// Layer 0: ternary attention
let (seq0, _) = LaunchBuilder::new(&stream)
    .name("ternary_attention")
    .block_size(256)
    .shared_mem(8192)
    .elements(65536)  // 64K packed trits
    .launch()?;

// Layer 1: ternary FFN (depends on layer 0)
let (seq1, _) = LaunchBuilder::new(&stream)
    .name("ternary_ffn")
    .block_size(128)
    .shared_mem(4096)
    .elements(32768)
    .launch()?;

// Record a sync point after attention
let sync_point = events.record(seq0, Some("post_attention".into()));

// Simulate execution
stream.flush(1_000_000_000);
completed_seq.store(seq1 + 1, Ordering::SeqCst);

// Check if attention is done before proceeding
if events.is_completed(sync_point) == Some(true) {
    println!("Attention complete, FFN started");
}
```

## Performance Characteristics

This crate is **entirely CPU-side**. It simulates GPU scheduling semantics but does not execute GPU code. All operations are:

- `KernelConfig::validate()`: O(1)
- `LaunchParams::compute_*`: O(1)
- `StreamQueue::enqueue()`: O(1) (push back)
- `StreamQueue::flush()`: O(n) where n = pending kernels
- `EventPool::record()`: O(1)
- `EventPool::is_completed()`: O(n) linear scan (suitable for dozens of events, not thousands)

Memory: O(n) for n queued kernels. O(m) for m recorded events. Both are lightweight — this is scheduling infrastructure, not data processing.

## Ecosystem Connections

This crate is the dispatch layer for the ternary GPU pipeline:

- [`ternary-matmul`](https://github.com/SuperInstance/ternary-matmul) — the matmul kernel this dispatches
- [`ternary-conv`](https://github.com/SuperInstance/ternary-conv) — the convolution kernel this dispatches
- [`ternary-memory-pool`](https://github.com/SuperInstance/ternary-memory-pool) — manages the memory these kernels operate on

Together, these three crates form the CPU-side infrastructure for a complete ternary GPU inference engine.

## Open Questions

- **Multi-stream scheduling**: Current implementation uses a single stream. Real GPU workloads use multiple streams for concurrent execution. A multi-stream scheduler with cross-stream event dependencies would be the next step.
- **Real GPU backend**: The simulation is faithful, but at some point you need `cudaLaunchKernel` or the ROCm equivalent. The design here is backend-agnostic — a real backend would implement the same traits.
- **Kernel dependency graphs**: For models with non-sequential topology (residual connections, branching), a DAG-based scheduler would be more appropriate than FIFO streams.

## Testing

```bash
cargo test
```

94+ tests covering: config validation (total threads, zero block, zero grid, max block exceeded), launch params (1D exact/remainder/single/zero/max-clamp/large, 2D basic/small, ternary 1D basic/large), stream queue (enqueue/count, FIFO ordering, flush all, empty step, sequence preservation), events (record/check, unknown returns None, wait completion, wait timeout, pool len), builder (basic launch, single element, zero elements, bad block, max block, preserves FIFO), and stream+event integration.

## License

Dual-licensed under MIT or Apache-2.0.
