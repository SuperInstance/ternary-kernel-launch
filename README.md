# ternary-kernel-launch

**CPU-side orchestration abstractions for launching ternary-valued GPU kernels.**

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

---

## Overview

`ternary-kernel-launch` provides the **CPU-side launch infrastructure** for dispatching ternary neural network kernels onto a GPU. It does not depend on CUDA, ROCm, or any GPU runtime — instead, it models the configuration, scheduling, and completion-tracking abstractions that a real ternary inference engine would need.

This crate is one layer in the **Ternary Neural Network (TNN) ecosystem**: ternary weights ({-1, 0, +1}) dramatically reduce memory and compute compared to full-precision networks, and this crate manages *how* those kernels get launched.

## Key Abstractions

### `KernelConfig`

Describes a single kernel launch:

```rust
use ternary_kernel_launch::KernelConfig;

let config = KernelConfig::new(256, 128)       // 256 blocks × 128 threads
    .with_shared_mem(8192)                      // 8 KB shared memory per block
    .with_name("ternary_matmul_4x4");           // symbolic name for profiling

config.validate()?;  // checks block_size ≤ 1024, grid > 0, etc.
println!("Total threads: {}", config.total_threads()); // 32768
```

### `LaunchParams`

Computes launch dimensions from data size — the "how many blocks do I need?" question:

```rust
use ternary_kernel_launch::LaunchParams;

// 1-D: 1,000,000 elements, 256 threads per block
let (grid, block) = LaunchParams::compute_1d(1_000_000, 256);
assert_eq!(block, 256);
assert_eq!(grid, 3907); // ceil(1_000_000 / 256)

// 2-D: 1024×1024 matrix, 256 threads per block
let params = LaunchParams::compute_2d(1024, 1024, 256);

// Ternary-specific: packed trits (16 trits per u32 word)
let (grid, block) = LaunchParams::compute_ternary_1d(1_000_000, 256);
```

### `StreamQueue`

A FIFO queue of kernel launches on a logical stream, simulating GPU stream ordering:

```rust
use ternary_kernel_launch::{StreamQueue, KernelConfig};

let stream = StreamQueue::new(0);

stream.enqueue(KernelConfig::new(10, 256).with_name("layer_0"), 102400);
stream.enqueue(KernelConfig::new(20, 128).with_name("layer_1"), 204800);

stream.flush(1_000_000_000); // simulate at 1 GiB/s throughput

assert_eq!(stream.completed_count(), 2);
// Kernels complete in FIFO order
```

### `EventPool`

Completion tracking via events — record a snapshot of stream progress and query later:

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use ternary_kernel_launch::EventPool;

let completed_seq = Arc::new(AtomicU64::new(0));
let pool = EventPool::new(completed_seq.clone());

let ev = pool.record(5, Some("checkpoint".into()));

// ... GPU work advances ...
completed_seq.store(6, Ordering::SeqCst);

assert_eq!(pool.is_completed(ev), Some(true));
```

### `LaunchBuilder`

Fluent builder for constructing and enqueueing kernel launches:

```rust
use ternary_kernel_launch::{StreamQueue, LaunchBuilder};

let stream = StreamQueue::new(0);

let (seq, config) = LaunchBuilder::new(&stream)
    .name("ternary_attention")
    .block_size(256)
    .shared_mem(4096)
    .elements(16384)
    .launch()?;

println!("Launched kernel {} as seq {}", config.name.unwrap(), seq);
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  LaunchBuilder                    │
│  (fluent API: name, block_size, elements, ...)   │
└──────────────────┬──────────────────────────────┘
                   │ builds
                   ▼
┌─────────────────────────────────────────────────┐
│                  KernelConfig                     │
│  (grid_size, block_size, shared_mem, name)        │
└──────────────────┬──────────────────────────────┘
                   │ enqueued into
                   ▼
┌─────────────────────────────────────────────────┐
│                  StreamQueue                      │
│  (FIFO kernel ordering, simulated execution)      │
└──────────────────┬──────────────────────────────┘
                   │ tracked by
                   ▼
┌─────────────────────────────────────────────────┐
│                  EventPool                        │
│  (completion events, wait/timeout, queries)       │
└─────────────────────────────────────────────────┘
```

## When to Use This

- **Research**: Model ternary kernel launch patterns without GPU hardware
- **Simulation**: Test scheduling algorithms, stream ordering, event dependencies
- **Prototyping**: Design the CPU-side API before implementing GPU backends
- **Education**: Understand GPU launch abstractions in a pure-Rust, no-hardware context

## Performance Notes

This crate is **entirely CPU-side**. It simulates GPU scheduling semantics (streams, events, FIFO ordering) but does not execute actual GPU code. The simulation is lightweight — suitable for unit testing and algorithmic prototyping at scale.

## Testing

```bash
cargo test
```

94+ tests cover: launch param computation (1D, 2D, ternary-packed), stream FIFO ordering, event completion tracking, builder validation, edge cases (1 element, max block size, empty queues), and stream+event integration.

## License

Dual-licensed under MIT or Apache-2.0.
