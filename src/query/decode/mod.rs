//! Mmap chunk decode and parallel scatter into logical buffers.

pub mod chunk_decode;
pub mod dense_visit;
pub mod indexing;

pub use chunk_decode::planned_chunk_mmap_slices;
