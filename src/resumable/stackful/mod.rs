//! Machinery only a genuinely stackful (real context-switch) system needs.

pub mod context;
pub mod desc;
pub mod init;
#[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
pub mod preempt;
pub mod suspended;
pub mod sync;
pub mod system;
pub mod thread;
pub mod tls;
pub mod waker;
pub mod worker;
