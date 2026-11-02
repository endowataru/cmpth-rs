//! Machinery shared by all `resumable` flavors (stackful, stackless, dual)
//! alike.

pub mod deque;
pub mod desc;
pub mod dispatch;
pub mod external_queue;
pub mod idle;
pub mod lifecycle;
pub mod lookup;
pub mod observe;
pub mod placement;
pub mod pool;
pub mod qos;
pub mod scheduler;
pub mod stack;
pub mod steal;
pub mod sync;
pub mod system;
pub mod thread;
pub mod waker;
pub mod worker;
pub mod worker_core;
pub mod yield_policy;
