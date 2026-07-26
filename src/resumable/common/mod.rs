//! Machinery shared by all `resumable` flavors (stackful, stackless, dual)
//! alike.

pub mod deque;
pub mod desc;
pub mod external_queue;
pub mod lookup;
pub mod pool;
pub mod scheduler;
pub mod stack;
pub mod sync;
pub mod system;
pub mod thread;
pub mod waker;
pub mod worker;
