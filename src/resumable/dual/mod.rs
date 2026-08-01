//! Machinery only a dual (stackful ULTs and stackless tasks sharing one
//! scheduler) system needs.

pub mod desc;
pub mod dual_wait;
pub mod worker;
