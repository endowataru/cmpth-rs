//! Macros for defining systems: [`worker_tls_anchor!`](crate::worker_tls_anchor)
//! gives a type its own TLS anchor.

/// Implements [`WorkerTlsAnchor`](crate::WorkerTlsAnchor) for each listed
/// type with a `static` of its own.
///
/// ```
/// pub struct MyMarker;
/// cmpth::worker_tls_anchor!(MyMarker);
/// ```
///
/// The types must not be generic: a `static` cannot depend on a type
/// parameter, so every instance of a generic type would share one anchor.
#[macro_export]
macro_rules! worker_tls_anchor {
    ($($t:ty),+ $(,)?) => {
        $(
            // SAFETY: the static is local to this impl, so no other type
            // returns it.
            unsafe impl $crate::WorkerTlsAnchor for $t {
                fn anchor() -> &'static $crate::TlsAnchor {
                    static A: $crate::TlsAnchor = $crate::TlsAnchor::new();
                    &A
                }
            }
        )+
    };
}
