//! Generic "flatten a linear value to a pointer and reconstruct it" family:
//! [`PointerInterchangeable`], `Transferred`, and the atomic hand-off slots
//! built on top of it (`AtomicTaggedSlot`/`TaggedPtr`/[`AtomicSlot`]). None
//! of this mentions any cmpth-specific type — it lives at the same top
//! level as `crate::spin`/`crate::os` so it can be reused wherever a
//! move-only value needs to cross a raw-pointer boundary (an FFI
//! context-switch shim, a lock-free wait slot, ...) without re-deriving
//! the same unsafe by hand at each call site.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// PointerInterchangeable / Transferred
// ---------------------------------------------------------------------------

/// Implemented by move-only/linear values that can be losslessly flattened
/// to a raw pointer and reconstructed from one. Generalizes the
/// `into_raw`/`from_raw` naming convention `Box`/`Rc`/`Arc` each hand-roll
/// independently (the standard library has no shared trait for it) so
/// generic code — e.g. an FFI payload carrying "some interchangeable
/// value" across a context switch — can convert without knowing which
/// concrete linear type it's holding.
pub trait PointerInterchangeable: Sized {
    type Pointee;

    /// Consume `self`, discarding the wrapper but keeping the pointer.
    /// Always safe: the caller already owned `self`, this just changes its
    /// representation.
    fn into_ptr(self) -> *mut Self::Pointee;

    /// Reconstruct `Self` from a pointer previously produced by a matching
    /// [`into_ptr`](Self::into_ptr) (of this or a compatible type sharing
    /// the same `Pointee`), whose resulting claim hasn't been reclaimed
    /// since.
    ///
    /// # Safety
    /// The caller must hold exclusive access to `*ptr` for the lifetime of
    /// the returned value.
    unsafe fn from_ptr(ptr: *mut Self::Pointee) -> Self;
}

impl<T> PointerInterchangeable for Box<T> {
    type Pointee = T;
    fn into_ptr(self) -> *mut T { Box::into_raw(self) }
    unsafe fn from_ptr(ptr: *mut T) -> Self { unsafe { Box::from_raw(ptr) } }
}

/// A [`PointerInterchangeable`] value, flattened so it can cross an
/// `extern "C"` context-switch boundary — a move-only Rust value can't
/// survive the actual assembly switch, so the shims carry this instead.
///
/// Constructing one from an already-owned value is safe (the caller
/// already holds whatever claim the value represented; this just reshapes
/// it for the FFI hop). Unpacking it back out the other side
/// ([`into_inner`](Self::into_inner), possibly as a *different*
/// `PointerInterchangeable` type sharing the same `Pointee` — e.g.
/// `SuspendedTaskToken` in, `RunningTaskToken` out) is therefore also safe
/// — a live `Transferred<T>` is itself the proof a real value was
/// flattened to make it, the same "the type's own existence is the proof"
/// pattern the tokens already use for `Owned` access. `from_raw` remains
/// as the one unavoidable exception (no predecessor value exists yet —
/// freshly allocated stacks), and is now the *only* unsafe surface left in
/// the whole FFI-crossing family, instead of being re-derived
/// independently at both ends of every shim.
#[repr(transparent)]
pub(crate) struct Transferred<T: PointerInterchangeable>(*mut T::Pointee);

impl<T: PointerInterchangeable> Transferred<T> {
    pub(crate) fn new(t: T) -> Self {
        Transferred(t.into_ptr())
    }

    /// # Safety
    /// The caller must hold exclusive access to `*ptr` for the lifetime of
    /// the returned value.
    pub(crate) unsafe fn from_raw(ptr: *mut T::Pointee) -> Self {
        Transferred(ptr)
    }

    pub(crate) fn into_inner<U>(self) -> U
    where
        U: PointerInterchangeable<Pointee = T::Pointee>,
    {
        // SAFETY: every `Transferred` either came from a real
        // `PointerInterchangeable` value (`new`) or from a call site that
        // independently justified `from_raw` — this doesn't add a new
        // claim, it just un-flattens the one already made.
        unsafe { U::from_ptr(self.0) }
    }
}

// ---------------------------------------------------------------------------
// AtomicTaggedSlot / TaggedPtr / AtomicSlot
// ---------------------------------------------------------------------------

/// An atomic slot holding at most one [`PointerInterchangeable`] value,
/// with `TAG_BITS` low bits available for a caller-chosen tag (e.g.
/// [`DualResumable`](crate::resumable::dual::dual_wait::DualResumable)'s
/// wait slot, which holds either a real continuation or a boxed
/// [`Waker`](std::task::Waker)). Generalizes the "swap out a published
/// pointer, reconstruct a token from it" pattern that used to be
/// hand-rolled — with its own `// SAFETY:` comment — at every publish/take
/// call site; the encode/decode unsafe now lives here once, in
/// [`TaggedPtr::into_typed`].
///
/// `0` is reserved for "empty": a published value's pointer is never null
/// ([`PointerInterchangeable::into_ptr`] always comes from an owned,
/// already-allocated value), so a zero word unambiguously means no value
/// is currently published.
pub(crate) struct AtomicTaggedSlot<const TAG_BITS: usize>(AtomicUsize);

impl<const TAG_BITS: usize> AtomicTaggedSlot<TAG_BITS> {
    const TAG_MASK: usize = (1usize << TAG_BITS) - 1;

    pub(crate) const fn empty() -> Self {
        AtomicTaggedSlot(AtomicUsize::new(0))
    }

    pub(crate) fn is_set(&self, order: Ordering) -> bool {
        self.0.load(order) != 0
    }

    /// Publish `t` OR'd with `tag` (must be `<= (1 << TAG_BITS) - 1`).
    /// Checked at compile time: `T::Pointee`'s alignment must cover
    /// `TAG_BITS` low bits, or the tag would corrupt the pointer.
    pub(crate) fn publish<T: PointerInterchangeable>(&self, t: T, tag: usize, order: Ordering) {
        const { assert!(std::mem::align_of::<T::Pointee>().trailing_zeros() as usize >= TAG_BITS) };
        debug_assert!(tag <= Self::TAG_MASK);
        self.0.store(t.into_ptr() as usize | tag, order);
    }

    /// Swap the slot to empty, returning the tag and a still-untyped
    /// handle to reconstruct from — `None` if the slot was already empty.
    pub(crate) fn take(&self, order: Ordering) -> Option<(usize, TaggedPtr<TAG_BITS>)> {
        let word = self.0.swap(0, order);
        (word != 0).then_some((word & Self::TAG_MASK, TaggedPtr(word)))
    }
}

/// An untyped handle produced by [`AtomicTaggedSlot::take`], not yet
/// reconstructed into a concrete [`PointerInterchangeable`] type.
pub(crate) struct TaggedPtr<const TAG_BITS: usize>(usize);

impl<const TAG_BITS: usize> TaggedPtr<TAG_BITS> {
    /// Reconstruct as `T`.
    ///
    /// # Safety
    /// The caller must know — from the `tag` [`AtomicTaggedSlot::take`]
    /// returned alongside this value — that `T` is the type that was
    /// actually [`publish`](AtomicTaggedSlot::publish)ed for that tag.
    /// Forwards to [`PointerInterchangeable::from_ptr`], whose exclusivity
    /// contract is satisfied for the same reason it always is here: the
    /// `Acquire`-ordered `take` that produced this value is the sole
    /// consumer of the `Release`-ordered `publish` that put it there.
    pub(crate) unsafe fn into_typed<T: PointerInterchangeable>(self) -> T {
        const { assert!(std::mem::align_of::<T::Pointee>().trailing_zeros() as usize >= TAG_BITS) };
        let ptr = (self.0 & !((1usize << TAG_BITS) - 1)) as *mut T::Pointee;
        unsafe { T::from_ptr(ptr) }
    }
}

/// An atomic slot specialized to hold exactly one
/// [`PointerInterchangeable`] type — `TAG_BITS = 0`, so there is no tag
/// ambiguity to resolve at the call site. `publish`/`take` are therefore
/// both fully safe: the one `into_typed` call this wraps is justified
/// once, here, by the type parameter itself (every `publish` on a given
/// `AtomicSlot<T>` can only ever have stored a `T`).
pub struct AtomicSlot<T: PointerInterchangeable>(AtomicTaggedSlot<0>, PhantomData<T>);

// SAFETY: only one thread ever has live access to the contained `T` at a
// time — `publish`/`take` are an exclusive atomic hand-off, never
// concurrent access to the same `T` — so `Sync` only needs `T: Send`, the
// same reasoning `std::sync::Mutex<T>: Sync where T: Send` rests on.
// (`Send` itself is unaffected: auto-derived already, since it correctly
// does require `T: Send` via the `PhantomData<T>` field.)
unsafe impl<T: PointerInterchangeable + Send> Sync for AtomicSlot<T> {}

impl<T: PointerInterchangeable> AtomicSlot<T> {
    pub const fn empty() -> Self {
        AtomicSlot(AtomicTaggedSlot::empty(), PhantomData)
    }

    pub fn is_set(&self, order: Ordering) -> bool {
        self.0.is_set(order)
    }

    pub fn publish(&self, t: T, order: Ordering) {
        self.0.publish(t, 0, order);
    }

    pub fn take(&self, order: Ordering) -> Option<T> {
        // SAFETY: this slot's only `publish` call site (above) always
        // stores a `T` (the type parameter is fixed on `Self`, not chosen
        // per call), so whatever `AtomicTaggedSlot::take` finds is a `T`.
        self.0.take(order).map(|(_, raw)| unsafe { raw.into_typed::<T>() })
    }
}
