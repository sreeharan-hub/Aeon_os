//! The list behind a wait queue (`c-queue`): which task ids are waiting. Pure, so it is host-tested; the
//! scheduler wraps it with the blocking and waking (`sched::WaitQueue`).

/// A fixed list of up to `N` waiting task ids (each below 255). A task appears at most once.
#[derive(Clone, Copy, Debug)]
pub struct WaitList<const N: usize> {
    ids: [u8; N],
    len: usize,
}

impl<const N: usize> Default for WaitList<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> WaitList<N> {
    pub const fn new() -> Self {
        Self { ids: [0; N], len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn contains(&self, id: u8) -> bool {
        self.ids[..self.len].contains(&id)
    }

    /// Adds `id`. Returns false when the list is full or `id` cannot be stored. Adding an id that is
    /// already waiting succeeds and changes nothing.
    pub fn push(&mut self, id: u8) -> bool {
        if id == u8::MAX {
            return false;
        }
        if self.contains(id) {
            return true;
        }
        if self.len == N {
            return false;
        }
        self.ids[self.len] = id;
        self.len += 1;
        true
    }

    /// Removes `id` if present, keeping the order of the others. Returns whether it was there.
    pub fn remove(&mut self, id: u8) -> bool {
        match self.ids[..self.len].iter().position(|&x| x == id) {
            Some(i) => {
                self.ids.copy_within(i + 1..self.len, i);
                self.len -= 1;
                true
            }
            None => false,
        }
    }

    /// Empties the list, calling `f` for each id in the order they started waiting.
    pub fn drain(&mut self, mut f: impl FnMut(u8)) {
        let n = self.len;
        self.len = 0;
        for &id in &self.ids[..n] {
            f(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_is_idempotent_and_bounded() {
        let mut l = WaitList::<3>::new();
        assert!(l.is_empty());
        assert!(l.push(4));
        assert!(l.push(4));
        assert_eq!(l.len(), 1);
        assert!(l.push(5) && l.push(6));
        assert!(!l.push(7), "full");
        assert!(l.push(5), "already present still succeeds when full");
        assert_eq!(l.len(), 3);
        assert!(!WaitList::<3>::new().push(u8::MAX));
    }

    #[test]
    fn drain_visits_in_order_and_empties() {
        let mut l = WaitList::<4>::new();
        for id in [9, 2, 7] {
            l.push(id);
        }
        let mut seen = Vec::new();
        l.drain(|id| seen.push(id));
        assert_eq!(seen, [9, 2, 7]);
        assert!(l.is_empty());
        l.drain(|_| panic!("nothing left"));
        assert!(l.push(9), "usable again");
    }

    #[test]
    fn remove_keeps_order() {
        let mut l = WaitList::<4>::new();
        for id in [1, 2, 3, 4] {
            l.push(id);
        }
        assert!(l.remove(2));
        assert!(!l.remove(2));
        let mut seen = Vec::new();
        l.drain(|id| seen.push(id));
        assert_eq!(seen, [1, 3, 4]);
    }

    #[test]
    fn zero_capacity_list_never_stores() {
        let mut l = WaitList::<0>::new();
        assert!(!l.push(1));
        assert!(l.is_empty());
        l.drain(|_| panic!("empty"));
    }
}
