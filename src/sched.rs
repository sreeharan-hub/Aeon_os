//! Preemptive round-robin scheduler for kernel threads.
//!
//! A task's whole CPU context lives on its own kernel stack. The timer (and the voluntary-yield
//! interrupt) pushes registers, calls [`aeon_sched_irq`] with the resulting stack pointer, and resumes
//! whichever stack pointer it returns (see the stubs in `arch`). Task 0 is the boot thread and doubles
//! as the idle task: it only runs when nothing else is ready.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::arch;
use crate::global::Global;
use crate::mm;
use crate::readyq::ReadyQueue;
use crate::schedtrace::{Event, Reason, TraceRing};
use crate::waitq::WaitList;

pub const MAX_TASKS: usize = 32;
const STACK_PAGES: usize = 8; // 32 KiB per task
const STACK_BYTES: usize = STACK_PAGES * 4096;
/// Stack-overflow canary. Starts as a fixed value and is replaced by a boot-seed-derived one before
/// the first task is created (`set_canary`), so it cannot be guessed from the binary.
static CANARY_VALUE: AtomicU64 = AtomicU64::new(0xAE0A_5AFE_C0DE_D00D);

pub fn set_canary(v: u64) {
    CANARY_VALUE.store(v, Ordering::Relaxed);
}

fn canary() -> u64 {
    CANARY_VALUE.load(Ordering::Relaxed)
}
const NAME_LEN: usize = 16;
/// The address bits of CR3 (the low 12 are flags/PCID).
const CR3_ADDR: u64 = 0x000F_FFFF_FFFF_F000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Free,
    Ready,
    Running,
    Sleeping,
    /// Waiting on a [`WaitQueue`] until another task or a handler wakes it.
    Blocked,
    Dead,
}

impl State {
    pub fn name(self) -> &'static str {
        match self {
            State::Free => "free",
            State::Ready => "ready",
            State::Running => "running",
            State::Sleeping => "sleeping",
            State::Blocked => "blocked",
            State::Dead => "dead",
        }
    }
}

#[derive(Clone, Copy)]
struct Task {
    state: State,
    rsp: u64,
    /// Lowest address of the stack allocation (0 for the boot thread), holds the overflow canary.
    stack: u64,
    wake_tick: u64,
    ticks: u64,
    name: [u8; NAME_LEN],
    name_len: usize,
    /// Runs in ring 3 (see `usermode`).
    user: bool,
    /// Page-table root of the task's process; 0 for kernel threads, which use the kernel's own tables.
    cr3: u64,
}

/// Where a ring-3 task starts: instruction and stack pointers, flags and the general registers in the
/// order the switch frame holds them (r15 r14 r13 r12 r11 r10 r9 r8 rdi rsi rbp rbx rdx rcx rax).
#[derive(Clone, Copy)]
pub struct UserStart {
    pub rip: u64,
    pub rsp: u64,
    pub rflags: u64,
    pub gpr: [u64; GPR_COUNT],
    pub cr3: u64,
}

impl UserStart {
    /// A fresh program: every register zero, interrupts on.
    pub fn fresh(rip: u64, rsp: u64, cr3: u64) -> Self {
        Self { rip, rsp, rflags: 0x202, gpr: [0; GPR_COUNT], cr3 }
    }
}

const FREE_TASK: Task =
    Task { state: State::Free, rsp: 0, stack: 0, wake_tick: 0, ticks: 0, name: [0; NAME_LEN], name_len: 0, user: false, cr3: 0 };

struct Sched {
    tasks: [Task; MAX_TASKS],
    current: usize,
    active: bool,
    /// Top of the boot thread's stack: the kernel stack ring-3 entries use while task 0 runs.
    boot_stack_top: u64,
    /// The ready queue: ids of the tasks that can run, front first. It holds exactly the
    /// tasks whose state is `Ready`; the running task, the idle task and any waiting task are not in it.
    ready: ReadyQueue<MAX_TASKS>,
}

static SCHED: Global<Sched> =
    Global::new(Sched { tasks: [FREE_TASK; MAX_TASKS], current: 0, active: false, boot_stack_top: 0, ready: ReadyQueue::new() });

/// How many context switches the trace remembers (the totals keep counting past this).
pub const TRACE_LEN: usize = 1024;

/// The scheduler trace. A separate all-zero static, so its 16 KiB stay in `.bss` and out of the image.
static TRACE: Global<TraceRing<TRACE_LEN>> = Global::new(TraceRing::new());

/// Initial x87/SSE state for new tasks: FCW = 0x037F, MXCSR = 0x1F80, everything else zero.
const fn fx_template() -> [u8; 512] {
    let mut t = [0u8; 512];
    t[0] = 0x7F;
    t[1] = 0x03;
    t[24] = 0x80;
    t[25] = 0x1F;
    t
}
static FX_TEMPLATE: [u8; 512] = fx_template();

/// Register-frame layout built for a new task (low to high): fxsave area, 15 GPRs, iretq frame.
pub const GPR_COUNT: usize = 15;
const FRAME_QWORDS: usize = 512 / 8 + GPR_COUNT + 5;

/// Turns the running boot thread into task 0. Call once, before enabling preemption.
pub fn init() {
    arch::without_interrupts(|| {
        // SAFETY: interrupts off on a single CPU.
        let s = unsafe { SCHED.get_irqs_off() };
        s.tasks[0] = Task { state: State::Running, name: *b"idle\0\0\0\0\0\0\0\0\0\0\0\0", name_len: 4, ..FREE_TASK };
        s.current = 0;
        s.boot_stack_top = arch::rsp0();
        s.active = true;
    });
}

/// Gives a task stack (and its guard page) back to the frame allocator.
fn free_stack(stack: u64, guarded: bool) {
    let base = stack - 4096;
    if guarded {
        crate::paging::restore_page(base);
    }
    // The stack held saved registers and kernel data: wipe it before the frames can be reused.
    // SAFETY: the stack pages (above the guard page) are mapped and no longer in use by any task.
    unsafe { core::ptr::write_bytes(stack as *mut u8, 0, STACK_BYTES) };
    mm::free_frames(base, STACK_PAGES + 1);
}

extern "C" fn task_exit() -> ! {
    exit()
}

/// Creates a kernel thread running `entry(arg)`. Returns its task id.
pub fn spawn(name: &str, entry: extern "C" fn(usize), arg: usize) -> Option<usize> {
    create(name, entry as *const () as u64, arg, None)
}

/// Creates a task that starts in ring 3 at `rip` with stack pointer `user_rsp`. It runs on its own
/// kernel stack whenever it enters the kernel.
pub fn spawn_user(name: &str, start: &UserStart) -> Option<usize> {
    create(name, start.rip, 0, Some(start))
}

fn create(name: &str, rip: u64, arg: usize, user: Option<&UserStart>) -> Option<usize> {
    // One extra page below the stack is unmapped as a guard, so an overflow faults immediately.
    let base = mm::alloc_frames(STACK_PAGES + 1)?;
    // No guard page means a stack overflow would silently corrupt a neighbour: refuse to start the task.
    if !crate::paging::guard_page(base) {
        mm::free_frames(base, STACK_PAGES + 1);
        return None;
    }
    let guarded = true;
    let stack = base + 4096;
    let top = stack + STACK_BYTES as u64;
    // The entry function starts as if it had been called: rsp = 8 (mod 16) with a return address on
    // top (task_exit, in case it returns). The saved register frame sits below that reserved area.
    let entry_rsp = top - 40;
    let sp = (top - 64 - 8 * FRAME_QWORDS as u64) & !0xF;
    // SAFETY: the frames were just allocated and are identity-mapped, writable and exclusively ours.
    unsafe {
        (stack as *mut u64).write(canary());
        // Return address for an entry function that returns: terminate the task.
        ((entry_rsp) as *mut u64).write(task_exit as *const () as u64);
        core::ptr::copy_nonoverlapping(FX_TEMPLATE.as_ptr(), sp as *mut u8, 512);
        let g = (sp + 512) as *mut u64;
        for i in 0..GPR_COUNT {
            g.add(i).write(0);
        }
        // GPR order from the lowest address: r15 r14 r13 r12 r11 r10 r9 r8 rdi rsi rbp rbx rdx rcx rax.
        g.add(8).write(arg as u64); // rdi (unused by win64, kept for clarity)
        g.add(13).write(arg as u64); // rcx: first argument in the Microsoft x64 ABI
        if let Some(u) = user {
            for (i, v) in u.gpr.iter().enumerate() {
                g.add(i).write(*v);
            }
        }
        let f = g.add(GPR_COUNT);
        f.write(rip);
        f.add(1).write(if user.is_some() { arch::USER_CS } else { arch::KERNEL_CS } as u64); // cs
        // Only the arithmetic flags of a forked parent carry over; IF is always set, IOPL never.
        f.add(2).write(user.map_or(0x202, |u| (u.rflags & 0xCD5) | 0x202));
        f.add(3).write(user.map_or(entry_rsp, |u| u.rsp)); // rsp
        f.add(4).write(if user.is_some() { arch::USER_DS } else { arch::KERNEL_DS } as u64); // ss
    }

    arch::without_interrupts(|| {
        // SAFETY: interrupts off on a single CPU.
        let s = unsafe { SCHED.get_irqs_off() };
        let slot = s.tasks.iter().position(|t| t.state == State::Free)?;
        let mut nm = [0u8; NAME_LEN];
        let n = name.len().min(NAME_LEN);
        nm[..n].copy_from_slice(&name.as_bytes()[..n]);
        s.tasks[slot] = Task { state: State::Ready, rsp: sp, stack, wake_tick: 0, ticks: 0, name: nm, name_len: n, user: user.is_some(), cr3: user.map_or(0, |u| u.cr3) };
        // A new task is ready: it joins the back of the queue.
        s.ready.push_back(slot as u8);
        Some(slot)
    })
    .or_else(|| {
        free_stack(stack, guarded);
        None
    })
}

/// Called by the interrupt stubs with the interrupted context's stack pointer; returns the stack
/// pointer to resume. Runs with interrupts disabled.
#[unsafe(no_mangle)]
extern "C" fn aeon_sched_irq(sp: u64, timer: u64) -> u64 {
    if timer != 0 {
        arch::timer_tick();
    }
    // SAFETY: interrupt context on a single CPU; no other reference to SCHED is live because every
    // other access disables interrupts first.
    let s = unsafe { SCHED.get_irqs_off() };
    if !s.active {
        return sp;
    }
    let now = arch::ticks();
    let cur = s.current;
    s.tasks[cur].rsp = sp;
    if timer != 0 {
        s.tasks[cur].ticks += 1;
    }
    if cur != 0 && s.tasks[cur].stack != 0 {
        // SAFETY: the canary word is the first word of the task's stack allocation.
        let c = unsafe { (s.tasks[cur].stack as *const u64).read() };
        if c != canary() {
            panic!("kernel stack overflow in task {} ({})", cur, core::str::from_utf8(&s.tasks[cur].name[..s.tasks[cur].name_len]).unwrap_or("?"));
        }
    }
    // Why the task is leaving the CPU: this is what the trace records.
    let leaving = s.tasks[cur].state;
    let reason = match leaving {
        State::Blocked => Reason::Block,
        State::Sleeping => Reason::Sleep,
        State::Dead => Reason::Exit,
        _ if timer != 0 => Reason::Timer,
        _ => Reason::Yield,
    };
    // Release stacks of finished tasks (never the one we are switching away from this instant).
    for (i, task) in s.tasks.iter_mut().enumerate().skip(1) {
        if task.state == State::Dead && i != cur {
            free_stack(task.stack, true);
            let was_user = task.user;
            *task = FREE_TASK;
            if was_user {
                // The process behind the task gives back its memory and files. This CPU is not using
                // its page tables: `i` is not the running task and CR3 follows the running task.
                crate::usermode::task_gone(i);
            }
        }
    }
    // Tasks whose sleep is over become ready and join the back of the queue first ...
    for i in 1..MAX_TASKS {
        if s.tasks[i].state == State::Sleeping && s.tasks[i].wake_tick <= now {
            s.tasks[i].state = State::Ready;
            s.ready.push_back(i as u8);
        }
    }
    // ... then the task that was running goes behind them (the idle task is never queued).
    if s.tasks[cur].state == State::Running {
        s.tasks[cur].state = State::Ready;
        if cur != 0 {
            s.ready.push_back(cur as u8);
        }
    }
    // Selection: the front of the ready queue, or the idle task when the queue is empty. A stale id
    // (a task that is no longer ready) is skipped; `kill` removes ids, so this should never happen.
    let next = loop {
        match s.ready.pop_front() {
            Some(id) if s.tasks[id as usize % MAX_TASKS].state == State::Ready => break id as usize % MAX_TASKS,
            Some(_) => continue,
            None => break 0,
        }
    };
    if next != cur {
        // SAFETY: interrupts are off (this is the interrupt path), so no other reference to the trace exists.
        unsafe { TRACE.get_irqs_off() }.push(Event { tick: now, from: cur as u8, to: next as u8, reason });
    }
    s.current = next;
    s.tasks[next].state = State::Running;
    let ktop = if next == 0 { s.boot_stack_top } else { s.tasks[next].stack + STACK_BYTES as u64 };
    // SAFETY: interrupt context (interrupts off); `ktop` is the top of the incoming task's mapped kernel stack.
    unsafe { crate::usermode::set_kernel_stack(ktop) };
    // The address space follows the task: a process's own tables, or the kernel's for kernel threads.
    let root = if s.tasks[next].cr3 != 0 { s.tasks[next].cr3 } else { crate::paging::kernel_root() };
    if arch::read_cr3() & CR3_ADDR != root {
        // SAFETY: `root` is the kernel PML4 or a process PML4 that holds the kernel's two entries, so
        // the code, stacks and data in use right now stay mapped across the switch.
        unsafe { arch::write_cr3(root) };
    }
    s.tasks[next].rsp
}

/// Tasks waiting for something to happen (a key, a device, another task). A waiting task is not
/// scheduled at all, so it costs no CPU until it is woken. Like a condition variable, a wake-up only
/// says "look again": the woken task must re-check its condition, and a stale id (a task killed while
/// waiting whose slot was reused) can cause one harmless extra wake-up.
pub struct WaitQueue {
    list: Global<WaitList<MAX_TASKS>>,
}

impl WaitQueue {
    pub const fn new() -> Self {
        Self { list: Global::new(WaitList::new()) }
    }

    /// Blocks the calling task until `wake_all`. Does nothing for the idle task or before the
    /// scheduler runs (they have nobody to switch to), so callers must loop on their own condition.
    pub fn wait(&self) {
        let blocked = arch::without_interrupts(|| {
            // SAFETY: interrupts off on a single CPU.
            let s = unsafe { SCHED.get_irqs_off() };
            let c = s.current;
            if !s.active || c == 0 {
                return false;
            }
            // SAFETY: as above; the list is only touched with interrupts off.
            let list = unsafe { self.list.get_irqs_off() };
            if !list.push(c as u8) {
                return false;
            }
            s.tasks[c].state = State::Blocked;
            true
        });
        if blocked {
            yield_now();
        }
    }

    /// Makes every waiting task ready again. Safe from task context; returns how many were woken.
    pub fn wake_all(&self) -> usize {
        arch::without_interrupts(|| {
            // SAFETY: interrupts off on a single CPU.
            let s = unsafe { SCHED.get_irqs_off() };
            // SAFETY: as above.
            let list = unsafe { self.list.get_irqs_off() };
            let mut woken = 0;
            list.drain(|id| {
                let t = &mut s.tasks[id as usize % MAX_TASKS];
                if t.state == State::Blocked {
                    t.state = State::Ready;
                    s.ready.push_back((id as usize % MAX_TASKS) as u8);
                    woken += 1;
                }
            });
            woken
        })
    }
}

impl Default for WaitQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Gives up the CPU until the scheduler runs this task again.
pub fn yield_now() {
    arch::yield_cpu();
}

/// Sleeps for at least `ms` milliseconds (10 ms granularity).
pub fn sleep_ms(ms: u64) {
    // SAFETY: plain read of a flag written only during init.
    if !unsafe { (*SCHED.as_ptr()).active } {
        // No scheduler (timer unavailable): wait on the TSC instead.
        let t0 = arch::rdtsc();
        let ticks = crate::log::tsc_hz() / 1000 * ms;
        while arch::rdtsc().wrapping_sub(t0) < ticks {
            arch::pause();
        }
        return;
    }
    arch::without_interrupts(|| {
        // SAFETY: interrupts off on a single CPU.
        let s = unsafe { SCHED.get_irqs_off() };
        let c = s.current;
        // One extra tick: the current tick is already partly over, and a sleep must last at least `ms`.
        s.tasks[c].wake_tick = arch::ticks().saturating_add(ms.div_ceil(1000 / arch::TIMER_HZ).max(1)).saturating_add(1);
        s.tasks[c].state = State::Sleeping;
    });
    yield_now();
}

/// Ends the calling task.
pub fn exit() -> ! {
    arch::without_interrupts(|| {
        // SAFETY: interrupts off on a single CPU.
        let s = unsafe { SCHED.get_irqs_off() };
        let c = s.current;
        if c != 0 {
            s.tasks[c].state = State::Dead;
        }
    });
    loop {
        yield_now();
        arch::wait_for_interrupt();
    }
}

/// Terminates task `id` (not the idle task). Returns false if there is no such live task.
pub fn kill(id: usize) -> bool {
    arch::without_interrupts(|| {
        // SAFETY: interrupts off on a single CPU.
        let s = unsafe { SCHED.get_irqs_off() };
        if id == 0 || id >= MAX_TASKS || id == s.current || matches!(s.tasks[id].state, State::Free | State::Dead) {
            return false;
        }
        s.tasks[id].state = State::Dead;
        // A task that was waiting its turn must not stay in the ready queue.
        s.ready.remove(id as u8);
        true
    })
}

/// Calls `f(id, name)` for every task in the ready queue, front first (the order they will run in).
pub fn ready_order(mut f: impl FnMut(usize, &str)) {
    arch::without_interrupts(|| {
        // SAFETY: interrupts off on a single CPU.
        let s = unsafe { SCHED.get_irqs_off() };
        for id in s.ready.iter() {
            let t = &s.tasks[id as usize % MAX_TASKS];
            f(id as usize, core::str::from_utf8(&t.name[..t.name_len]).unwrap_or("?"));
        }
    });
}

/// True when the ready queue holds exactly the tasks whose state is `Ready` (the invariant the scheduler
/// relies on). For tests and the `queue` command.
pub fn queue_consistent() -> bool {
    arch::without_interrupts(|| {
        // SAFETY: interrupts off on a single CPU.
        let s = unsafe { SCHED.get_irqs_off() };
        let ready_tasks = s.tasks.iter().enumerate().filter(|(i, t)| *i != 0 && t.state == State::Ready).count();
        s.ready.len() == ready_tasks && s.ready.iter().all(|id| s.tasks[id as usize % MAX_TASKS].state == State::Ready)
    })
}

/// Totals of the scheduler trace: switches in all, how many were kept, and the count per reason.
pub struct TraceSummary {
    pub total: u64,
    pub held: usize,
    pub by_reason: [u64; Reason::COUNT],
}

pub fn trace_summary() -> TraceSummary {
    arch::without_interrupts(|| {
        // SAFETY: interrupts off on a single CPU.
        let t = unsafe { TRACE.get_irqs_off() };
        let mut by_reason = [0u64; Reason::COUNT];
        for r in Reason::ALL {
            by_reason[r.index()] = t.count(r);
        }
        TraceSummary { total: t.total(), held: t.len(), by_reason }
    })
}

/// The last `max` recorded switches, oldest first. Copied out so the caller can print slowly with
/// interrupts on.
pub fn trace_snapshot(max: usize) -> alloc::vec::Vec<Event> {
    let mut out = alloc::vec::Vec::with_capacity(max.min(TRACE_LEN));
    arch::without_interrupts(|| {
        // SAFETY: interrupts off on a single CPU.
        let t = unsafe { TRACE.get_irqs_off() };
        let skip = t.len().saturating_sub(max);
        for e in t.iter().skip(skip) {
            out.push(e);
        }
    });
    out
}

pub fn trace_reset() {
    arch::without_interrupts(|| {
        // SAFETY: interrupts off on a single CPU.
        unsafe { TRACE.get_irqs_off() }.reset();
    });
}

/// True while task `id` is a live ring-3 task.
pub fn user_alive(id: usize) -> bool {
    arch::without_interrupts(|| {
        // SAFETY: interrupts off on a single CPU.
        let s = unsafe { SCHED.get_irqs_off() };
        id < MAX_TASKS && s.tasks[id].user && !matches!(s.tasks[id].state, State::Free | State::Dead)
    })
}

/// Points the running task at a new page-table root and loads it (`execve` swapping the program image).
///
/// # Safety
/// `root` must be a live PML4 that contains the kernel's entries, and interrupts must be off.
pub unsafe fn set_current_root(root: u64) {
    // SAFETY: interrupts are off (caller's contract), so this is the only reference to the scheduler.
    let s = unsafe { SCHED.get_irqs_off() };
    let c = s.current;
    s.tasks[c].cr3 = root;
    // SAFETY: forwarded contract: the new tables map everything the kernel is using.
    unsafe { arch::write_cr3(root) };
}

/// The state of task `id`, or `Free` for an id out of range.
pub fn state_of(id: usize) -> State {
    arch::without_interrupts(|| {
        // SAFETY: interrupts off on a single CPU.
        let s = unsafe { SCHED.get_irqs_off() };
        s.tasks.get(id).map_or(State::Free, |t| t.state)
    })
}

pub fn current_id() -> usize {
    // SAFETY: plain read of one word; written only with interrupts off.
    unsafe { (*SCHED.as_ptr()).current }
}

pub struct TaskInfo<'a> {
    pub id: usize,
    pub name: &'a str,
    pub state: State,
    pub ticks: u64,
}

/// Calls `f` for every live task (with interrupts disabled, so keep `f` short).
pub fn for_each(mut f: impl FnMut(TaskInfo<'_>)) {
    arch::without_interrupts(|| {
        // SAFETY: interrupts off on a single CPU.
        let s = unsafe { SCHED.get_irqs_off() };
        for (id, t) in s.tasks.iter().enumerate() {
            if t.state != State::Free {
                f(TaskInfo { id, name: core::str::from_utf8(&t.name[..t.name_len]).unwrap_or("?"), state: t.state, ticks: t.ticks });
            }
        }
    });
}
