//! Processes: ring-3 programs with their own address space, file descriptors and parent/child links.
//!
//! One process is one scheduler task. The `syscall` entry saves every register on the task's kernel stack
//! (a [`SysFrame`]), so `fork` can copy it for the child and `execve` can rewrite it. Only `rax`, `rcx`
//! and `r11` change across a call, as in the Linux ABI. System calls run with interrupts off; a call that
//! must wait (pipe, `wait4`, console input) yields the CPU in a loop and re-checks its condition.
//!
//! Kernel code never touches user memory through user addresses: `vmspace` walks the page tables and
//! copies through physical frames, so a bad pointer is an error code, never a fault in ring 0.
//!
//! The shell task (interrupts on) calls the `pub` functions below; each takes interrupts off for as long
//! as it touches process state. The shell holds [`FsGuard`] while it uses the file systems, and system
//! calls that need them wait until it lets go, so the two never overlap.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use core::arch::global_asm;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use crate::arch::{self, ExcFrame};
use crate::global::Global;
use crate::loader::{self, LoadError, HEAP_BASE, MMAP_BASE};
use crate::mm;
use crate::paging;
use crate::pipe::{End, PipeTable, ReadResult, WriteResult};
use crate::ramfs;
use crate::sched::{self, UserStart};
use crate::syscall::{self, errno, Number};
use crate::usbfs;
use crate::vmspace::{self, AddressSpace, PhysMem, PAGE, USER_END};

pub const MAX_PROCS: usize = 28;
const MAX_FDS: usize = 16;
/// Page budget for one process (16 MiB).
const MAX_USER_PAGES: usize = 4096;
const MAX_STDIN_LINE: usize = 512;

const MSR_STAR: u32 = 0xC000_0081;
const MSR_LSTAR: u32 = 0xC000_0082;
const MSR_SFMASK: u32 = 0xC000_0084;
const EFER_SCE: u64 = 1;
/// Cleared on `syscall` entry: TF, IF, DF, NT, AC (AC off keeps SMAP meaningful).
const SFMASK: u64 = 0x100 | 0x200 | 0x400 | 0x4000 | 0x4_0000;

/// Kernel stack the `syscall` entry switches to; kept equal to TSS.rsp0 for the running task.
#[unsafe(no_mangle)]
static AEON_SYSCALL_KSTACK: AtomicU64 = AtomicU64::new(0);
/// Holds the user stack pointer for the few instructions between `syscall` and the stack switch.
#[unsafe(no_mangle)]
static AEON_SYSCALL_SCRATCH: AtomicU64 = AtomicU64::new(0);

/// Set once `init` has enabled `syscall`; without it a ring-3 task could never make a call or exit.
static READY: AtomicBool = AtomicBool::new(false);
/// Pids start high so they never look like the small task ids `ps` shows for kernel tasks.
static NEXT_PID: AtomicU32 = AtomicU32::new(100);
/// Held by the shell while it uses the file systems; system calls that need them wait.
static FS_BUSY: AtomicBool = AtomicBool::new(false);

/// Keeps system calls out of the file systems while the shell uses them.
pub struct FsGuard(());

impl FsGuard {
    pub fn new() -> Self {
        FS_BUSY.store(true, Ordering::SeqCst);
        FsGuard(())
    }
}

impl Default for FsGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FsGuard {
    fn drop(&mut self) {
        FS_BUSY.store(false, Ordering::SeqCst);
    }
}

fn wait_fs() {
    while FS_BUSY.load(Ordering::SeqCst) {
        sched::yield_now();
    }
}

// ---------------------------------------------------------------------------------------------
// Register frame and entry stub
// ---------------------------------------------------------------------------------------------

/// Registers saved by `aeon_syscall_entry`, lowest address first.
#[repr(C)]
pub struct SysFrame {
    r15: u64,
    r14: u64,
    r13: u64,
    r12: u64,
    rbp: u64,
    rbx: u64,
    r9: u64,
    r8: u64,
    r10: u64,
    rdx: u64,
    rsi: u64,
    rdi: u64,
    rax: u64,
    /// User instruction pointer (`syscall` saves it in rcx).
    rip: u64,
    /// User flags (`syscall` saves them in r11).
    rflags: u64,
    rsp: u64,
}

/// Points both ring-3 entry paths (interrupts and `syscall`) at the incoming task's kernel stack.
///
/// # Safety
/// Interrupts must be off and `top` must be the 16-byte-aligned top of a mapped kernel stack.
pub unsafe fn set_kernel_stack(top: u64) {
    // SAFETY: forwarded contract.
    unsafe { arch::set_rsp0(top) };
    AEON_SYSCALL_KSTACK.store(top, Ordering::Relaxed);
}

global_asm!(
    r#"
    .text
    .p2align 4
    .globl aeon_syscall_entry
    aeon_syscall_entry:
        // rcx = user rip, r11 = user rflags, rax = number, rdi/rsi/rdx/r10/r8/r9 = arguments, interrupts masked.
        mov [rip + AEON_SYSCALL_SCRATCH], rsp
        mov rsp, [rip + AEON_SYSCALL_KSTACK]
        push qword ptr [rip + AEON_SYSCALL_SCRATCH]   // user rsp lives on this task's own kernel stack
        push r11
        push rcx
        push rax
        push rdi
        push rsi
        push rdx
        push r10
        push r8
        push r9
        push rbx
        push rbp
        push r12
        push r13
        push r14
        push r15
        mov rcx, rsp            // Microsoft x64 ABI: first argument = the frame
        sub rsp, 32
        cld
        call aeon_syscall_dispatch
        add rsp, 32
        pop r15
        pop r14
        pop r13
        pop r12
        pop rbp
        pop rbx
        pop r9
        pop r8
        pop r10
        pop rdx
        pop rsi
        pop rdi
        pop rax
        pop rcx
        pop r11
        pop rsp
        sysretq

    // Demo programs, copied to the start of a fresh address space. Position independent.
    .p2align 4
    .globl aeon_user_hello_start
    aeon_user_hello_start:
        mov edi, 1
        lea rsi, [rip + 2f]
        mov edx, 18
        mov eax, 1
        syscall
        xor edi, edi
        mov eax, 60
        syscall
    1:  jmp 1b
    2:  .ascii "hello from ring 3\n"
    3:
    .globl aeon_user_hello_end
    aeon_user_hello_end:

    .globl aeon_user_fault_start
    aeon_user_fault_start:
        mov eax, 0x100000
        mov qword ptr [rax], rax
    1:  jmp 1b
    .globl aeon_user_fault_end
    aeon_user_fault_end:

    .globl aeon_user_gp_start
    aeon_user_gp_start:
        hlt
    1:  jmp 1b
    .globl aeon_user_gp_end
    aeon_user_gp_end:

    .globl aeon_user_spin_start
    aeon_user_spin_start:
    1:  jmp 1b
    .globl aeon_user_spin_end
    aeon_user_spin_end:

    // Bad requests must fail cleanly: an unknown call number must return -ENOSYS (-38) and a write from a
    // kernel address must return -EFAULT (-14). Exit code 0 only if both did.
    .globl aeon_user_badcall_start
    aeon_user_badcall_start:
        mov eax, 99
        syscall
        cmp rax, -38
        jne 4f
        mov edi, 1
        mov esi, 0x100000
        mov edx, 8
        mov eax, 1
        syscall
        cmp rax, -14
        jne 4f
        xor edi, edi
        jmp 5f
    4:  mov edi, 1
    5:  mov eax, 60
        syscall
    1:  jmp 1b
    .globl aeon_user_badcall_end
    aeon_user_badcall_end:

    // Calls `getpid`, which this program is not allowed to make. The kernel must refuse (-1); the program
    // exits with code 0 only if it did, and with 1 if the call was obeyed.
    .globl aeon_user_denied_start
    aeon_user_denied_start:
        mov eax, 39
        syscall
        add rax, 1
        mov rdi, rax
        mov eax, 60
        syscall
    1:  jmp 1b
    .globl aeon_user_denied_end
    aeon_user_denied_end:

    // Linux-style userland program: dynamic memory (brk), file creation/write/read, stdout, clean exit.
    .p2align 4
    .globl aeon_user_linux_start
    aeon_user_linux_start:
        xor edi, edi
        mov eax, 12
        syscall
        mov rbx, rax
        lea rdi, [rbx + 4096]
        mov eax, 12
        syscall
        lea rsi, [rip + 3f]
        mov rdi, rbx
        mov ecx, 26
        cld
        rep movsb
        lea rdi, [rip + 2f]
        mov esi, 0x42
        mov edx, 420
        mov eax, 2
        syscall
        mov r12, rax
        mov rdi, r12
        mov rsi, rbx
        mov edx, 26
        mov eax, 1
        syscall
        mov rdi, r12
        mov eax, 3
        syscall
        lea rdi, [rip + 2f]
        xor esi, esi
        xor edx, edx
        mov eax, 2
        syscall
        mov r12, rax
        mov rdi, r12
        lea rsi, [rbx + 1024]
        mov edx, 26
        xor eax, eax
        syscall
        mov r13, rax
        mov rdi, r12
        mov eax, 3
        syscall
        mov edi, 1
        lea rsi, [rbx + 1024]
        mov rdx, r13
        mov eax, 1
        syscall
        xor edi, edi
        mov eax, 60
        syscall
    1:  jmp 1b
    2:  .asciz "/linux.txt"
    3:  .ascii "hello from linux userland\n"
    .globl aeon_user_linux_end
    aeon_user_linux_end:
    "#
);

unsafe extern "C" {
    fn aeon_syscall_entry();
    static aeon_user_hello_start: u8;
    static aeon_user_hello_end: u8;
    static aeon_user_fault_start: u8;
    static aeon_user_fault_end: u8;
    static aeon_user_gp_start: u8;
    static aeon_user_gp_end: u8;
    static aeon_user_spin_start: u8;
    static aeon_user_spin_end: u8;
    static aeon_user_badcall_start: u8;
    static aeon_user_badcall_end: u8;
    static aeon_user_denied_start: u8;
    static aeon_user_denied_end: u8;
    static aeon_user_linux_start: u8;
    static aeon_user_linux_end: u8;
}

/// The built-in programs `run` can start.
pub const PROGRAMS: [&str; 7] = ["hello", "fault", "gp", "spin", "badcall", "denied", "linux"];

fn program(name: &str) -> Option<&'static [u8]> {
    // SAFETY: the symbols mark the start and end of byte ranges emitted by the global_asm! block above.
    unsafe {
        let (s, e) = match name {
            "hello" => (&aeon_user_hello_start as *const u8, &aeon_user_hello_end as *const u8),
            "fault" => (&aeon_user_fault_start as *const u8, &aeon_user_fault_end as *const u8),
            "gp" => (&aeon_user_gp_start as *const u8, &aeon_user_gp_end as *const u8),
            "spin" => (&aeon_user_spin_start as *const u8, &aeon_user_spin_end as *const u8),
            "badcall" => (&aeon_user_badcall_start as *const u8, &aeon_user_badcall_end as *const u8),
            "denied" => (&aeon_user_denied_start as *const u8, &aeon_user_denied_end as *const u8),
            "linux" => (&aeon_user_linux_start as *const u8, &aeon_user_linux_end as *const u8),
            _ => return None,
        };
        Some(core::slice::from_raw_parts(s, e as usize - s as usize))
    }
}

/// Turns on `syscall`/`sysret`. Call once after the GDT is loaded.
///
/// # Safety
/// `arch::init_descriptors` must have run (the selectors in STAR refer to its GDT), interrupts off.
pub unsafe fn init() {
    // SAFETY: standard SYSCALL setup; the selectors match the GDT built by `init_descriptors`.
    unsafe {
        arch::wrmsr(arch::MSR_EFER, arch::rdmsr(arch::MSR_EFER) | EFER_SCE);
        arch::wrmsr(MSR_STAR, ((arch::SYSRET_BASE as u64) << 48) | ((arch::KERNEL_CS as u64) << 32));
        arch::wrmsr(MSR_LSTAR, aeon_syscall_entry as *const () as u64);
        arch::wrmsr(MSR_SFMASK, SFMASK);
    }
    READY.store(true, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------------------------
// Physical memory for the page-table code
// ---------------------------------------------------------------------------------------------

struct KernelMem;

impl PhysMem for KernelMem {
    fn alloc_frame(&mut self) -> Option<u64> {
        let f = mm::alloc_frames(1)?;
        // SAFETY: a frame just taken from the allocator is identity-mapped, writable and ours.
        unsafe { core::ptr::write_bytes(f as *mut u8, 0, PAGE as usize) };
        Some(f)
    }
    fn free_frame(&mut self, phys: u64) {
        mm::free_frames(phys, 1);
    }
    fn table(&mut self, phys: u64) -> &mut [u64; 512] {
        // SAFETY: `phys` is a frame owned by an address space; RAM is identity-mapped in the kernel.
        unsafe { &mut *(phys as *mut [u64; 512]) }
    }
    fn page(&mut self, phys: u64) -> &mut [u8; 4096] {
        // SAFETY: as in `table`.
        unsafe { &mut *(phys as *mut [u8; 4096]) }
    }
    fn copy_frame(&mut self, from: u64, to: u64) {
        // SAFETY: two distinct owned frames, identity-mapped.
        unsafe { core::ptr::copy_nonoverlapping(from as *const u8, to as *mut u8, PAGE as usize) };
    }
}

// ---------------------------------------------------------------------------------------------
// Process table
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
struct OpenFile {
    path: String,
    offset: usize,
    flags: u32,
    is_usb: bool,
}

#[derive(Clone)]
enum Fd {
    Null,
    ConsoleIn,
    ConsoleOut,
    File(OpenFile),
    /// A directory and how many of its entries `getdents64` has already returned.
    Dir(String, usize),
    PipeRead(usize),
    PipeWrite(usize),
}

/// What the shell hands a new program as its standard input or output.
pub enum Io {
    Console,
    Null,
    PipeRead(usize),
    PipeWrite(usize),
    /// A file to write (`>` or `>>`), already created.
    File { path: String, append: bool },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PState {
    Running,
    Zombie(i32),
}

struct Proc {
    pid: u32,
    /// 0 means the shell (or an orphan the shell adopts).
    ppid: u32,
    task: usize,
    state: PState,
    name: String,
    aspace: Option<AddressSpace>,
    fds: Vec<Fd>,
    brk: u64,
    mmap_next: u64,
    mask: u64,
    cwd: String,
    /// Started by `run`: announces its exit on the console.
    verbose: bool,
}

struct Table {
    slots: [Option<Proc>; MAX_PROCS],
    pipes: PipeTable<32>,
    stdin: ConsoleIn,
    /// Processes the shell is waiting for (the foreground job).
    fg: Vec<u32>,
}

struct ConsoleIn {
    line: Vec<u8>,
    ready: VecDeque<u8>,
    eof: bool,
}

static TABLE: Global<Table> = Global::new(Table {
    slots: [const { None }; MAX_PROCS],
    pipes: PipeTable::new(),
    stdin: ConsoleIn { line: Vec::new(), ready: VecDeque::new(), eof: false },
    fg: Vec::new(),
});

fn table() -> &'static mut Table {
    // SAFETY: every caller runs with interrupts off (system calls, the scheduler hook) or wraps the call
    // in `without_interrupts`, and no result is held across a yield.
    unsafe { TABLE.get_irqs_off() }
}

fn current() -> Option<usize> {
    let task = sched::current_id();
    table().slots.iter().position(|s| s.as_ref().is_some_and(|p| p.task == task && p.state == PState::Running))
}

fn proc(me: usize) -> &'static mut Proc {
    table().slots[me].as_mut().expect("live process")
}

fn close_fd(t: &mut Table, fd: Fd) {
    match fd {
        Fd::PipeRead(id) => t.pipes.release(id, End::Read),
        Fd::PipeWrite(id) => t.pipes.release(id, End::Write),
        _ => {}
    }
}

fn retain_fd(t: &mut Table, fd: &Fd) {
    match fd {
        Fd::PipeRead(id) => {
            t.pipes.retain(*id, End::Read);
        }
        Fd::PipeWrite(id) => {
            t.pipes.retain(*id, End::Write);
        }
        _ => {}
    }
}

/// Ends the process in `slot`: closes its files, frees its memory, hands its children to the shell and
/// leaves a zombie holding the exit code. The caller must not be running on this process's page tables.
fn finish(slot: usize, code: i32) {
    let t = table();
    let Some(p) = t.slots[slot].as_mut() else { return };
    if matches!(p.state, PState::Zombie(_)) {
        return;
    }
    let fds = core::mem::take(&mut p.fds);
    let aspace = p.aspace.take();
    let pid = p.pid;
    let verbose = p.verbose;
    p.state = PState::Zombie(code);
    p.task = usize::MAX;
    for fd in fds {
        close_fd(t, fd);
    }
    if let Some(a) = aspace {
        a.destroy(&mut KernelMem);
    }
    for q in t.slots.iter_mut().flatten() {
        if q.ppid == pid {
            q.ppid = 0;
        }
    }
    if verbose {
        crate::log::print_raw(b"[user] task exited with code ");
        print_num(code as u64 & 0xFF);
        crate::log::print_raw(b"\n");
    }
}

/// Called by the scheduler when it frees the task of a process that never got to exit (killed from
/// outside). It runs in the timer interrupt, with another task's page tables loaded.
pub fn task_gone(task: usize) {
    if let Some(slot) = table().slots.iter().position(|s| s.as_ref().is_some_and(|p| p.task == task && p.state == PState::Running)) {
        finish(slot, 137);
    }
}

/// Ends the calling process with `code`. Never returns.
fn exit_current(code: i32) -> ! {
    arch::without_interrupts(|| {
        if let Some(me) = current() {
            // Leave the process's page tables before they are freed.
            // SAFETY: interrupts are off and the kernel PML4 maps everything the kernel is using.
            unsafe { sched::set_current_root(paging::kernel_root()) };
            finish(me, code);
        }
    });
    sched::exit()
}

fn find_free_slot(t: &Table) -> Option<usize> {
    t.slots.iter().position(Option::is_none)
}

#[derive(Debug)]
pub enum RunError {
    Unknown,
    NoMemory,
    /// Recovery or single-task mode: no scheduler and no `syscall` setup.
    NotReady,
    BadFile(&'static str),
}

fn load_error(e: LoadError) -> RunError {
    match e {
        LoadError::NoMemory => RunError::NoMemory,
        LoadError::BadArgs => RunError::BadFile("too many or too long arguments"),
        LoadError::TooBig => RunError::BadFile("program too big"),
        LoadError::Elf(_) => RunError::BadFile("not a static x86-64 ELF executable Aeon can run"),
    }
}

fn default_fds(stdin: Fd, stdout: Fd) -> Vec<Fd> {
    let mut v = alloc::vec![Fd::Null; MAX_FDS];
    v[0] = stdin;
    v[1] = stdout.clone();
    v[2] = stdout;
    v
}

/// Turns a shell-side description into a descriptor.
fn io_to_fd(io: Io, input: bool) -> Fd {
    match io {
        Io::Console if input => Fd::ConsoleIn,
        Io::Console => Fd::ConsoleOut,
        Io::Null => Fd::Null,
        Io::PipeRead(id) => Fd::PipeRead(id),
        Io::PipeWrite(id) => Fd::PipeWrite(id),
        Io::File { path, append } => {
            let is_usb = usbfs::is_usb(&path);
            Fd::File(OpenFile { path, offset: 0, flags: syscall::O_WRONLY | if append { syscall::O_APPEND } else { 0 }, is_usb })
        }
    }
}

/// Creates the process and its task from a loaded address space. On failure the address space is freed.
#[allow(clippy::too_many_arguments)]
fn start(name: &str, ppid: u32, aspace: AddressSpace, loaded: loader::Loaded, fds: Vec<Fd>, cwd: String, mask: u64, verbose: bool) -> Result<u32, RunError> {
    let t = table();
    let Some(slot) = find_free_slot(t) else {
        aspace.destroy(&mut KernelMem);
        return Err(RunError::NoMemory);
    };
    let first = UserStart::fresh(loaded.entry, loaded.rsp, aspace.root());
    let Some(task) = sched::spawn_user(name, &first) else {
        aspace.destroy(&mut KernelMem);
        return Err(RunError::NoMemory);
    };
    let pid = NEXT_PID.fetch_add(1, Ordering::Relaxed);
    t.slots[slot] = Some(Proc {
        pid,
        ppid,
        task,
        state: PState::Running,
        name: String::from(name),
        aspace: Some(aspace),
        fds,
        brk: HEAP_BASE,
        mmap_next: MMAP_BASE,
        mask,
        cwd,
        verbose,
    });
    Ok(pid)
}

fn new_space() -> Result<AddressSpace, RunError> {
    AddressSpace::new(&mut KernelMem, paging::kernel_entries(), paging::nx_enabled(), MAX_USER_PAGES).ok_or(RunError::NoMemory)
}

/// Starts the named built-in demo program as a process of the shell and returns its pid.
pub fn run(name: &str) -> Result<u32, RunError> {
    if !READY.load(Ordering::Relaxed) {
        return Err(RunError::NotReady);
    }
    let code = program(name).ok_or(RunError::Unknown)?;
    arch::without_interrupts(|| {
        let mut a = new_space()?;
        let loaded = match loader::load_flat(&mut a, &mut KernelMem, code, &[name.as_bytes()]) {
            Ok(l) => l,
            Err(e) => {
                a.destroy(&mut KernelMem);
                return Err(load_error(e));
            }
        };
        start(name, 0, a, loaded, default_fds(Fd::Null, Fd::ConsoleOut), String::from("/"), syscall::allow_mask(name), true)
    })
}

/// Starts the ELF program `file` (already read) with `args`, wired to `stdin` and `stdout`. The pipe ends
/// inside `stdin`/`stdout` belong to the new process, or are released here when it cannot start.
pub fn spawn_elf(name: &str, file: &[u8], args: &[&[u8]], cwd: &str, stdin: Io, stdout: Io) -> Result<u32, RunError> {
    let fds = default_fds(io_to_fd(stdin, true), io_to_fd(stdout, false));
    arch::without_interrupts(|| {
        // stdout and stderr are two descriptors for one pipe end: the pipe needs to know.
        if let Fd::PipeWrite(id) = &fds[1] {
            table().pipes.retain(*id, End::Write);
        }
        let fail = |fds: Vec<Fd>, e: RunError| {
            let t = table();
            for fd in fds {
                close_fd(t, fd);
            }
            Err(e)
        };
        if !READY.load(Ordering::Relaxed) {
            return fail(fds, RunError::NotReady);
        }
        let mut a = match new_space() {
            Ok(a) => a,
            Err(e) => return fail(fds, e),
        };
        let loaded = match loader::load_elf(&mut a, &mut KernelMem, file, args) {
            Ok(l) => l,
            Err(e) => {
                a.destroy(&mut KernelMem);
                return fail(fds, load_error(e));
            }
        };
        if find_free_slot(table()).is_none() {
            a.destroy(&mut KernelMem);
            return fail(fds, RunError::NoMemory);
        }
        // `start` gives the address space back itself when the task cannot be created.
        let keep = fds.clone();
        match start(name, 0, a, loaded, fds, String::from(cwd), syscall::STANDARD, false) {
            Ok(pid) => Ok(pid),
            Err(e) => fail(keep, e),
        }
    })
}

// ---------------------------------------------------------------------------------------------
// Shell-side API
// ---------------------------------------------------------------------------------------------

/// A new pipe, with one read end and one write end for the caller to hand to programs.
pub fn pipe_create() -> Option<usize> {
    arch::without_interrupts(|| table().pipes.create())
}

/// Puts `data` in a pipe (up to its capacity) and returns how much fitted.
pub fn pipe_fill(id: usize, data: &[u8]) -> usize {
    arch::without_interrupts(|| match table().pipes.write(id, data) {
        WriteResult::Wrote(n) => n,
        _ => 0,
    })
}

/// Gives back a pipe end the shell still holds (one that was not handed to a program).
pub fn pipe_release(id: usize, write_end: bool) {
    arch::without_interrupts(|| table().pipes.release(id, if write_end { End::Write } else { End::Read }));
}

/// Makes `pids` the foreground job: the shell waits for them, and the keyboard feeds their standard input.
pub fn set_foreground(pids: &[u32]) {
    arch::without_interrupts(|| {
        let t = table();
        t.fg = pids.to_vec();
        t.stdin = ConsoleIn { line: Vec::new(), ready: VecDeque::new(), eof: false };
    });
}

pub fn has_foreground() -> bool {
    arch::without_interrupts(|| !table().fg.is_empty())
}

fn is_foreground(pid: u32) -> bool {
    table().fg.contains(&pid)
}

/// The foreground job's exit status once every process of it has ended (and been collected), else `None`.
/// The status is the last stage's code.
pub fn foreground_done() -> Option<i32> {
    arch::without_interrupts(|| {
        let t = table();
        if t.fg.is_empty() {
            return None;
        }
        let mut last = 0;
        for pid in t.fg.clone() {
            match t.slots.iter().flatten().find(|p| p.pid == pid).map(|p| p.state) {
                Some(PState::Running) => return None,
                Some(PState::Zombie(c)) => last = c,
                None => {}
            }
        }
        for pid in core::mem::take(&mut t.fg) {
            if let Some(i) = t.slots.iter().position(|s| s.as_ref().is_some_and(|p| p.pid == pid)) {
                t.slots[i] = None;
            }
        }
        Some(last)
    })
}

/// Ends every process of the foreground job (Ctrl+C).
pub fn interrupt_foreground() {
    arch::without_interrupts(|| {
        let t = table();
        for pid in t.fg.clone() {
            if let Some(task) = t.slots.iter().flatten().find(|p| p.pid == pid && p.state == PState::Running).map(|p| p.task) {
                sched::kill(task);
            }
        }
    });
}

/// Collects shell-owned background processes that have ended: `(pid, name, exit code)`.
pub fn reap_background() -> Vec<(u32, String, i32)> {
    arch::without_interrupts(|| {
        let t = table();
        let mut out = Vec::new();
        for slot in t.slots.iter_mut() {
            let done = match slot {
                Some(p) if p.ppid == 0 && !is_foreground(p.pid) && !p.verbose => match p.state {
                    PState::Zombie(c) => Some((p.pid, p.name.clone(), c)),
                    PState::Running => None,
                },
                _ => None,
            };
            if let Some(d) = done {
                *slot = None;
                out.push(d);
            }
        }
        out
    })
}

/// Demo programs started by `run` clean up their own zombie slot here (the shell never waits for them).
pub fn reap_demos() {
    arch::without_interrupts(|| {
        let t = table();
        for s in t.slots.iter_mut() {
            if s.as_ref().is_some_and(|p| p.verbose && p.ppid == 0 && matches!(p.state, PState::Zombie(_))) {
                *s = None;
            }
        }
    });
}

/// Ends the process `pid`. Returns false if there is no such running process.
pub fn kill_pid(pid: u32) -> bool {
    arch::without_interrupts(|| {
        let t = table();
        match t.slots.iter().flatten().find(|p| p.pid == pid && p.state == PState::Running).map(|p| p.task) {
            Some(task) => sched::kill(task),
            None => false,
        }
    })
}

/// A key typed while a foreground program runs: line editing and echo, like a terminal in cooked mode.
pub fn console_key(c: Option<u8>, enter: bool, backspace: bool, eof: bool) {
    arch::without_interrupts(|| {
        let s = &mut table().stdin;
        if eof {
            if s.line.is_empty() {
                s.eof = true;
            } else {
                let line = core::mem::take(&mut s.line);
                s.ready.extend(line);
            }
        } else if enter {
            crate::log::print_raw(b"\n");
            let line = core::mem::take(&mut s.line);
            s.ready.extend(line);
            s.ready.push_back(b'\n');
        } else if backspace {
            if s.line.pop().is_some() {
                crate::log::print_raw(b"\x08 \x08");
            }
        } else if let Some(c) = c {
            if s.line.len() < MAX_STDIN_LINE && (0x20..0x7F).contains(&c) {
                s.line.push(c);
                crate::log::print_raw(&[c]);
            }
        }
    });
}

pub fn process_count() -> usize {
    arch::without_interrupts(|| table().slots.iter().flatten().filter(|p| p.state == PState::Running).count())
}

/// Calls `f(pid, ppid, name, task)` for every running process.
pub fn for_each_process(mut f: impl FnMut(u32, u32, &str, usize)) {
    arch::without_interrupts(|| {
        for p in table().slots.iter().flatten().filter(|p| p.state == PState::Running) {
            f(p.pid, p.ppid, &p.name, p.task);
        }
    });
}

// ---------------------------------------------------------------------------------------------
// System calls
// ---------------------------------------------------------------------------------------------

fn with_space<R>(me: usize, f: impl FnOnce(&mut AddressSpace, &mut KernelMem) -> R) -> Option<R> {
    let a = proc(me).aspace.as_mut()?;
    Some(f(a, &mut KernelMem))
}

fn copy_in(me: usize, va: u64, len: usize) -> Option<Vec<u8>> {
    let mut v = alloc::vec![0u8; len];
    with_space(me, |a, m| a.copy_in(m, va, &mut v))?.then_some(v)
}

fn copy_out(me: usize, va: u64, data: &[u8]) -> bool {
    with_space(me, |a, m| a.copy_out(m, va, data)).unwrap_or(false)
}

fn user_path(me: usize, va: u64) -> Result<String, u64> {
    let raw = with_space(me, |a, m| a.read_cstr(m, va, ramfs::MAX_PATH)).flatten().ok_or(errno(syscall::EFAULT))?;
    let s = String::from_utf8(raw).map_err(|_| errno(syscall::EINVAL))?;
    ramfs::resolve(&proc(me).cwd, &s).ok_or(errno(syscall::ENAMETOOLONG))
}

fn fs_error(e: ramfs::FsError) -> u64 {
    use ramfs::FsError as F;
    errno(match e {
        F::IsDir => syscall::EISDIR,
        F::NotFound => syscall::ENOENT,
        F::Exists => syscall::EEXIST,
        F::NotEmpty => syscall::ENOTEMPTY,
        F::NotDir => syscall::ENOTDIR,
        F::NoParent => syscall::ENOENT,
        F::BadName => syscall::EINVAL,
        _ => syscall::ENOSPC,
    })
}

fn usb_error(e: &str) -> u64 {
    errno(match e {
        "no such file or folder" | "no USB drive connected" => syscall::ENOENT,
        "is a directory" => syscall::EISDIR,
        "not a directory" => syscall::ENOTDIR,
        "already exists" => syscall::EEXIST,
        "folder is not empty" => syscall::ENOTEMPTY,
        "illegal name" => syscall::EINVAL,
        s if s.contains("read only") || s.contains("prohibited") => syscall::EACCES,
        _ => syscall::ENOSPC,
    })
}

fn fd_slot(me: usize) -> Option<usize> {
    proc(me).fds.iter().skip(3).position(|f| matches!(f, Fd::Null)).map(|i| i + 3)
}

fn path_is_dir(path: &str) -> bool {
    if usbfs::is_usb(path) {
        usbfs::hw_is_dir(usbfs::rel_path(path))
    } else {
        crate::shell::fs().is_dir(path)
    }
}

fn read_whole(path: &str, usb: bool) -> Result<Vec<u8>, u64> {
    if usb {
        usbfs::hw_read(usbfs::rel_path(path), syscall::MAX_IO as usize).map_err(usb_error)
    } else {
        match crate::shell::fs().read(path) {
            Some(d) => Ok(d.to_vec()),
            None if crate::shell::fs().is_dir(path) => Err(errno(syscall::EISDIR)),
            None => Err(errno(syscall::ENOENT)),
        }
    }
}

fn write_whole(path: &str, usb: bool, data: &[u8]) -> Result<(), u64> {
    if usb {
        usbfs::hw_write(usbfs::rel_path(path), data, true).map_err(usb_error)
    } else {
        crate::shell::fs().write(path, data).map_err(fs_error)
    }
}

fn sys_open(me: usize, path_ptr: u64, flags: u32) -> u64 {
    wait_fs();
    let path = match user_path(me, path_ptr) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let is_usb = usbfs::is_usb(&path);
    let wants_write = flags & syscall::O_ACCMODE != syscall::O_RDONLY || flags & (syscall::O_CREAT | syscall::O_TRUNC) != 0;
    if path_is_dir(&path) {
        if wants_write {
            return errno(syscall::EISDIR);
        }
        let Some(fd) = fd_slot(me) else { return errno(syscall::EMFILE) };
        proc(me).fds[fd] = Fd::Dir(path, 0);
        return fd as u64;
    }
    if flags & syscall::O_DIRECTORY != 0 {
        return errno(syscall::ENOTDIR);
    }
    if is_usb && wants_write {
        if let Err(e) = usbfs::check_write_safety(usbfs::rel_path(&path)) {
            return usb_error(e);
        }
    }
    let exists = read_whole(&path, is_usb).is_ok();
    if !exists {
        if flags & syscall::O_CREAT == 0 {
            return errno(syscall::ENOENT);
        }
        if let Err(e) = write_whole(&path, is_usb, b"") {
            return e;
        }
    } else if flags & syscall::O_TRUNC != 0 && flags & syscall::O_ACCMODE != syscall::O_RDONLY {
        if let Err(e) = write_whole(&path, is_usb, b"") {
            return e;
        }
    }
    let Some(fd) = fd_slot(me) else { return errno(syscall::EMFILE) };
    proc(me).fds[fd] = Fd::File(OpenFile { path, offset: 0, flags, is_usb });
    fd as u64
}

fn sys_close(me: usize, fd: u64) -> u64 {
    let p = proc(me);
    let Some(slot) = p.fds.get_mut(fd as usize) else { return errno(syscall::EBADF) };
    if matches!(slot, Fd::Null) {
        return errno(syscall::EBADF);
    }
    let old = core::mem::replace(slot, Fd::Null);
    close_fd(table(), old);
    0
}

fn sys_read(me: usize, fd: u64, buf: u64, count: u64) -> u64 {
    let count = count.min(syscall::MAX_IO) as usize;
    let Some(slot) = proc(me).fds.get(fd as usize).cloned() else { return errno(syscall::EBADF) };
    if count == 0 {
        return 0;
    }
    match slot {
        Fd::Null | Fd::ConsoleOut | Fd::PipeWrite(_) => errno(syscall::EBADF),
        Fd::Dir(..) => errno(syscall::EISDIR),
        Fd::ConsoleIn => {
            // Only the foreground job reads the keyboard; anything else sees end of file.
            let pid = proc(me).pid;
            loop {
                if !is_foreground(pid) {
                    return 0;
                }
                let s = &mut table().stdin;
                if !s.ready.is_empty() {
                    let n = count.min(s.ready.len());
                    let data: Vec<u8> = s.ready.drain(..n).collect();
                    return if copy_out(me, buf, &data) { n as u64 } else { errno(syscall::EFAULT) };
                }
                if s.eof {
                    s.eof = false;
                    return 0;
                }
                sched::yield_now();
            }
        }
        Fd::PipeRead(id) => {
            let mut tmp = alloc::vec![0u8; count.min(4096)];
            loop {
                match table().pipes.read(id, &mut tmp) {
                    ReadResult::Data(n) => return if copy_out(me, buf, &tmp[..n]) { n as u64 } else { errno(syscall::EFAULT) },
                    ReadResult::Eof => return 0,
                    ReadResult::WouldBlock => sched::yield_now(),
                }
            }
        }
        Fd::File(f) => {
            if f.flags & syscall::O_ACCMODE == syscall::O_WRONLY {
                return errno(syscall::EBADF);
            }
            wait_fs();
            let data = match read_whole(&f.path, f.is_usb) {
                Ok(d) => d,
                Err(e) => return e,
            };
            if f.offset >= data.len() {
                return 0;
            }
            let n = count.min(data.len() - f.offset);
            if !copy_out(me, buf, &data[f.offset..f.offset + n]) {
                return errno(syscall::EFAULT);
            }
            if let Some(Fd::File(of)) = proc(me).fds.get_mut(fd as usize) {
                of.offset = f.offset + n;
            }
            n as u64
        }
    }
}

fn sys_write(me: usize, fd: u64, buf: u64, count: u64) -> u64 {
    let Some(slot) = proc(me).fds.get(fd as usize).cloned() else { return errno(syscall::EBADF) };
    match slot {
        Fd::Null | Fd::ConsoleIn | Fd::PipeRead(_) => errno(syscall::EBADF),
        Fd::Dir(..) => errno(syscall::EISDIR),
        Fd::ConsoleOut => {
            let n = count.min(syscall::MAX_WRITE) as usize;
            let Some(mut data) = copy_in(me, buf, n) else { return errno(syscall::EFAULT) };
            syscall::sanitize_output(&mut data);
            crate::log::print_raw(&data);
            n as u64
        }
        Fd::PipeWrite(id) => {
            let n = count.min(4096) as usize;
            let Some(data) = copy_in(me, buf, n) else { return errno(syscall::EFAULT) };
            loop {
                match table().pipes.write(id, &data) {
                    WriteResult::Wrote(w) => return w as u64,
                    WriteResult::Broken => return errno(syscall::EPIPE),
                    WriteResult::WouldBlock => sched::yield_now(),
                }
            }
        }
        Fd::File(f) => {
            if f.flags & syscall::O_ACCMODE == syscall::O_RDONLY {
                return errno(syscall::EBADF);
            }
            let n = count.min(syscall::MAX_IO) as usize;
            let Some(data) = copy_in(me, buf, n) else { return errno(syscall::EFAULT) };
            wait_fs();
            let mut cur = read_whole(&f.path, f.is_usb).unwrap_or_default();
            let at = if f.flags & syscall::O_APPEND != 0 { cur.len() } else { f.offset.min(cur.len()) };
            if at + n > cur.len() {
                cur.resize(at + n, 0);
            }
            cur[at..at + n].copy_from_slice(&data);
            if let Err(e) = write_whole(&f.path, f.is_usb, &cur) {
                return e;
            }
            if let Some(Fd::File(of)) = proc(me).fds.get_mut(fd as usize) {
                of.offset = at + n;
            }
            n as u64
        }
    }
}

fn sys_lseek(me: usize, fd: u64, off: i64, whence: u32) -> u64 {
    let Some(Fd::File(f)) = proc(me).fds.get(fd as usize).cloned() else {
        let pipe_like = matches!(proc(me).fds.get(fd as usize), Some(Fd::PipeRead(_) | Fd::PipeWrite(_) | Fd::ConsoleIn | Fd::ConsoleOut));
        return errno(if pipe_like { syscall::ESPIPE } else { syscall::EBADF });
    };
    wait_fs();
    let len = read_whole(&f.path, f.is_usb).map(|d| d.len()).unwrap_or(0) as i64;
    let base = match whence {
        syscall::SEEK_SET => 0,
        syscall::SEEK_CUR => f.offset as i64,
        syscall::SEEK_END => len,
        _ => return errno(syscall::EINVAL),
    };
    match base.checked_add(off) {
        Some(n) if n >= 0 => {
            if let Some(Fd::File(of)) = proc(me).fds.get_mut(fd as usize) {
                of.offset = n as usize;
            }
            n as u64
        }
        _ => errno(syscall::EINVAL),
    }
}

fn sys_brk(me: usize, addr: u64) -> u64 {
    let cur = proc(me).brk;
    if addr == 0 || !(HEAP_BASE..=MMAP_BASE - PAGE).contains(&addr) {
        return cur;
    }
    let (cur_page, want_page) = (cur.next_multiple_of(PAGE), addr.next_multiple_of(PAGE));
    let ok = with_space(me, |a, m| {
        if want_page > cur_page {
            let mut v = cur_page;
            while v < want_page {
                if a.map(m, v, true, false).is_err() {
                    // Give back what this call took.
                    let mut u = cur_page;
                    while u < v {
                        a.unmap(m, u);
                        u += PAGE;
                    }
                    return false;
                }
                v += PAGE;
            }
        } else {
            let mut v = want_page;
            while v < cur_page {
                a.unmap(m, v);
                v += PAGE;
            }
        }
        true
    })
    .unwrap_or(false);
    if ok {
        proc(me).brk = addr;
        addr
    } else {
        cur
    }
}

fn sys_mmap(me: usize, len: u64, prot: u64, flags: u64) -> u64 {
    if len == 0 || len > 256 * 1024 * 1024 || flags & syscall::MAP_ANONYMOUS == 0 || flags & syscall::MAP_FIXED != 0 {
        return errno(syscall::EINVAL);
    }
    if prot & syscall::PROT_EXEC != 0 {
        return errno(syscall::EACCES);
    }
    let writable = prot & syscall::PROT_WRITE != 0;
    let size = len.next_multiple_of(PAGE);
    let base = proc(me).mmap_next;
    if base + size > USER_END - loader::STACK_PAGES * PAGE * 2 {
        return errno(syscall::ENOMEM);
    }
    let ok = with_space(me, |a, m| {
        let mut v = base;
        while v < base + size {
            if a.map(m, v, writable, false).is_err() {
                let mut u = base;
                while u < v {
                    a.unmap(m, u);
                    u += PAGE;
                }
                return false;
            }
            v += PAGE;
        }
        true
    })
    .unwrap_or(false);
    if ok {
        proc(me).mmap_next = base + size + PAGE; // one unmapped page between areas
        base
    } else {
        errno(syscall::ENOMEM)
    }
}

fn sys_munmap(me: usize, addr: u64, len: u64) -> u64 {
    if addr % PAGE != 0 || len == 0 || !vmspace::in_user_range(addr, len) {
        return errno(syscall::EINVAL);
    }
    with_space(me, |a, m| {
        let mut v = addr;
        while v < addr + len.next_multiple_of(PAGE) {
            a.unmap(m, v);
            v += PAGE;
        }
    });
    0
}

fn sys_pipe(me: usize, out: u64) -> u64 {
    let Some(r) = fd_slot(me) else { return errno(syscall::EMFILE) };
    let Some(w) = proc(me).fds.iter().skip(r + 1).position(|f| matches!(f, Fd::Null)).map(|i| i + r + 1) else {
        return errno(syscall::EMFILE);
    };
    let t = table();
    let Some(id) = t.pipes.create() else { return errno(syscall::EMFILE) };
    let mut b = [0u8; 8];
    b[..4].copy_from_slice(&(r as u32).to_le_bytes());
    b[4..].copy_from_slice(&(w as u32).to_le_bytes());
    if !copy_out(me, out, &b) {
        t.pipes.release(id, End::Read);
        t.pipes.release(id, End::Write);
        return errno(syscall::EFAULT);
    }
    let p = proc(me);
    p.fds[r] = Fd::PipeRead(id);
    p.fds[w] = Fd::PipeWrite(id);
    0
}

fn sys_dup2(me: usize, old: u64, new: u64) -> u64 {
    let (old, new) = (old as usize, new as usize);
    if new >= MAX_FDS || proc(me).fds.get(old).is_none_or(|f| matches!(f, Fd::Null)) {
        return errno(syscall::EBADF);
    }
    if old == new {
        return new as u64;
    }
    let copy = proc(me).fds[old].clone();
    retain_fd(table(), &copy);
    let prev = core::mem::replace(&mut proc(me).fds[new], copy);
    close_fd(table(), prev);
    new as u64
}

fn sys_fork(me: usize, f: &SysFrame) -> u64 {
    let t = table();
    let Some(slot) = find_free_slot(t) else { return errno(syscall::EAGAIN) };
    let parent = proc(me);
    let Some(child_as) = parent.aspace.as_ref().and_then(|a| a.duplicate(&mut KernelMem)) else { return errno(syscall::ENOMEM) };
    let mut gpr = [0u64; sched::GPR_COUNT];
    // Switch-frame order: r15 r14 r13 r12 r11 r10 r9 r8 rdi rsi rbp rbx rdx rcx rax.
    gpr[0] = f.r15;
    gpr[1] = f.r14;
    gpr[2] = f.r13;
    gpr[3] = f.r12;
    gpr[4] = f.rflags;
    gpr[5] = f.r10;
    gpr[6] = f.r9;
    gpr[7] = f.r8;
    gpr[8] = f.rdi;
    gpr[9] = f.rsi;
    gpr[10] = f.rbp;
    gpr[11] = f.rbx;
    gpr[12] = f.rdx;
    gpr[13] = f.rip;
    gpr[14] = 0; // the child's fork() returns 0
    let first = UserStart { rip: f.rip, rsp: f.rsp, rflags: f.rflags, gpr, cr3: child_as.root() };
    let Some(task) = sched::spawn_user(&parent.name, &first) else {
        child_as.destroy(&mut KernelMem);
        return errno(syscall::EAGAIN);
    };
    let fds = parent.fds.clone();
    for fd in &fds {
        retain_fd(t, fd);
    }
    let pid = NEXT_PID.fetch_add(1, Ordering::Relaxed);
    t.slots[slot] = Some(Proc {
        pid,
        ppid: parent.pid,
        task,
        state: PState::Running,
        name: parent.name.clone(),
        aspace: Some(child_as),
        fds,
        brk: parent.brk,
        mmap_next: parent.mmap_next,
        mask: parent.mask,
        cwd: parent.cwd.clone(),
        verbose: false,
    });
    pid as u64
}

fn read_user_ptr(me: usize, va: u64) -> Option<u64> {
    let b = copy_in(me, va, 8)?;
    Some(u64::from_le_bytes(b.try_into().ok()?))
}

fn sys_execve(me: usize, f: &mut SysFrame, path_ptr: u64, argv_ptr: u64) -> u64 {
    wait_fs();
    let path = match user_path(me, path_ptr) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let mut args: Vec<Vec<u8>> = Vec::new();
    if argv_ptr != 0 {
        for i in 0..crate::procargs::MAX_ARGS as u64 + 1 {
            let Some(p) = read_user_ptr(me, argv_ptr + i * 8) else { return errno(syscall::EFAULT) };
            if p == 0 {
                break;
            }
            if i as usize == crate::procargs::MAX_ARGS {
                return errno(syscall::E2BIG);
            }
            match with_space(me, |a, m| a.read_cstr(m, p, 512)).flatten() {
                Some(s) => args.push(s),
                None => return errno(syscall::EFAULT),
            }
        }
    }
    if args.is_empty() {
        args.push(path.clone().into_bytes());
    }
    let file = match read_whole(&path, usbfs::is_usb(&path)) {
        Ok(d) => d,
        Err(e) => return e,
    };
    let mut a = match new_space() {
        Ok(a) => a,
        Err(_) => return errno(syscall::ENOMEM),
    };
    let refs: Vec<&[u8]> = args.iter().map(|v| v.as_slice()).collect();
    let loaded = match loader::load_elf(&mut a, &mut KernelMem, &file, &refs) {
        Ok(l) => l,
        Err(e) => {
            a.destroy(&mut KernelMem);
            return errno(match e {
                LoadError::NoMemory => syscall::ENOMEM,
                LoadError::BadArgs => syscall::E2BIG,
                _ => syscall::ENOEXEC,
            });
        }
    };
    // Point of no return: the new image replaces the old one.
    let root = a.root();
    let p = proc(me);
    let old = p.aspace.replace(a);
    p.brk = HEAP_BASE;
    p.mmap_next = MMAP_BASE;
    p.mask = syscall::STANDARD;
    if let Some(n) = path.rsplit('/').next() {
        p.name = String::from(n);
    }
    // SAFETY: interrupts are off in a system call and the new PML4 holds the kernel's entries.
    unsafe { sched::set_current_root(root) };
    if let Some(o) = old {
        o.destroy(&mut KernelMem);
    }
    f.rip = loaded.entry;
    f.rsp = loaded.rsp;
    f.rflags = 0x202;
    f.rbx = 0;
    f.rbp = 0;
    f.r12 = 0;
    f.r13 = 0;
    f.r14 = 0;
    f.r15 = 0;
    f.rdi = 0;
    f.rsi = 0;
    f.rdx = 0;
    f.r8 = 0;
    f.r9 = 0;
    f.r10 = 0;
    0
}

fn sys_wait4(me: usize, pid: u64, status: u64, options: u64) -> u64 {
    let want = pid as i64;
    let my = proc(me).pid;
    loop {
        let t = table();
        let mut have_child = false;
        let mut hit: Option<(usize, u32, i32)> = None;
        for (i, s) in t.slots.iter().enumerate() {
            let Some(p) = s else { continue };
            if p.ppid != my || (want > 0 && p.pid as i64 != want) {
                continue;
            }
            have_child = true;
            if let PState::Zombie(c) = p.state {
                hit = Some((i, p.pid, c));
                break;
            }
        }
        if let Some((i, cpid, code)) = hit {
            if status != 0 {
                let st = ((code & 0xFF) << 8) as u32;
                if !copy_out(me, status, &st.to_le_bytes()) {
                    return errno(syscall::EFAULT);
                }
            }
            table().slots[i] = None;
            return cpid as u64;
        }
        if !have_child {
            return errno(syscall::ECHILD);
        }
        if options & syscall::WNOHANG != 0 {
            return 0;
        }
        sched::yield_now();
    }
}

fn sys_kill(me: usize, pid: u64, _sig: u64) -> u64 {
    let pid = pid as u32;
    let t = table();
    let Some(target) = t.slots.iter().flatten().find(|p| p.pid == pid && p.state == PState::Running).map(|p| p.task) else {
        return errno(syscall::ESRCH);
    };
    if target == proc(me).task {
        exit_current(137);
    }
    if sched::kill(target) { 0 } else { errno(syscall::ESRCH) }
}

fn sys_uname(me: usize, buf: u64) -> u64 {
    let mut b = [0u8; 65 * 6];
    let fields: [&str; 6] = ["Aeon", "aeon", crate::VERSION, "UEFI micro-OS", "x86_64", ""];
    for (i, f) in fields.iter().enumerate() {
        b[i * 65..i * 65 + f.len()].copy_from_slice(f.as_bytes());
    }
    if copy_out(me, buf, &b) { 0 } else { errno(syscall::EFAULT) }
}

fn sys_getcwd(me: usize, buf: u64, size: u64) -> u64 {
    let mut c = proc(me).cwd.clone().into_bytes();
    if c.is_empty() {
        c.push(b'/');
    }
    c.push(0);
    if (c.len() as u64) > size {
        return errno(syscall::ERANGE);
    }
    if copy_out(me, buf, &c) { c.len() as u64 } else { errno(syscall::EFAULT) }
}

fn sys_chdir(me: usize, path_ptr: u64) -> u64 {
    wait_fs();
    match user_path(me, path_ptr) {
        Ok(p) if path_is_dir(&p) => {
            proc(me).cwd = p;
            0
        }
        Ok(_) => errno(syscall::ENOTDIR),
        Err(e) => e,
    }
}

fn sys_mkdir(me: usize, path_ptr: u64) -> u64 {
    wait_fs();
    let p = match user_path(me, path_ptr) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let r = if usbfs::is_usb(&p) { usbfs::hw_mkdir(usbfs::rel_path(&p)).map_err(usb_error) } else { crate::shell::fs().mkdir(&p).map_err(fs_error) };
    r.map_or_else(|e| e, |_| 0)
}

fn sys_unlink(me: usize, path_ptr: u64) -> u64 {
    wait_fs();
    let p = match user_path(me, path_ptr) {
        Ok(p) => p,
        Err(e) => return e,
    };
    if usbfs::is_usb(&p) {
        if let Err(e) = usbfs::check_write_safety(usbfs::rel_path(&p)) {
            return usb_error(e);
        }
        return usbfs::hw_remove_file(usbfs::rel_path(&p)).map_or_else(usb_error, |_| 0);
    }
    if crate::shell::fs().remove(&p) {
        0
    } else if crate::shell::fs().is_dir(&p) {
        errno(syscall::EISDIR)
    } else {
        errno(syscall::ENOENT)
    }
}

fn sys_getdents(me: usize, fd: u64, buf: u64, count: u64) -> u64 {
    let Some(Fd::Dir(path, cursor)) = proc(me).fds.get(fd as usize).cloned() else { return errno(syscall::EBADF) };
    wait_fs();
    let mut names: Vec<(String, bool)> = Vec::new();
    if usbfs::is_usb(&path) {
        if let Err(e) = usbfs::hw_list(usbfs::rel_path(&path), |e| names.push((String::from(e.name()), e.is_dir))) {
            return usb_error(e);
        }
    } else {
        for (n, size) in crate::shell::fs().list(&path) {
            names.push((String::from(n), size.is_none()));
        }
    }
    let mut out = alloc::vec![0u8; count.min(4096) as usize];
    let mut used = 0;
    let mut idx = cursor;
    while idx < names.len() {
        let (n, d) = &names[idx];
        match syscall::write_dirent(&mut out[used..], idx as u64 + 1, idx as u64 + 1, if *d { syscall::DT_DIR } else { syscall::DT_REG }, n.as_bytes()) {
            Some(l) => used += l,
            None => break,
        }
        idx += 1;
    }
    if used == 0 && idx < names.len() {
        return errno(syscall::EINVAL);
    }
    if let Some(Fd::Dir(_, c)) = proc(me).fds.get_mut(fd as usize) {
        *c = idx;
    }
    if !copy_out(me, buf, &out[..used]) {
        return errno(syscall::EFAULT);
    }
    used as u64
}

fn sys_nanosleep(me: usize, req: u64) -> u64 {
    let Some(b) = copy_in(me, req, 16) else { return errno(syscall::EFAULT) };
    let sec = u64::from_le_bytes(b[0..8].try_into().unwrap_or([0; 8]));
    let nsec = u64::from_le_bytes(b[8..16].try_into().unwrap_or([0; 8]));
    if nsec >= 1_000_000_000 || sec > 3600 {
        return errno(syscall::EINVAL);
    }
    sched::sleep_ms((sec * 1000 + nsec / 1_000_000).max(1));
    0
}

fn sys_clock_gettime(me: usize, clk: u64, out: u64) -> u64 {
    let (sec, nsec) = match clk {
        syscall::CLOCK_REALTIME => match crate::rtc::now() {
            Some(t) => (t.unix_seconds().max(0) as u64, (crate::log::uptime_us() % 1_000_000) * 1000),
            None => (0, 0),
        },
        syscall::CLOCK_MONOTONIC => {
            let us = crate::log::uptime_us();
            (us / 1_000_000, (us % 1_000_000) * 1000)
        }
        _ => return errno(syscall::EINVAL),
    };
    let mut ts = [0u8; 16];
    ts[..8].copy_from_slice(&sec.to_le_bytes());
    ts[8..].copy_from_slice(&nsec.to_le_bytes());
    if copy_out(me, out, &ts) { 0 } else { errno(syscall::EFAULT) }
}

/// Rust side of the `syscall` entry. Runs with interrupts off on the task's kernel stack.
#[unsafe(no_mangle)]
extern "C" fn aeon_syscall_dispatch(f: &mut SysFrame) {
    let Some(me) = current() else {
        f.rax = syscall::ERR;
        return;
    };
    let Some(num) = syscall::decode(f.rax) else {
        f.rax = errno(syscall::ENOSYS);
        return;
    };
    if !syscall::permits(proc(me).mask, num) {
        f.rax = syscall::ERR;
        return;
    }
    let (a0, a1, a2, a3) = (f.rdi, f.rsi, f.rdx, f.r10);
    f.rax = match num {
        Number::Read => sys_read(me, a0, a1, a2),
        Number::Write => sys_write(me, a0, a1, a2),
        Number::Open => sys_open(me, a0, a1 as u32),
        Number::Close => sys_close(me, a0),
        Number::Lseek => sys_lseek(me, a0, a1 as i64, a2 as u32),
        Number::Mmap => sys_mmap(me, a1, a2, a3),
        Number::Munmap => sys_munmap(me, a0, a1),
        Number::Brk => sys_brk(me, a0),
        Number::Pipe => sys_pipe(me, a0),
        Number::Yield => {
            sched::yield_now();
            0
        }
        Number::Dup2 => sys_dup2(me, a0, a1),
        Number::Nanosleep => sys_nanosleep(me, a0),
        Number::GetPid => proc(me).pid as u64,
        Number::GetPpid => proc(me).ppid as u64,
        Number::Fork => sys_fork(me, f),
        Number::Execve => sys_execve(me, f, a0, a1),
        Number::Exit | Number::ExitGroup => exit_current(a0 as i32),
        Number::Wait4 => sys_wait4(me, a0, a1, a2),
        Number::Kill => sys_kill(me, a0, a1),
        Number::Uname => sys_uname(me, a0),
        Number::GetCwd => sys_getcwd(me, a0, a1),
        Number::Chdir => sys_chdir(me, a0),
        Number::Mkdir => sys_mkdir(me, a0),
        Number::Unlink => sys_unlink(me, a0),
        Number::GetDents64 => sys_getdents(me, a0, a1, a2),
        Number::ClockGetTime => sys_clock_gettime(me, a0, a1),
    };
}

fn print_num(mut v: u64) {
    let mut b = [0u8; 20];
    let mut i = b.len();
    loop {
        i -= 1;
        b[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    crate::log::print_raw(&b[i..]);
}

/// Called from the exception entry when the CPU faulted in ring 3: report it and end the process.
pub fn user_fault(f: &ExcFrame, name: &str, cr2: u64) -> ! {
    warn!("user", "task {} killed: {} at rip={:#x} error_code={:#x} cr2={:#x}", sched::current_id(), name, f.rip, f.error_code, cr2);
    crate::log::print_raw(b"[user] task killed: ");
    crate::log::print_raw(name.as_bytes());
    crate::log::print_raw(b"\n");
    exit_current(139)
}
