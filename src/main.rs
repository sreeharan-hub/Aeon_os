//! Aeon boot flow.
//!
//! firmware entry -> early init -> platform discovery -> ExitBootServices -> kernel environment
//! (GDT/IDT, page tables, CPU protections, own stack) -> `kernel_main`.

#![no_std]
#![no_main]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

use core::ffi::c_void;
use core::fmt::Write;
use core::sync::atomic::{AtomicBool, Ordering};

use aeon::acpi::{self, Rsdp};
use aeon::anim;
use aeon::arch;
use aeon::bootcount;
use aeon::bootinfo::{BootInfo, FbInfo, SecureBoot, MAX_REGIONS};
use aeon::bootopts::{self, BootOpts};
use aeon::cpuid;
use aeon::entropy;
use aeon::eventlog;
use aeon::fb::Console;
use aeon::gopmode;
use aeon::splash;
use aeon::tcg2;
use aeon::log;
use aeon::global::Global;
use aeon::logfmt::{self, Level};
use aeon::memmap;
use aeon::mm;
use aeon::paging;
use aeon::sched;
use aeon::shell;
use aeon::usermode;
use aeon::pixel::PixelFormat;
use aeon::uefi::{
    self, BootServices, Gop, GopModeInfo, Guid, Handle, InputKey, LoadedImage, Rng, RuntimeServices, Status, SystemTable,
    Tcg2,
};
use aeon::{debug, error, info, warn};

// ---------------------------------------------------------------------------------------------
// Kernel-owned state
// ---------------------------------------------------------------------------------------------

const KSTACK_SIZE: usize = 64 * 1024;

#[repr(C, align(4096))]
struct KernelStack([u8; KSTACK_SIZE]);

/// The kernel stack. Its lowest page is unmapped by `paging::build` as a guard page.
static KSTACK: Global<KernelStack> = Global::new(KernelStack([0; KSTACK_SIZE]));

/// Boot stages that move the splash progress bar: firmware-entry, early-init, platform-discovery,
/// security-init, exit-boot-services, kernel-env, kernel-memory, kernel-tasks.
const TOTAL_STAGES: usize = 8;

struct Stages {
    n: usize,
    items: [(&'static str, u64); 12],
}

static STAGES: Global<Stages> = Global::new(Stages { n: 0, items: [("", 0); 12] });

/// Busy-waits `us` microseconds on the TSC (usable before and after ExitBootServices).
fn wait_us(us: u64) {
    let t0 = arch::rdtsc();
    let ticks = log::tsc_hz() / 1_000_000 * us;
    while arch::rdtsc().wrapping_sub(t0) < ticks {
        arch::pause();
    }
}

/// Moves the splash bar to `target` in a few short steps, so it glides instead of jumping.
fn bar_to(target: u32) {
    let from = log::with_console(|c| c.bar()).unwrap_or(target);
    if target <= from {
        let _ = log::with_console(|c| c.progress(target));
        return;
    }
    // A short eased glide (40 ms in five frames); the wait is outside the console lock, so logging
    // is never blocked by the animation.
    for i in 1..=5u16 {
        let pos = anim::lerp(from, target, anim::ease_out_cubic(i * 200));
        let _ = log::with_console(|c| c.progress(pos));
        wait_us(8_000);
    }
}

fn stage(name: &'static str) {
    // SAFETY: single-threaded boot; the reference is dropped before logging.
    let done = unsafe {
        let s = STAGES.get();
        if s.n < s.items.len() {
            s.items[s.n] = (name, arch::rdtsc());
            s.n += 1;
        }
        s.n - 1
    };
    bar_to(splash::stage_permille(done, TOTAL_STAGES));
    info!("stage", "enter {}", name);
}

fn boot_info() -> &'static mut BootInfo {
    // SAFETY: only the boot flow in this file uses it, single-threaded, with short-lived references.
    unsafe { aeon::boot::info_mut() }
}

// ---------------------------------------------------------------------------------------------
// Firmware helpers (valid only until ExitBootServices)
// ---------------------------------------------------------------------------------------------

/// Measures the TSC against the firmware's `Stall` timer. Returns Hz, or 0 if implausible.
fn calibrate_tsc(bs: &BootServices) -> u64 {
    let t0 = arch::rdtsc();
    (bs.stall)(20_000);
    let hz = arch::rdtsc().wrapping_sub(t0).saturating_mul(50);
    if (200_000_000..=10_000_000_000).contains(&hz) {
        hz
    } else {
        0
    }
}

/// Lists the firmware's screen modes and lets the pure `gopmode::choose` pick one: the largest with a
/// linear framebuffer (so the console fills the panel), or the size asked for with `mode=WxH`.
fn pick_gop_mode(gop: &Gop, want: Option<(u32, u32)>) -> Option<u32> {
    const MAX_MODES: usize = 128;
    // SAFETY: `gop.mode` is set by firmware and valid for the protocol's lifetime.
    let (max_mode, current) = unsafe { ((*gop.mode).max_mode, (*gop.mode).mode) };
    let mut modes = [gopmode::ModeInfo { index: 0, width: 0, height: 0, linear: false }; MAX_MODES];
    let mut n = 0;
    for m in 0..max_mode.min(MAX_MODES as u32) {
        let mut size = 0usize;
        let mut info: *const GopModeInfo = core::ptr::null();
        if (gop.query_mode)(gop as *const Gop as *mut Gop, m, &mut size, &mut info) != uefi::SUCCESS || info.is_null() {
            continue;
        }
        // SAFETY: query_mode succeeded, so `info` points at a firmware-allocated GopModeInfo.
        let i = unsafe { &*info };
        modes[n] = gopmode::ModeInfo { index: m, width: i.horizontal_resolution, height: i.vertical_resolution, linear: i.pixel_format != 3 };
        n += 1;
    }
    gopmode::choose(&modes[..n], current, want)
}

fn init_gop(bs: &BootServices, want: Option<(u32, u32)>) -> Option<FbInfo> {
    let mut p: *mut c_void = core::ptr::null_mut();
    if (bs.locate_protocol)(&uefi::GOP_GUID, core::ptr::null_mut(), &mut p) != uefi::SUCCESS || p.is_null() {
        return None;
    }
    let gop = p as *mut Gop;
    // SAFETY: locate_protocol returned a live Gop interface.
    let g = unsafe { &*gop };
    if g.mode.is_null() {
        return None;
    }
    if let Some(m) = pick_gop_mode(g, want) {
        // SAFETY: reading the current mode number from the firmware-owned mode record.
        if m != unsafe { (*g.mode).mode } {
            let _ = (g.set_mode)(gop, m);
        }
    }
    // SAFETY: `mode` and `info` stay valid after set_mode; re-read them since the mode changed.
    let (mode, i) = unsafe { (&*g.mode, &*(*g.mode).info) };
    let format = match i.pixel_format {
        0 => PixelFormat::Rgb,
        1 => PixelFormat::Bgr,
        2 => PixelFormat::Mask { r: i.red_mask, g: i.green_mask, b: i.blue_mask },
        _ => return None,
    };
    Some(FbInfo {
        base: mode.frame_buffer_base,
        size: mode.frame_buffer_size,
        width: i.horizontal_resolution as usize,
        height: i.vertical_resolution as usize,
        stride: i.pixels_per_scan_line as usize,
        format,
    })
}

fn read_var_u8(rt: &RuntimeServices, name: &[u16]) -> Option<u8> {
    let mut size = 1usize;
    let mut data = 0u8;
    let mut attr = 0u32;
    let s = (rt.get_variable)(name.as_ptr(), &uefi::GLOBAL_VARIABLE_GUID, &mut attr, &mut size, &mut data);
    (s == uefi::SUCCESS && size == 1).then_some(data)
}

fn secure_boot_state(rt: &RuntimeServices) -> SecureBoot {
    const SECURE_BOOT: [u16; 11] = uefi::utf16(b"SecureBoot\0");
    const SETUP_MODE: [u16; 10] = uefi::utf16(b"SetupMode\0");
    if read_var_u8(rt, &SETUP_MODE) == Some(1) {
        return SecureBoot::SetupMode;
    }
    match read_var_u8(rt, &SECURE_BOOT) {
        Some(1) => SecureBoot::Enabled,
        Some(0) => SecureBoot::Disabled,
        _ => SecureBoot::Unknown,
    }
}

/// True if the firmware exposes the TCG2 protocol (a TPM 2.0 is present and measured boot is possible).
fn tpm2_present(bs: &BootServices) -> bool {
    let mut p: *mut c_void = core::ptr::null_mut();
    (bs.locate_protocol)(&uefi::TCG2_GUID, core::ptr::null_mut(), &mut p) == uefi::SUCCESS && !p.is_null()
}

/// Builds the boot random seed from the firmware RNG and RDRAND (the strong sources) plus timing.
/// Returns the seed and how many strong sources contributed. The seed is never logged.
fn gather_seed(bs: &BootServices, rt: &RuntimeServices, rdrand: bool) -> ([u8; entropy::SEED_LEN], u8, bool) {
    let mut pool = entropy::Pool::new();
    let mut strong = 0u8;

    let mut p: *mut c_void = core::ptr::null_mut();
    if (bs.locate_protocol)(&uefi::RNG_GUID, core::ptr::null_mut(), &mut p) == uefi::SUCCESS && !p.is_null() {
        let mut bytes = [0u8; 32];
        let rng = p as *mut Rng;
        // SAFETY: locate_protocol returned a live EFI_RNG_PROTOCOL; a null algorithm selects the default.
        let s = unsafe { ((*rng).get_rng)(rng, core::ptr::null(), bytes.len(), bytes.as_mut_ptr()) };
        if s == uefi::SUCCESS {
            pool.add(&bytes);
            strong += 1;
        } else {
            debug!("seed", "firmware RNG returned status={:#x}", s);
        }
        bytes.fill(0);
        core::hint::black_box(&bytes);
    }
    if rdrand {
        let words = [arch::rdrand64(), arch::rdrand64(), arch::rdrand64(), arch::rdrand64()];
        if words.iter().all(Option::is_some) {
            for w in words.into_iter().flatten() {
                pool.add_u64(w);
            }
            strong += 1;
        }
    }
    // A seed left by the previous boot adds history, so a machine without RNG hardware still varies.
    let mut old = [0u8; entropy::SEED_LEN];
    let persisted = read_var_bytes(rt, &SEED_NAME, &uefi::AEON_GUID, &mut old) == Some(entropy::SEED_LEN);
    if persisted {
        pool.add(&old);
    }
    old.fill(0);
    core::hint::black_box(&old);
    pool.add_u64(arch::rdtsc());
    pool.add_u64(boot_info().image_base);
    let seed = pool.finish();
    // The next boot's input is derived from this seed, never the seed itself.
    let mut next = entropy::Pool::new();
    next.add(b"aeon-next-seed");
    next.add(&seed);
    let mut next = next.finish();
    // An older build stored this variable with runtime access; attributes cannot change in place, so delete first.
    let _ = (rt.set_variable)(SEED_NAME.as_ptr(), &uefi::AEON_GUID, 0, 0, core::ptr::null());
    let _ = (rt.set_variable)(SEED_NAME.as_ptr(), &uefi::AEON_GUID, uefi::VAR_NV_BS, next.len(), next.as_ptr());
    next.fill(0);
    core::hint::black_box(&next);
    (seed, strong, persisted)
}

/// Reads one SHA-256 PCR through the TPM, or `None` if the TPM refuses or answers nonsense.
fn pcr_read(tcg: *mut Tcg2, pcr: u32) -> Option<[u8; 32]> {
    let mut cmd = [0u8; 20];
    let n = tcg2::build_pcr_read(&mut cmd, pcr);
    let mut resp = [0u8; 128];
    // SAFETY: `tcg` is a live EFI_TCG2_PROTOCOL; both blocks are valid for the sizes passed.
    let s = unsafe { ((*tcg).submit_command)(tcg, n as u32, cmd.as_ptr(), resp.len() as u32, resp.as_mut_ptr()) };
    if s != uefi::SUCCESS {
        debug!("tpm", "PCR_Read submit status={:#x}", s);
        return None;
    }
    tcg2::parse_pcr_read(&resp)
}

fn locate_tcg2(bs: &BootServices) -> Option<*mut Tcg2> {
    let mut p: *mut c_void = core::ptr::null_mut();
    ((bs.locate_protocol)(&uefi::TCG2_GUID, core::ptr::null_mut(), &mut p) == uefi::SUCCESS && !p.is_null()).then_some(p as *mut Tcg2)
}

/// Extends `data` into `pcr` as an `EV_IPL` event and reads the PCR back before and after. Returns
/// `(before, after, verified)`; `verified` means `after == SHA256(before || SHA256(data))`.
fn measure_event(tcg: *mut Tcg2, pcr: u32, data: &[u8]) -> Option<([u8; 32], [u8; 32], bool)> {
    let before = pcr_read(tcg, pcr)?;
    let mut ev = [0u8; tcg2::EVENT_OVERHEAD + 2 * 130];
    tcg2::build_event(&mut ev, pcr, tcg2::EV_IPL, data)?;
    // SAFETY: live protocol; `data` and `ev` stay valid for the call and `ev` is a well-formed
    // EFI_TCG2_EVENT built by `tcg2::build_event`.
    let s = unsafe { ((*tcg).hash_log_extend_event)(tcg, 0, data.as_ptr() as u64, data.len() as u64, ev.as_ptr()) };
    if s != uefi::SUCCESS {
        warn!("tpm", "HashLogExtendEvent pcr={} status={:#x}", pcr, s);
        return None;
    }
    let after = pcr_read(tcg, pcr)?;
    Some((before, after, tcg2::expected_extend(&before, data) == after))
}

fn log_pcr(label: &str, pcr: u32, before: &[u8; 32], after: &[u8; 32], verified: bool, bytes: usize) {
    let mut hex = [0u8; 64];
    tcg2::hex32(before, &mut hex);
    info!("tpm", "pcr{}_before={}", pcr, core::str::from_utf8(&hex).unwrap_or("?"));
    tcg2::hex32(after, &mut hex);
    info!("tpm", "pcr{}_after={}", pcr, core::str::from_utf8(&hex).unwrap_or("?"));
    info!("tpm", "measured {} into pcr={} event=EV_IPL bytes={} verified={}", label, pcr, bytes, verified);
    if !verified {
        error!("tpm", "PCR {} does not hold the expected value after the extend", pcr);
    }
}

/// Measured boot, part 1: the boot command line goes into PCR 12 (and the firmware event log), and
/// the PCR is read back to prove the extend happened. The firmware has already measured this image
/// itself into PCR 4.
fn measure_cmdline(bs: &BootServices, cmdline: &str) {
    let Some(tcg) = locate_tcg2(bs) else {
        info!("tpm", "measured boot unavailable: no TPM 2.0 (EFI_TCG2_PROTOCOL not found)");
        return;
    };
    boot_info().tpm_present = true;

    let (mut loc, mut last, mut truncated) = (0u64, 0u64, 0u8);
    // SAFETY: live protocol; format 2 is the crypto-agile log; the three out-pointers are valid.
    let s = unsafe { ((*tcg).get_event_log)(tcg, 2, &mut loc, &mut last, &mut truncated) };
    if s == uefi::SUCCESS {
        info!("tpm", "event log at {:#x}, last entry {:#x}, truncated={}", loc, last, truncated != 0);
    } else {
        warn!("tpm", "GetEventLog status={:#x}", s);
    }

    let mut data = [0u8; 2 * 130];
    let Some(dlen) = tcg2::utf16le_cmdline(cmdline, &mut data) else {
        warn!("tpm", "command line too long to measure");
        return;
    };
    match measure_event(tcg, tcg2::PCR_CMDLINE, &data[..dlen]) {
        Some((before, after, verified)) => {
            log_pcr("cmdline", tcg2::PCR_CMDLINE, &before, &after, verified, dlen);
            let b = boot_info();
            b.pcr12 = after;
            b.tpm_measured = verified;
        }
        None => warn!("tpm", "could not measure the command line into pcr={}", tcg2::PCR_CMDLINE),
    }
}

/// Measured boot, part 2: marks the point where Aeon leaves the firmware, in PCR 11, so a secret
/// sealed to PCR 11 only opens for a boot that got this far (the idea behind systemd's boot phases).
fn measure_phase(bs: &BootServices, phase: &str) {
    let Some(tcg) = locate_tcg2(bs) else { return };
    match measure_event(tcg, tcg2::PCR_PHASE, phase.as_bytes()) {
        Some((before, after, verified)) => {
            log_pcr("phase", tcg2::PCR_PHASE, &before, &after, verified, phase.len());
            boot_info().pcr11 = after;
        }
        None => warn!("tpm", "could not measure the boot phase into pcr={}", tcg2::PCR_PHASE),
    }
}

/// Reads the firmware event log back and replays it, then compares the replayed SHA-256 PCRs with
/// what the TPM reports. A mismatch means the log, the TPM or a measurement cannot be trusted.
fn verify_eventlog(bs: &BootServices) {
    let Some(tcg) = locate_tcg2(bs) else { return };
    let (mut loc, mut last, mut truncated) = (0u64, 0u64, 0u8);
    // SAFETY: live protocol; format 2 is the crypto-agile log; the three out-pointers are valid.
    let s = unsafe { ((*tcg).get_event_log)(tcg, 2, &mut loc, &mut last, &mut truncated) };
    if s != uefi::SUCCESS || loc == 0 || last < loc || truncated != 0 {
        warn!("evlog", "no usable event log (status={:#x}, truncated={})", s, truncated != 0);
        return;
    }
    let last_off = (last - loc) as usize;
    if last_off > 60 * 1024 {
        warn!("evlog", "event log larger than {} KiB, not replayed", 60);
        return;
    }
    // The last record starts at `last_off`; records are small, so this window covers it.
    let len = last_off + 2048;
    // SAFETY: the log lives in firmware memory that is identity-mapped before ExitBootServices;
    // the firmware reserves at least 64 KiB for it and `len` stays below that.
    let log = unsafe { core::slice::from_raw_parts(loc as *const u8, len) };
    let rp = match eventlog::replay(log, last_off) {
        Ok(r) => r,
        Err(e) => {
            warn!("evlog", "event log could not be parsed: {:?}", e);
            return;
        }
    };
    let mut compared = 0u8;
    let mut mismatched = 0u8;
    let mut hex = [0u8; 64];
    for pcr in (0..8u32).chain([tcg2::PCR_PHASE, tcg2::PCR_CMDLINE]) {
        let Some(actual) = pcr_read(tcg, pcr) else { continue };
        compared += 1;
        let want = &rp.pcrs[pcr as usize];
        if actual != *want {
            mismatched += 1;
            warn!("evlog", "pcr{} differs between the TPM and the replayed log", pcr);
        }
        #[cfg(feature = "dump-eventlog")]
        {
            tcg2::hex32(&actual, &mut hex);
            info!("evlog", "dump pcr{}={}", pcr, core::str::from_utf8(&hex).unwrap_or("?"));
        }
    }
    let _ = &mut hex;
    #[cfg(feature = "dump-eventlog")]
    {
        info!("evlog", "dump last_offset={} len={}", last_off, len);
        for (i, chunk) in log.chunks(40).enumerate() {
            let mut line = [0u8; 80];
            for (j, b) in chunk.iter().enumerate() {
                line[j * 2] = b"0123456789abcdef"[(b >> 4) as usize];
                line[j * 2 + 1] = b"0123456789abcdef"[(b & 15) as usize];
            }
            info!("evlog", "dump hex {:04} {}", i, core::str::from_utf8(&line[..chunk.len() * 2]).unwrap_or("?"));
        }
    }
    let b = boot_info();
    b.evlog_checked = true;
    b.evlog_records = rp.records as u32;
    b.evlog_compared = compared;
    b.evlog_mismatched = mismatched;
    info!(
        "evlog",
        "replayed {} records ({} extends); compared {} PCRs with the TPM, {} differ",
        rp.records,
        rp.extended,
        compared,
        mismatched
    );
}

/// Revocation metadata (SBAT, see the shim project's SBAT.md): lets firmware or tooling revoke old
/// Aeon builds by generation number without touching the signing key. Raise the number after a
/// security fix that old images must not be trusted without.
const SBAT_TEXT: &str = concat!(
    "sbat,1,SBAT Version,sbat,1,https://github.com/rhboot/shim/blob/main/SBAT.md\n",
    "aeon,1,Aeon,aeon,",
    env!("CARGO_PKG_VERSION"),
    ",-\n\0"
);
#[unsafe(link_section = ".sbat")]
#[used]
static SBAT: [u8; SBAT_TEXT.len()] = {
    let mut a = [0u8; SBAT_TEXT.len()];
    let b = SBAT_TEXT.as_bytes();
    let mut i = 0;
    while i < a.len() {
        a[i] = b[i];
        i += 1;
    }
    a
};

/// Reads the config file `\EFI\BOOT\aeon.cfg` from the volume the image was loaded from into `out`.
/// Returns the byte count, 0 when there is no file or no readable volume.
fn read_config_file(bs: &BootServices, device: Handle, out: &mut [u8]) -> usize {
    const PATH: [u16; 19] = uefi::utf16(b"\\EFI\\BOOT\\aeon.cfg\0");
    let mut fs: *mut c_void = core::ptr::null_mut();
    if device.is_null() || (bs.handle_protocol)(device, &uefi::SIMPLE_FS_GUID, &mut fs) != uefi::SUCCESS || fs.is_null() {
        return 0;
    }
    let fs = fs as *mut uefi::SimpleFs;
    let mut root: *mut uefi::File = core::ptr::null_mut();
    // SAFETY: handle_protocol returned a live EFI_SIMPLE_FILE_SYSTEM_PROTOCOL.
    if unsafe { ((*fs).open_volume)(fs, &mut root) } != uefi::SUCCESS || root.is_null() {
        return 0;
    }
    let mut file: *mut uefi::File = core::ptr::null_mut();
    // SAFETY: `root` is the volume's open root directory; PATH is a NUL-terminated UTF-16 string.
    let s = unsafe { ((*root).open)(root, &mut file, PATH.as_ptr(), uefi::FILE_MODE_READ, 0) };
    let mut n = 0;
    if s == uefi::SUCCESS && !file.is_null() {
        let mut size = out.len();
        // SAFETY: `file` is open for reading; `out` is valid for `size` bytes, which is its length.
        if unsafe { ((*file).read)(file, &mut size, out.as_mut_ptr()) } == uefi::SUCCESS {
            n = size.min(out.len());
        }
        // SAFETY: `file` was opened above and is closed exactly once.
        unsafe { ((*file).close)(file) };
    }
    // SAFETY: `root` was opened above and is closed exactly once.
    unsafe { ((*root).close)(root) };
    n
}

fn read_var_bytes(rt: &RuntimeServices, name: &[u16], guid: &Guid, buf: &mut [u8]) -> Option<usize> {
    let mut size = buf.len();
    let mut attr = 0u32;
    let s = (rt.get_variable)(name.as_ptr(), guid, &mut attr, &mut size, buf.as_mut_ptr());
    (s == uefi::SUCCESS).then_some(size.min(buf.len()))
}

const BOOT_COUNT_NAME: [u16; 14] = uefi::utf16(b"AeonBootCount\0");
const SEED_NAME: [u16; 9] = uefi::utf16(b"AeonSeed\0");

fn write_boot_count(rt: &RuntimeServices, value: u8) -> Status {
    (rt.set_variable)(BOOT_COUNT_NAME.as_ptr(), &uefi::AEON_GUID, uefi::VAR_NV_BS_RT, 1, &value)
}

/// Image base/size and the command line (UEFI load options), copied into `BootInfo`.
fn read_loaded_image(bs: &BootServices, image: Handle) {
    let b = boot_info();
    let mut li: *mut c_void = core::ptr::null_mut();
    if (bs.handle_protocol)(image, &uefi::LOADED_IMAGE_GUID, &mut li) == uefi::SUCCESS && !li.is_null() {
        // SAFETY: handle_protocol returned the LoadedImage of our own image handle.
        let li = unsafe { &*(li as *const LoadedImage) };
        b.image_base = li.image_base as u64;
        b.image_size = li.image_size;
        let units = (li.load_options_size as usize / 2).min(b.cmdline.len() - 1);
        if !li.load_options.is_null() {
            let opts = li.load_options as *const u16;
            let mut len = 0;
            for (i, out) in b.cmdline.iter_mut().enumerate().take(units) {
                // SAFETY: `units` 16-bit words lie within load_options_size bytes.
                let c = unsafe { opts.add(i).read_unaligned() };
                if c == 0 {
                    break; // a terminating NUL is not part of the options
                }
                *out = if (0x20..0x7F).contains(&c) { c as u8 } else { b'?' };
                len = i + 1;
            }
            b.cmdline_len = len;
        }
        // Options from `\EFI\BOOT\aeon.cfg` follow the load options, like loader.conf for systemd-boot.
        let mut raw = [0u8; 2048];
        let got = read_config_file(bs, li.device_handle, &mut raw);
        if got > 0 {
            let mut flat = [0u8; 120];
            let n = bootopts::flatten_config(&raw[..got], &mut flat);
            let mut at = b.cmdline_len;
            if n > 0 && at > 0 && at < b.cmdline.len() - 1 {
                b.cmdline[at] = b' ';
                at += 1;
            }
            let take = n.min(b.cmdline.len() - 1 - at);
            b.cmdline[at..at + take].copy_from_slice(&flat[..take]);
            b.cmdline_len = at + take;
            b.cfg_len = take;
        }
    }
}

fn find_acpi(st: &SystemTable) -> (u64, Rsdp) {
    let guids: [&Guid; 2] = [&uefi::ACPI_20_GUID, &uefi::ACPI_10_GUID];
    for g in guids {
        let Some(p) = st.config_table(g) else { continue };
        let p = p as *const u8;
        // SAFETY: an RSDP is at least 20 bytes and firmware allocates the ACPI 2.0 structure (36
        // bytes) whole; we first read only the 24-byte header to learn how much more is valid.
        let head = unsafe { core::slice::from_raw_parts(p, 24) };
        let Some((rev, len)) = acpi::peek_header(head) else {
            return (p as u64, Rsdp::Invalid);
        };
        let n = if rev == 0 { 24 } else { len.clamp(24, acpi::V2_MAX_LEN) };
        // SAFETY: `n` is the table length declared by the table itself, capped at V2_MAX_LEN; a
        // lying length is caught by the checksum, and the read stays within one firmware table.
        let all = unsafe { core::slice::from_raw_parts(p, n) };
        return (p as u64, acpi::validate(all));
    }
    (0, Rsdp::Invalid)
}

/// Fetches the memory map into `buf`. On `BUFFER_TOO_SMALL` the needed size is returned in `Err.1`.
fn fetch_map(bs: &BootServices, buf: *mut u8, cap: usize) -> Result<(usize, usize, usize), (Status, usize)> {
    let (mut size, mut key, mut dsz, mut dver) = (cap, 0usize, 0usize, 0u32);
    let s = (bs.get_memory_map)(&mut size, buf, &mut key, &mut dsz, &mut dver);
    if s == uefi::SUCCESS {
        Ok((size, key, dsz))
    } else {
        Err((s, size))
    }
}

fn alloc_pool(bs: &BootServices, size: usize) -> Option<*mut u8> {
    let mut p: *mut u8 = core::ptr::null_mut();
    ((bs.allocate_pool)(uefi::MEM_LOADER_DATA, size, &mut p) == uefi::SUCCESS && !p.is_null()).then_some(p)
}

/// Size of a variable in bytes, or `None` if it does not exist.
fn var_size(rt: &RuntimeServices, name: &[u16], guid: &Guid) -> Option<usize> {
    let mut size = 0usize;
    let mut attr = 0u32;
    let s = (rt.get_variable)(name.as_ptr(), guid, &mut attr, &mut size, core::ptr::null_mut());
    (s == uefi::BUFFER_TOO_SMALL || s == uefi::SUCCESS).then_some(size)
}

/// Publishes the rest of the systemd Boot Loader Interface, so `bootctl status` on Linux can show what
/// booted the machine and when. Claims no `LoaderFeatures`: Aeon does not implement those mechanisms.
fn publish_loader_interface(rt: &RuntimeServices) {
    const FW_TYPE: [u16; 19] = uefi::utf16(b"LoaderFirmwareType\0");
    const FW_INFO: [u16; 19] = uefi::utf16(b"LoaderFirmwareInfo\0");
    const IMAGE_ID: [u16; 22] = uefi::utf16(b"LoaderImageIdentifier\0");
    const EXEC: [u16; 19] = uefi::utf16(b"LoaderTimeExecUSec\0");
    const INFO: [u16; 11] = uefi::utf16(b"LoaderInfo\0");
    let b = boot_info();
    let rev = b.uefi_revision;
    let mut buf = logfmt::LineBuf::<48>::new();
    let _ = write!(buf, "UEFI {}.{}", rev >> 16, (rev & 0xFFFF) / 10);
    let t = buf.finish();
    set_loader_var(rt, &FW_TYPE, core::str::from_utf8(&t[..t.len() - 1]).unwrap_or("UEFI"));
    let mut buf = logfmt::LineBuf::<48>::new();
    let _ = write!(buf, "{} {:#x}", b.fw_vendor_str(), b.fw_revision);
    let t = buf.finish();
    set_loader_var(rt, &FW_INFO, core::str::from_utf8(&t[..t.len() - 1]).unwrap_or("?"));
    set_loader_var(rt, &IMAGE_ID, r"\EFI\BOOT\BOOTX64.EFI");
    let mut buf = logfmt::LineBuf::<24>::new();
    let _ = write!(buf, "{}", log::uptime_us());
    let t = buf.finish();
    set_loader_var(rt, &EXEC, core::str::from_utf8(&t[..t.len() - 1]).unwrap_or("0"));
    let published = [&INFO[..], &FW_TYPE[..], &FW_INFO[..], &IMAGE_ID[..], &EXEC[..]]
        .iter()
        .filter(|n| var_size(rt, n, &uefi::LOADER_GUID).is_some_and(|s| s > 0))
        .count() as u8;
    b.loader_vars = published;
    info!("loader", "Boot Loader Interface: {} of 5 variables published and read back", published);
}

/// Publishes an ASCII string as a volatile NUL-terminated UTF-16 variable under the systemd Boot
/// Loader Interface GUID, so Linux tools such as `bootctl` can see which loader started the system.
fn set_loader_var(rt: &RuntimeServices, name: &[u16], value: &str) {
    let mut buf = [0u16; 48];
    let n = value.len().min(buf.len() - 1);
    for (i, b) in value.bytes().take(n).enumerate() {
        buf[i] = b as u16;
    }
    let _ = (rt.set_variable)(name.as_ptr(), &uefi::LOADER_GUID, uefi::VAR_BS_RT, (n + 1) * 2, buf.as_ptr().cast());
}

/// Next key waiting in the firmware's input queue, as `(scan_code, unicode)`.
fn read_key(st: &SystemTable) -> Option<(u16, u16)> {
    if st.con_in.is_null() {
        return None;
    }
    let mut k = InputKey { scan_code: 0, unicode_char: 0 };
    // SAFETY: con_in is the firmware's text-input protocol, valid until ExitBootServices.
    let s = unsafe { ((*st.con_in).read_key_stroke)(st.con_in, &mut k) };
    if s != uefi::SUCCESS && s != uefi::NOT_READY {
        debug!("menu", "read_key_stroke status={:#x}", s);
    }
    (s == uefi::SUCCESS).then_some((k.scan_code, k.unicode_char))
}

fn read_var_u64(rt: &RuntimeServices, name: &[u16]) -> Option<u64> {
    let mut size = 8usize;
    let mut data = [0u8; 8];
    let mut attr = 0u32;
    let s = (rt.get_variable)(name.as_ptr(), &uefi::GLOBAL_VARIABLE_GUID, &mut attr, &mut size, data.as_mut_ptr());
    (s == uefi::SUCCESS && size <= 8).then(|| u64::from_le_bytes(data))
}

/// Asks the firmware to open its setup UI on the next boot, then resets. This is the OsIndications
/// mechanism behind `systemctl reboot --firmware-setup` and Windows' "UEFI Firmware Settings".
fn reboot_to_firmware_setup(rt: &RuntimeServices) {
    const SUPPORTED: [u16; 23] = uefi::utf16(b"OsIndicationsSupported\0");
    const INDICATIONS: [u16; 14] = uefi::utf16(b"OsIndications\0");
    const BOOT_TO_FW_UI: u64 = 1;
    if read_var_u64(rt, &SUPPORTED).unwrap_or(0) & BOOT_TO_FW_UI == 0 {
        warn!("menu", "firmware does not support booting to its setup UI");
        return;
    }
    let value = read_var_u64(rt, &INDICATIONS).unwrap_or(0) | BOOT_TO_FW_UI;
    let s = (rt.set_variable)(INDICATIONS.as_ptr(), &uefi::GLOBAL_VARIABLE_GUID, uefi::VAR_NV_BS_RT, 8, value.to_le_bytes().as_ptr());
    if s != uefi::SUCCESS {
        warn!("menu", "could not set OsIndications, status={:#x}", s);
        return;
    }
    info!("menu", "rebooting into firmware setup");
    (rt.reset_system)(uefi::RESET_COLD, 0, 0, core::ptr::null());
}

/// A small shell that runs while the firmware's boot services are still active. Input goes through
/// the firmware, so USB keyboards work here even though the kernel has no USB driver.
fn firmware_shell(st: &SystemTable, bs: &BootServices, rt: &RuntimeServices) {
    const PROMPT: &[u8] = b"fw> ";
    const HELP: &[u8] = b"commands: info version security tpm lastpanic log echo <text> | boot (continue) | reboot | shutdown\n";
    info!("fwsh", "firmware shell: boot services are still active, so USB keyboards work. `help` lists commands, `boot` continues.");
    let mut line = [0u8; 120];
    let mut len = 0usize;
    log::print_raw(PROMPT);
    loop {
        let _ = (bs.stall)(10_000);
        let Some((_, ch)) = read_key(st) else { continue };
        match ch {
            0x0D => {
                log::print_raw(b"\n");
                let text = core::str::from_utf8(&line[..len]).unwrap_or("").trim();
                match text {
                    "boot" | "exit" => return,
                    "" => {}
                    "help" | "?" => log::print_raw(HELP),
                    "reboot" => (rt.reset_system)(uefi::RESET_COLD, 0, 0, core::ptr::null()),
                    "shutdown" => (rt.reset_system)(uefi::RESET_SHUTDOWN, 0, 0, core::ptr::null()),
                    other => {
                        if !shell::run_firmware_command(other) {
                            log::print_raw(b"not available before the kernel starts; try `help`\n");
                        }
                    }
                }
                len = 0;
                log::print_raw(PROMPT);
            }
            0x08 if len > 0 => {
                len -= 1;
                log::print_raw(b"\x08 \x08");
            }
            0x20..=0x7E if len < line.len() => {
                line[len] = ch as u8;
                len += 1;
                log::print_raw(&[ch as u8]);
            }
            _ => {}
        }
    }
}

/// Dawn, the boot menu. It opens at once with "Boot Aeon" highlighted and a countdown (15 s by default).
/// Arrow keys move the highlight and restart the countdown; when it ends the highlighted entry runs, but only
/// "Boot Aeon" and "Recovery mode" may start by themselves (see `aeon::menulogic`). Enter runs the highlighted
/// entry at once; any other key stops the countdown. Keys are read through the firmware, so USB keyboards work.
/// Returns the possibly updated options.
fn boot_menu(st: &SystemTable, bs: &BootServices, rt: &RuntimeServices, mut opts: BootOpts) -> BootOpts {
    use aeon::menulogic::{Entry, Key, Menu, Outcome};
    if opts.menu_timeout == 0 || st.con_in.is_null() {
        return opts;
    }
    // SAFETY: con_in is a live protocol (checked non-null); resetting also drops stale keystrokes.
    unsafe { ((*st.con_in).reset)(st.con_in, false) };
    info!("menu", "menu open: boot starts in {} s (arrow keys choose another entry, Enter selects, any other key stops the countdown)", opts.menu_timeout);
    // Dawn owns the screen; log lines are held back and replayed when it closes.
    info!("menu", "options: Boot Aeon | Wi-Fi networks | Recovery mode | Firmware shell | Debug logging | Firmware setup | Restart | Power off");
    let hz = u128::from(log::tsc_hz().max(1));
    let mut menu = Menu::new(opts.menu_timeout);
    let mut last = aeon::arch::rdtsc();
    draw_menu(&menu, opts.debug);
    loop {
        let _ = (bs.stall)(20_000);
        // Time first, so a key press never hides time that has already passed.
        let now = aeon::arch::rdtsc();
        let ms = (u128::from(now.wrapping_sub(last)) * 1000 / hz).min(u128::from(u32::MAX)) as u32;
        last = now;
        let mut outcome = menu.tick(ms);
        if matches!(outcome, Outcome::Idle | Outcome::Footer) {
            if outcome == Outcome::Footer {
                show_footer(&menu);
            }
            let Some((scan, ch)) = read_key(st) else { continue };
            // UEFI scan codes: 0x01 up, 0x02 down, 0x05 home, 0x06 end.
            let key = match (scan, ch as u8 as char) {
                (0x01, _) => Key::Up,
                (0x02, _) => Key::Down,
                (0x05, _) => Key::Home,
                (0x06, _) => Key::End,
                (_, '\r') => Key::Enter,
                _ => Key::Other,
            };
            outcome = menu.key(key);
        }
        let entry = match outcome {
            Outcome::Idle => continue,
            Outcome::Footer => {
                show_footer(&menu);
                continue;
            }
            Outcome::List => {
                draw_menu(&menu, opts.debug);
                continue;
            }
            Outcome::Run(e) => e,
        };
        match entry {
            Entry::Boot => {
                leave_menu(true);
                return opts;
            }
            Entry::Wifi => {
                wifi_picker(st, bs);
                // SAFETY: con_in is a live protocol (checked at the top); drops keys typed during the exchange.
                unsafe { ((*st.con_in).reset)(st.con_in, false) };
                let _ = log::with_console(|c| c.menu_clear_body());
                menu.restart();
                last = aeon::arch::rdtsc();
                draw_menu(&menu, opts.debug);
                show_footer(&menu);
            }
            Entry::Recovery => {
                boot_info().recovery = true;
                info!("menu", "recovery mode selected: no timer, no scheduler");
                leave_menu(false);
                return opts;
            }
            Entry::FirmwareShell => {
                leave_menu(false);
                firmware_shell(st, bs, rt);
                return opts;
            }
            Entry::Debug => {
                opts.debug = !opts.debug;
                opts.quiet = false;
                log::set_max_level(if opts.debug { Level::Trace } else { Level::Debug });
                info!("menu", "debug logging {}", if opts.debug { "on" } else { "off" });
                draw_menu(&menu, opts.debug);
            }
            Entry::FirmwareSetup => reboot_to_firmware_setup(rt),
            Entry::Restart => (rt.reset_system)(uefi::RESET_COLD, 0, 0, core::ptr::null()),
            Entry::PowerOff => (rt.reset_system)(uefi::RESET_SHUTDOWN, 0, 0, core::ptr::null()),
        }
    }
}

/// Paints the whole menu: the list, the line about the highlighted entry and the countdown line.
fn draw_menu(menu: &aeon::menulogic::Menu, debug: bool) {
    let mut ssid = [0u8; aeon::wifihost::SSID_MAX];
    let mut status = aeon::menulogic::FooterBuf::new();
    if let Some(n) = aeon::wifihost::connected(&mut ssid) {
        let _ = write!(status, "Connected to {}. Enter to choose another network.", core::str::from_utf8(&ssid[..n]).unwrap_or("?"));
    }
    let entries = aeon::menulogic::ENTRIES.map(|e| {
        let desc = if e == aeon::menulogic::Entry::Wifi && !status.as_str().is_empty() { status.as_str() } else { e.description() };
        (e.label(debug), desc)
    });
    let mut footer = aeon::menulogic::FooterBuf::new();
    menu.footer(&mut footer);
    let _ = log::with_console(|c| c.menu_draw(&entries, menu.selected(), footer.as_str()));
}

/// Shows a line at the bottom of the boot menu.
fn menu_say(text: &str) {
    let _ = log::with_console(|c| c.menu_footer(text));
}

/// Next key from the firmware, waiting for it.
fn next_key(st: &SystemTable, bs: &BootServices) -> (u16, u16) {
    loop {
        let _ = (bs.stall)(10_000);
        if let Some(k) = read_key(st) {
            return k;
        }
    }
}

/// Password entry on the footer line, shown as stars. `None` when Esc cancels.
fn read_password(st: &SystemTable, bs: &BootServices, ssid: &str, pass: &mut [u8; 63]) -> Option<usize> {
    let mut plen = 0usize;
    loop {
        let mut f = aeon::menulogic::FooterBuf::new();
        let _ = write!(f, "Password for {}: ", ssid);
        for _ in 0..plen.min(24) {
            let _ = f.write_str("*");
        }
        if plen < 8 {
            let _ = f.write_str("  (8+ characters)");
        }
        menu_say(f.as_str());
        match next_key(st, bs) {
            (0x17, _) => return None,
            (_, 0x0D) if plen >= 8 => return Some(plen),
            (_, 0x08) => plen = plen.saturating_sub(1),
            (_, ch @ 0x20..=0x7E) if plen < pass.len() => {
                pass[plen] = ch as u8;
                plen += 1;
            }
            _ => {}
        }
    }
}

/// The Wi-Fi list opened from Dawn: asks the host helper (COM2) to scan, lets the reader pick a network with the
/// arrow keys, asks for the password and has the host join it. Nothing here touches a radio: in a VM the host's
/// Wi-Fi is the only Wi-Fi there is (see `aeon::wifihost`).
fn wifi_picker(st: &SystemTable, bs: &BootServices) {
    use aeon::wifihost::{link, Net, MAX_NETS};
    const LABEL: usize = 64;
    'rescan: loop {
        let _ = log::with_console(|c| c.menu_clear_body());
        menu_say("Looking for Wi-Fi networks...");
        let mut nets = [Net::EMPTY; MAX_NETS];
        let count = match link::scan(&mut nets) {
            Ok(n) => n,
            Err(e) => {
                info!("wifi", "scan failed: {}", e.as_str());
                menu_say(e.as_str());
                let _ = next_key(st, bs);
                return;
            }
        };
        info!("wifi", "host scan found {} network(s)", count);
        // Labels like "Hostel_WiFi   82%  locked", then "Scan again" and "Back".
        let mut labels = [[0u8; LABEL]; MAX_NETS];
        let mut lens = [0usize; MAX_NETS];
        for i in 0..count {
            let mut b = aeon::menulogic::FooterBuf::new();
            let n = &nets[i];
            let _ = write!(b, "{}   {}%{}{}", n.ssid(), n.signal, if n.secure { "  locked" } else { "  open" }, if n.current { "  (now)" } else { "" });
            let t = b.as_str().as_bytes();
            let l = t.len().min(LABEL);
            labels[i][..l].copy_from_slice(&t[..l]);
            lens[i] = l;
        }
        let total = count + 2;
        let mut sel = 0usize;
        let _ = log::with_console(|c| c.menu_clear_body());
        loop {
            let mut entries = [("", ""); MAX_NETS + 2];
            for i in 0..count {
                let label = core::str::from_utf8(&labels[i][..lens[i]]).unwrap_or("?");
                let desc = if nets[i].current {
                    "The host is on this network now. Enter to use it."
                } else if nets[i].secure {
                    "Press Enter, then type the Wi-Fi password."
                } else {
                    "Open network, no password."
                };
                entries[i] = (label, desc);
            }
            entries[count] = ("Scan again", "Look for networks again.");
            entries[count + 1] = ("Back", "Return to the boot menu.");
            let _ = log::with_console(|c| c.menu_draw(&entries[..total], sel, ""));
            menu_say(if count == 0 { "No networks found. Scan again or go back." } else { "Arrow keys choose a network, Enter selects, Esc goes back." });
            match next_key(st, bs) {
                (0x01, _) => sel = (sel + total - 1) % total,
                (0x02, _) => sel = (sel + 1) % total,
                (0x17, _) => return,
                (_, 0x0D) if sel == count => continue 'rescan,
                (_, 0x0D) if sel == count + 1 => return,
                (_, 0x0D) => {
                    let net = nets[sel];
                    let mut pass = [0u8; 63];
                    let plen = if net.secure && !net.current {
                        match read_password(st, bs, net.ssid(), &mut pass) {
                            Some(n) => n,
                            None => continue,
                        }
                    } else {
                        0
                    };
                    let mut f = aeon::menulogic::FooterBuf::new();
                    let _ = write!(f, "Connecting to {}...", net.ssid());
                    menu_say(f.as_str());
                    let result = link::connect(net.ssid(), &pass[..plen]);
                    pass.fill(0); // the password is not kept
                    let mut f = aeon::menulogic::FooterBuf::new();
                    match result {
                        Ok(_) => {
                            aeon::wifihost::set_connected(net.ssid());
                            info!("wifi", "host joined the chosen network");
                            let _ = write!(f, "Connected to {}. Aeon goes online when it starts.", net.ssid());
                            menu_say(f.as_str());
                            let _ = (bs.stall)(1_500_000);
                            return;
                        }
                        Err(e) => {
                            info!("wifi", "connect failed: {}", e.as_str());
                            let _ = write!(f, "Not connected: {}", e.as_str());
                            menu_say(f.as_str());
                            let _ = next_key(st, bs);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

/// Repaints only the line under the menu: the countdown, or the key help once it has stopped.
fn show_footer(menu: &aeon::menulogic::Menu) {
    let mut footer = aeon::menulogic::FooterBuf::new();
    menu.footer(&mut footer);
    let _ = log::with_console(|c| c.menu_footer(footer.as_str()));
}

/// Closes the menu. With `boot_on` the splash comes back (the normal boot carries on); otherwise
/// everything logged so far is shown on the normal log screen.
fn leave_menu(boot_on: bool) {
    if boot_on {
        let _ = log::with_console(|c| c.menu_resume());
        if log::with_console(|c| c.splash_active()).unwrap_or(false) {
            return; // the splash holds the log back, as in any normal boot
        }
    } else {
        let _ = log::with_console(|c| c.menu_end());
    }
    log::replay(|bytes| {
        let _ = log::with_console(|c| c.write_line(Level::Info, bytes));
    });
}

// ---------------------------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------------------------

// The firmware entry point cannot be `unsafe`; the firmware guarantees the system-table pointer.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "efiapi" fn efi_main(image: Handle, st_ptr: *mut SystemTable) -> Status {
    if st_ptr.is_null() {
        return uefi::INVALID_PARAMETER;
    }
    // SAFETY: the firmware passes a valid system table to the image entry point.
    let st = unsafe { &*st_ptr };
    if st.boot_services.is_null() || st.runtime_services.is_null() {
        return uefi::INVALID_PARAMETER;
    }
    // SAFETY: checked non-null above; boot services stay valid until ExitBootServices below.
    let bs = unsafe { &*st.boot_services };
    // SAFETY: checked non-null above; runtime services stay valid after ExitBootServices.
    let rt = unsafe { &*st.runtime_services };

    // ---- Stage 1: firmware entry ----
    log::init();
    // Stop the 5-minute firmware watchdog, which would otherwise reset the machine while we run.
    let wd = (bs.set_watchdog_timer)(0, 0, 0, core::ptr::null());
    let hz = calibrate_tsc(bs);
    let hz_ok = hz != 0;
    log::set_tsc_hz(if hz_ok { hz } else { 2_000_000_000 });
    let serial = log::enable_serial();
    let debugcon = log::enable_debugcon();
    // The firmware console is mirrored to the serial port by many firmwares, so use it as a log
    // sink only when there is no UART to read the log from.
    if !serial {
        log::attach_conout(st.con_out);
    }
    stage("firmware-entry");
    info!(
        "boot",
        "Aeon {} git={} arch=x86_64 profile={}",
        aeon::VERSION,
        aeon::GIT_HASH,
        if cfg!(debug_assertions) { "dev" } else { "release" }
    );
    info!("boot", "sinks serial={} debugcon={} watchdog_disabled={}", serial, debugcon, wd == uefi::SUCCESS);
    if hz_ok {
        info!("time", "tsc_hz={} source=uefi-stall", hz);
    } else {
        warn!("time", "tsc calibration implausible, assuming 2.0 GHz; timestamps are approximate");
    }
    if !serial {
        warn!("serial", "COM1 not present (scratch-register test failed); using screen and debugcon only");
    }

    // ---- Stage 2: early init ----
    stage("early-init");
    let leaves = arch::read_leaves();
    let feat = cpuid::decode(&leaves);
    boot_info().feat = feat;
    let vendor = arch::vendor();
    info!(
        "cpu",
        "vendor={} max_leaf={:#x} max_ext={:#x} phys_bits={} virt_bits={}",
        core::str::from_utf8(&vendor).unwrap_or("?"),
        leaves.max_basic,
        leaves.max_ext,
        feat.phys_bits,
        feat.virt_bits
    );
    info!(
        "cpu",
        "supported nx={} smep={} smap={} umip={} sse2={} rdrand={} rdseed={} cet_ss={} cet_ibt={} hypervisor={}",
        feat.nx,
        feat.smep,
        feat.smap,
        feat.umip,
        feat.sse2,
        feat.rdrand,
        feat.rdseed,
        feat.cet_ss,
        feat.cet_ibt,
        feat.hypervisor
    );
    let s = arch::cpu_state();
    info!(
        "cpu",
        "state-at-entry cr0.wp={} cr4.smep={} cr4.smap={} cr4.umip={} efer.nxe={}",
        s.cr0 & arch::CR0_WP != 0,
        s.cr4 & arch::CR4_SMEP != 0,
        s.cr4 & arch::CR4_SMAP != 0,
        s.cr4 & arch::CR4_UMIP != 0,
        s.efer & arch::EFER_NXE != 0
    );

    // Image info and boot options come first: they decide whether the screen shows the splash or the log.
    read_loaded_image(bs, image);
    let mut opts = bootopts::parse(boot_info().cmdline_str());
    if opts.quiet {
        log::set_max_level(Level::Warn);
    } else if opts.debug {
        log::set_max_level(Level::Trace);
    }

    // Boot counting: raise the counter now, clear it once the kernel is up. Enough failed boots in a
    // row switch this boot to recovery mode (log on screen, menu waits, no timer or scheduler).
    {
        let mut buf = [0u8; 4];
        let stored = read_var_bytes(rt, &BOOT_COUNT_NAME, &uefi::AEON_GUID, &mut buf);
        let count = bootcount::decode(stored.map(|n| &buf[..n]));
        let d = bootcount::decide(count, opts.nocount);
        let mut counting = d.write.is_some();
        if let Some(v) = d.write {
            let s = write_boot_count(rt, v);
            if s != uefi::SUCCESS {
                counting = false;
                warn!("bootcount", "could not store the boot counter, status={:#x}; counting is off for this boot", s);
            }
        }
        let b = boot_info();
        b.boot_count = count;
        b.count_enabled = counting;
        b.recovery = d.recovery || opts.recovery;
        b.nolapic = opts.nolapic;
        if b.recovery {
            opts.verbose = true;
            opts.menu_timeout = opts.menu_timeout.max(10);
        }
        info!("bootcount", "failed_boots={} counting={} recovery={}", count, counting, b.recovery);
    }

    // What did the previous boot say before it died? Shown once, then deleted.
    {
        let mut text = [0u8; aeon::panicrec::MAX_LEN];
        let n = aeon::panicrec::take(rt, &mut text);
        if n > 0 {
            let b = boot_info();
            b.last_panic[..n].copy_from_slice(&text[..n]);
            b.last_panic_len = n;
            warn!("bootcount", "previous boot ended in: {}", core::str::from_utf8(&text[..n]).unwrap_or("?"));
        }
    }

    let fb = init_gop(bs, opts.mode);
    if let Some(f) = fb {
        // SAFETY: GOP reported this linear framebuffer; it is mapped by the firmware and stays
        // valid across ExitBootServices, and nothing else draws to it after this point.
        if let Some(mut c) = unsafe { Console::new(&f) } {
            if opts.verbose || opts.recovery || boot_info().recovery {
                c.clear();
                c.fullscreen_log();
            } else {
                // Splash: the AEON wordmark fades in (ease-out, 180 ms) and a thin bar tracks the real boot stages.
                c.begin_splash();
                for step in 1..=10u16 {
                    c.fade_logo((u32::from(anim::ease_out_cubic(step * 100)) * 255 / 1000) as u8);
                    let _ = (bs.stall)(18_000);
                }
            }
            let (cols, rows) = c.size();
            log::detach_conout();
            log::attach_console(c);
            // The firmware console no longer shares the screen; hide its cursor.
            if !st.con_out.is_null() {
                // SAFETY: con_out is a live text-output protocol until ExitBootServices.
                unsafe { ((*st.con_out).enable_cursor)(st.con_out, false) };
            }
            info!("fb", "gop {}x{} stride={} fmt={:?} base={:#x} console={}x{}", f.width, f.height, f.stride, f.format, f.base, cols, rows);
        } else {
            warn!("fb", "framebuffer too small for console, using firmware console");
        }
    } else {
        warn!("fb", "no usable GOP framebuffer, using firmware console");
    }
    {
        let b = boot_info();
        b.fb = fb;
        b.tsc_hz = log::tsc_hz();
    }
    if boot_info().recovery {
        warn!("boot", "RECOVERY MODE: {} failed boot(s) in a row. Log on screen, no timer, no scheduler, single-task shell.", boot_info().boot_count);
    }
    if boot_info().cfg_len > 0 {
        info!("boot", "options from aeon.cfg: {} bytes; effective options \"{}\"", boot_info().cfg_len, boot_info().cmdline_str());
    }
    {
        // Catch the bar up with the stages that ran before the console existed.
        // SAFETY: single-threaded boot; plain read of the stage counter.
        let done = unsafe { (*STAGES.as_ptr()).n }.saturating_sub(1);
        let _ = log::with_console(|c| c.progress(splash::stage_permille(done, TOTAL_STAGES)));
    }

    // ---- Stage 3: platform discovery ----
    stage("platform-discovery");
    {
        let b = boot_info();
        b.fw_vendor_len = {
            // SAFETY: firmware_vendor is a NUL-terminated UTF-16 string supplied by the firmware.
            unsafe { uefi::utf16_to_ascii(st.firmware_vendor, &mut b.fw_vendor) }
        };
        b.fw_revision = st.firmware_revision;
        b.uefi_revision = st.hdr.revision;
        b.runtime_services = st.runtime_services as usize;

        let (rsdp, parsed) = find_acpi(st);
        b.acpi_rsdp = rsdp;
        b.acpi = parsed;
        b.smbios = st
            .config_table(&uefi::SMBIOS3_GUID)
            .or_else(|| st.config_table(&uefi::SMBIOS_GUID))
            .map_or(0, |p| p as u64);
        b.secure_boot = secure_boot_state(rt);
        const PK_NAME: [u16; 3] = uefi::utf16(b"PK\0");
        const DBX_NAME: [u16; 4] = uefi::utf16(b"dbx\0");
        b.sb_pk_enrolled = var_size(rt, &PK_NAME, &uefi::GLOBAL_VARIABLE_GUID).is_some_and(|n| n > 0);
        b.sb_dbx_bytes = var_size(rt, &DBX_NAME, &uefi::IMAGE_SECURITY_DB_GUID).unwrap_or(0) as u32;
    }
    {
        let b = boot_info();
        let rev = b.uefi_revision;
        info!(
            "fw",
            "vendor=\"{}\" fw_rev={:#x} uefi={}.{}.{}",
            b.fw_vendor_str(),
            b.fw_revision,
            rev >> 16,
            (rev & 0xFFFF) / 10,
            (rev & 0xFFFF) % 10
        );
        info!("fw", "secure_boot={} tpm2={}", b.secure_boot.name(), tpm2_present(bs));
        info!(
            "image",
            "base={:#x} size={:#x} cmdline=\"{}\"",
            b.image_base,
            b.image_size,
            b.cmdline_str()
        );
        match b.acpi {
            Rsdp::V2 { xsdt, revision } => {
                info!("acpi", "rsdp={:#x} rev={} xsdt={:#x} checksum=ok", b.acpi_rsdp, revision, xsdt)
            }
            Rsdp::V1 { rsdt } => info!("acpi", "rsdp={:#x} rev=0 rsdt={:#x} checksum=ok", b.acpi_rsdp, rsdt),
            Rsdp::Invalid if b.acpi_rsdp == 0 => warn!("acpi", "no RSDP in configuration table"),
            Rsdp::Invalid => warn!("acpi", "rsdp={:#x} failed validation, ignoring ACPI", b.acpi_rsdp),
        }
        if b.smbios != 0 {
            info!("smbios", "entry={:#x}", b.smbios);
        } else {
            debug!("smbios", "not present");
        }
    }

    // ---- Stage 4: security init (random seed, measured boot) ----
    stage("security-init");
    {
        let (seed, strong, persisted) = gather_seed(bs, rt, feat.rdrand);
        let b = boot_info();
        b.seed = seed;
        b.seed_sources = strong;
        b.seed_persisted = persisted;
        sched::set_canary(entropy::canary_from(&seed));
        if strong == 0 {
            warn!("seed", "no hardware entropy source (firmware RNG, RDRAND): seed rests on timing and history only");
        } else {
            info!("seed", "boot seed ready from {} hardware source(s); stack canary randomised", strong);
        }
        info!("seed", "seed from the previous boot mixed in: {}", persisted);
    }
    measure_cmdline(bs, boot_info().cmdline_str());

    // Boot menu (through the firmware console, so USB keyboards work here).
    opts = boot_menu(st, bs, rt, opts);
    let _ = opts;
    info!("menu", "boot continues after the menu");
    set_loader_var(rt, &uefi::utf16(b"LoaderInfo\0"), concat!("Aeon ", env!("CARGO_PKG_VERSION")));
    publish_loader_interface(rt);

    // ---- Stage 5: ExitBootServices ----
    stage("exit-boot-services");
    measure_phase(bs, if boot_info().recovery { tcg2::PHASE_RECOVERY } else { tcg2::PHASE_EXIT_BOOT });
    verify_eventlog(bs);
    // Size the buffer from the firmware's own answer, with slack because the allocation below (and
    // the first failed ExitBootServices) can add descriptors.
    let probe = fetch_map(bs, core::ptr::null_mut(), 0);
    let (need, desc_hint) = match probe {
        Err((uefi::BUFFER_TOO_SMALL, need)) => (need, 48usize),
        _ => (16 * 1024, 48usize),
    };
    let mut cap = need + 16 * desc_hint;
    let Some(mut buf) = alloc_pool(bs, cap) else {
        error!("mem", "cannot allocate {} bytes for the memory map", cap);
        return uefi::BUFFER_TOO_SMALL;
    };
    log::detach_conout();

    let mut attempts = 0;
    let (map_size, desc_size) = loop {
        let (size, key, dsz) = match fetch_map(bs, buf, cap) {
            Ok(v) => v,
            Err((uefi::BUFFER_TOO_SMALL, need)) if attempts < 4 => {
                cap = need + 16 * desc_hint;
                match alloc_pool(bs, cap) {
                    Some(p) => buf = p,
                    None => return uefi::BUFFER_TOO_SMALL,
                }
                attempts += 1;
                continue;
            }
            Err((s, _)) => {
                error!("mem", "GetMemoryMap failed status={:#x}", s);
                return s;
            }
        };
        // No logging and no allocation between GetMemoryMap and ExitBootServices: either could
        // change the map and invalidate `key`.
        let s = (bs.exit_boot_services)(image, key);
        if s == uefi::SUCCESS {
            break (size, dsz);
        }
        attempts += 1;
        if attempts > 4 {
            error!("boot", "ExitBootServices failed status={:#x} after {} attempts", s, attempts);
            return s;
        }
    };
    arch::disable_interrupts();

    // ---- Boot services are gone: from here on only our own code and runtime services ----
    {
        let b = boot_info();
        // SAFETY: `buf` holds `map_size` bytes written by GetMemoryMap; the pool allocation stays
        // valid because nothing frees it and the loader-data region is mapped by `paging::build`.
        let raw = unsafe { core::slice::from_raw_parts(buf, map_size) };
        let (n, dropped) = memmap::collect_regions(raw, desc_size, &mut b.regions);
        b.region_count = n;
        b.regions_dropped = dropped;
        b.mem = memmap::summarize(b.regions());
    }
    stage("kernel-env");
    info!("boot", "boot services exited after {} attempt(s)", attempts + 1);
    {
        let b = boot_info();
        info!(
            "mem",
            "descriptors={} kept={} dropped={} desc_size={}",
            map_size / desc_size,
            b.region_count,
            b.regions_dropped,
            desc_size
        );
        if b.regions_dropped > 0 {
            warn!("mem", "memory map has more than {} merged regions; {} were not recorded", MAX_REGIONS, b.regions_dropped);
        }
        let m = &b.mem;
        info!(
            "mem",
            "ram_mb={} usable_mb={} reclaimable_mb={} acpi_mb={} runtime_mb={} mmio_mb={} top_of_ram_mb={}",
            memmap::pages_to_mb(m.ram_pages),
            memmap::pages_to_mb(m.usable_pages),
            memmap::pages_to_mb(m.reclaimable_pages),
            memmap::pages_to_mb(m.acpi_pages),
            memmap::pages_to_mb(m.runtime_pages),
            memmap::pages_to_mb(m.mmio_pages),
            m.top_of_ram / (1024 * 1024)
        );
        info!("mem", "largest_free_block_mb={}", memmap::pages_to_mb(m.largest_usable_pages));
    }

    // Own execution environment. The order matters: NX must be enabled before page tables with NX
    // bits are loaded, and the IDT must exist before anything can fault.
    // SAFETY: single CPU, interrupts disabled, boot services exited; each step's own contract is
    // met by the arguments below.
    unsafe {
        let nx = feat.nx;
        if nx {
            arch::enable_nxe();
        }
        let wc = arch::has_pat();
        if wc {
            arch::program_pat_wc();
        }
        let stack = KSTACK.as_ptr() as u64;
        let stack_top = stack + KSTACK_SIZE as u64;
        arch::init_descriptors(stack_top);
        let stats = paging::build(boot_info(), nx, wc, stack);
        paging::activate();
        info!(
            "paging",
            "identity map active: pd_tables={} split_tables={} skipped={} nx={} wc={}",
            stats.pd_used,
            stats.pt_used,
            stats.skipped,
            nx,
            wc
        );
        let after = arch::enable_protections(&feat);
        info!(
            "cpu",
            "state-now cr0.wp={} cr4.smep={} cr4.smap={} cr4.umip={} efer.nxe={}",
            after.cr0 & arch::CR0_WP != 0,
            after.cr4 & arch::CR4_SMEP != 0,
            after.cr4 & arch::CR4_SMAP != 0,
            after.cr4 & arch::CR4_UMIP != 0,
            after.efer & arch::EFER_NXE != 0
        );
        info!("cpu", "gdt/tss/idt loaded, double-fault IST stack set, switching to kernel stack {:#x}..{:#x}", stack, stack_top);
        arch::call_on_stack(stack_top, kernel_main)
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "fault-stack")]
#[allow(unconditional_recursion)]
#[inline(never)]
fn overflow_stack(depth: u64) -> u64 {
    let pad = [depth as u8; 512];
    core::hint::black_box(&pad);
    overflow_stack(depth + 1) + pad[0] as u64
}

fn run_fault_tests() {
    #[cfg(feature = "fault-ud")]
    {
        warn!("test", "fault-ud: executing ud2");
        // SAFETY: deliberately raises #UD to exercise the exception path.
        unsafe { core::arch::asm!("ud2") };
    }
    #[cfg(feature = "fault-pf")]
    {
        warn!("test", "fault-pf: reading address 0");
        // SAFETY: deliberately faults; page 0 is unmapped.
        let v = unsafe { core::ptr::read_volatile(core::hint::black_box(core::ptr::null::<u8>())) };
        info!("test", "unexpectedly read {}", v);
    }
    #[cfg(feature = "fault-stack")]
    {
        warn!("test", "fault-stack: recursing until the guard page");
        let v = overflow_stack(0);
        info!("test", "unexpected return {}", v);
    }
    #[cfg(feature = "fault-wx")]
    {
        warn!("test", "fault-wx: writing to kernel code");
        let code = run_fault_tests as *const () as *mut u8;
        // SAFETY: deliberately faults; code pages are mapped read-only and CR0.WP is set.
        unsafe { code.write_volatile(0x90) };
        info!("test", "unexpected: code page was writable");
    }
    #[cfg(feature = "fault-nx")]
    {
        warn!("test", "fault-nx: executing from the stack");
        let mut buf = [0xC3u8; 16]; // `ret`
        // SAFETY: deliberately jumps into non-executable memory to provoke a #PF (instruction fetch).
        let f: extern "C" fn() = unsafe { core::mem::transmute(core::hint::black_box(buf.as_mut_ptr())) };
        f();
        info!("test", "unexpected: stack was executable");
    }
    #[cfg(feature = "fault-panic")]
    {
        warn!("test", "fault-panic: panicking");
        panic!("deliberate panic (fault-panic feature)");
    }
}

extern "C" fn kernel_main() -> ! {
    let rsp: u64;
    // SAFETY: reads the stack pointer register only.
    unsafe { core::arch::asm!("mov {}, rsp", out(reg) rsp, options(nomem, nostack, preserves_flags)) };
    info!("kernel", "kernel_main entered on kernel stack rsp={:#x}", rsp);

    stage("kernel-memory");
    // Memory management: physical frames first, then the heap carved from them.
    // SAFETY: boot services have exited, the map in BootInfo is final, and this runs once.
    let fs = unsafe { mm::init_frames(boot_info()) };
    info!(
        "mm",
        "frames: {} free ({} MB), largest run {} MB",
        fs.free,
        fs.free as u64 * 4 / 1024,
        fs.largest_run as u64 * 4 / 1024
    );
    // SAFETY: init_frames has run; this is the only heap initialisation.
    let heap = unsafe { mm::init_heap() };
    aeon::pci::scan();
    aeon::vblk::probe();
    aeon::xhci::init();
    aeon::touchpad_hw::init();
    aeon::ahci::probe();
    aeon::netdev::probe();
    if heap == 0 {
        error!("mm", "could not allocate a kernel heap");
    } else {
        info!("mm", "heap: {} KiB", heap / 1024);
    }

    // Tick source: PIC + PIT. If no interrupt arrives, fall back to polling instead of hanging.
    // Recovery mode never starts the timer: the point is to reach a shell with the fewest moving parts.
    let irq_mode = if boot_info().recovery {
        warn!("timer", "recovery mode: timer and scheduler are off; running the shell directly");
        false
    } else {
        // The local APIC timer is preferred; the PIT stays as the fallback (and with `nolapic`).
        // SAFETY: interrupts are off; the IDT routes vector 32 to the tick stub and 0xFF to the ignore stub.
        let lapic = !boot_info().nolapic && boot_info().feat.apic && unsafe { aeon::apic::start_timer(arch::TIMER_HZ) };
        if !lapic {
            // SAFETY: the IDT routes vector 32 to the tick stub (set up by `init_descriptors`).
            unsafe { arch::start_timer() };
        }
        arch::enable_interrupts();
        let t0 = arch::rdtsc();
        let limit = log::tsc_hz() / 3; // 330 ms
        while arch::ticks() < 3 && arch::rdtsc().wrapping_sub(t0) < limit {
            arch::pause();
        }
        let ok = arch::ticks() >= 3;
        if ok {
            if lapic {
                info!("timer", "lapic timer ok hz={} ticks_seen={}", arch::TIMER_HZ, arch::ticks());
            } else {
                info!("timer", "pit irq0 ok hz={} ticks_seen={}", arch::TIMER_HZ, arch::ticks());
            }
        } else {
            arch::disable_interrupts();
            warn!("timer", "no timer interrupt within 330 ms; no preemption, running the shell directly");
        }
        ok
    };

    stage("kernel-tasks");
    // SAFETY: reads the stage table written during boot; no writer is active.
    let stages = unsafe { &*STAGES.as_ptr() };
    let hz = log::tsc_hz();
    for w in stages.items[..stages.n].windows(2) {
        info!("stage", "{} took {} us", w[0].0, logfmt::ticks_to_us(w[1].1 - w[0].1, hz));
    }
    if stages.n > 0 {
        let last = stages.items[stages.n - 1];
        info!("stage", "{} took {} us", last.0, logfmt::ticks_to_us(arch::rdtsc() - last.1, hz));
    }
    info!("boot", "boot complete uptime_us={} mode={}", log::uptime_us(), if irq_mode { "preemptive" } else { "single-task" });

    // Finish the splash: bar to 100 %, a short hold so the eye sees it complete, a fade-out, then the shell.
    if log::with_console(|c| c.splash_active()).unwrap_or(false) {
        bar_to(1000);
        wait_us(120_000);
        for step in (0..10u16).rev() {
            let level = (u32::from(anim::ease_in_out(step * 100)) * 255 / 1000) as u8;
            let _ = log::with_console(|c| c.fade_splash(level));
            wait_us(15_000);
        }
        let _ = log::with_console(|c| c.end_splash());
    }

    run_fault_tests();

    // The kernel is up and the boot-time fault tests (if any) survived: this boot counts as good.
    {
        let b = boot_info();
        if b.runtime_services != 0 {
            // SAFETY: the firmware runtime-services table address was saved before ExitBootServices and
            // runtime memory is identity-mapped by `paging::build`, as for ResetSystem in `power`.
            let rt = unsafe { &*(b.runtime_services as *const RuntimeServices) };
            // The stored seed is boot-services-only: once the OS runs it must not be readable at all.
            let mut probe = [0u8; 32];
            let hidden = read_var_bytes(rt, &SEED_NAME, &uefi::AEON_GUID, &mut probe).is_none();
            probe.fill(0);
            core::hint::black_box(&probe);
            info!("seed", "stored seed hidden from runtime access: {}", hidden);
            if b.count_enabled {
                let s = write_boot_count(rt, 0);
                if s == uefi::SUCCESS {
                    info!("bootcount", "boot succeeded, counter cleared");
                } else {
                    warn!("bootcount", "could not clear the boot counter, status={:#x}", s);
                }
            }
        }
    }

    if !irq_mode {
        shell::shell_task(0);
        arch::halt_forever();
    }
    sched::init();
    // SAFETY: the GDT from `init_descriptors` is loaded, and interrupts are off for the duration of the call.
    arch::without_interrupts(|| unsafe { usermode::init() });
    for (name, entry) in [("status", shell::status_task as extern "C" fn(usize)), ("shell", shell::shell_task)] {
        match sched::spawn(name, entry, 0) {
            Some(id) => info!("sched", "task {} started as id {}", name, id),
            None => error!("sched", "cannot start task {}", name),
        }
    }
    // The boot thread becomes the idle task: it runs only when nothing else is ready.
    loop {
        arch::wait_for_interrupt();
    }
}

// ---------------------------------------------------------------------------------------------
// Panic
// ---------------------------------------------------------------------------------------------

static PANICKING: AtomicBool = AtomicBool::new(false);

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    arch::disable_interrupts();
    if PANICKING.swap(true, Ordering::SeqCst) {
        log::raw_panic_line("[AEON] nested panic, halting\n");
        arch::halt_forever();
    }
    // The panic may have happened inside the logger; release it and show every level.
    log::reset_after_fault();
    log::set_max_level(Level::Trace);
    match info.location() {
        Some(l) => error!("panic", "panicked at {}:{}:{}", l.file(), l.line(), l.column()),
        None => error!("panic", "panicked at unknown location"),
    }
    error!("panic", "{}", info.message());
    error!("panic", "cr2={:#x} cr3={:#x} system halted", arch::read_cr2(), arch::read_cr3());
    // The screen gets where and why, never addresses or registers (those stay on serial/debugcon).
    let mut where_ = logfmt::LineBuf::<96>::new();
    let mut why = logfmt::LineBuf::<160>::new();
    match info.location() {
        Some(l) => {
            let file = l.file().rsplit(['/', '\\']).next().unwrap_or("?");
            let _ = write!(where_, "panicked at {}:{}", file, l.line());
        }
        None => {
            let _ = write!(where_, "panicked at an unknown location");
        }
    }
    let _ = write!(why, "{}", info.message());
    let (where_line, why_line) = (where_.finish().trim_ascii_end(), why.finish().trim_ascii_end());
    {
        let mut rec = logfmt::LineBuf::<{ aeon::panicrec::MAX_LEN }>::new();
        let _ = write!(rec, "{} - {}", core::str::from_utf8(where_line).unwrap_or("?"), core::str::from_utf8(why_line).unwrap_or("?"));
        let r = rec.finish();
        aeon::panicrec::record(&r[..r.len() - 1]);
    }
    let _ = log::with_console(|c| c.panic_screen(&[where_line, why_line, b"Details are in the serial log."]));
    arch::halt_forever()
}
