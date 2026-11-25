//! AES-128, RFC 3394 key wrap and CCM (the mode behind WPA2's CCMP). Pure and host-testable.
//!
//! This is a plain table-free byte implementation: small and easy to check, not constant-time against
//! cache attacks. That trade-off is fine for an educational kernel; it is stated here so nobody mistakes it
//! for a hardened library.

use alloc::vec::Vec;

const fn gen_sbox() -> [u8; 256] {
    let mut sbox = [0u8; 256];
    let mut p: u8 = 1;
    let mut q: u8 = 1;
    loop {
        // p *= 3 in GF(2^8)
        p = p ^ (p << 1) ^ (if p & 0x80 != 0 { 0x1B } else { 0 });
        // q /= 3 in GF(2^8)
        q ^= q << 1;
        q ^= q << 2;
        q ^= q << 4;
        if q & 0x80 != 0 {
            q ^= 0x09;
        }
        let x = q ^ q.rotate_left(1) ^ q.rotate_left(2) ^ q.rotate_left(3) ^ q.rotate_left(4);
        sbox[p as usize] = x ^ 0x63;
        if p == 1 {
            break;
        }
    }
    sbox[0] = 0x63;
    sbox
}

const fn gen_inv(sbox: &[u8; 256]) -> [u8; 256] {
    let mut inv = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        inv[sbox[i] as usize] = i as u8;
        i += 1;
    }
    inv
}

const SBOX: [u8; 256] = gen_sbox();
const INV_SBOX: [u8; 256] = gen_inv(&SBOX);

fn xtime(x: u8) -> u8 {
    (x << 1) ^ if x & 0x80 != 0 { 0x1B } else { 0 }
}

fn gmul(mut a: u8, mut b: u8) -> u8 {
    let mut r = 0;
    while b != 0 {
        if b & 1 != 0 {
            r ^= a;
        }
        a = xtime(a);
        b >>= 1;
    }
    r
}

pub const BLOCK: usize = 16;

#[derive(Clone)]
pub struct Aes128 {
    round_keys: [[u8; 16]; 11],
}

impl Aes128 {
    pub fn new(key: &[u8; 16]) -> Self {
        let mut w = [[0u8; 4]; 44];
        for i in 0..4 {
            w[i].copy_from_slice(&key[i * 4..i * 4 + 4]);
        }
        let mut rcon: u8 = 1;
        for i in 4..44 {
            let mut t = w[i - 1];
            if i % 4 == 0 {
                t = [SBOX[t[1] as usize] ^ rcon, SBOX[t[2] as usize], SBOX[t[3] as usize], SBOX[t[0] as usize]];
                rcon = xtime(rcon);
            }
            for j in 0..4 {
                w[i][j] = w[i - 4][j] ^ t[j];
            }
        }
        let mut round_keys = [[0u8; 16]; 11];
        for (r, rk) in round_keys.iter_mut().enumerate() {
            for c in 0..4 {
                rk[c * 4..c * 4 + 4].copy_from_slice(&w[r * 4 + c]);
            }
        }
        Self { round_keys }
    }

    fn add_key(s: &mut [u8; 16], rk: &[u8; 16]) {
        for i in 0..16 {
            s[i] ^= rk[i];
        }
    }

    pub fn encrypt_block(&self, block: &[u8; 16]) -> [u8; 16] {
        let mut s = *block;
        Self::add_key(&mut s, &self.round_keys[0]);
        for (round, round_key) in self.round_keys.iter().enumerate().skip(1) {
            for b in s.iter_mut() {
                *b = SBOX[*b as usize];
            }
            // ShiftRows (state is column-major: s[col*4 + row]).
            let t = s;
            for c in 0..4 {
                for r in 0..4 {
                    s[c * 4 + r] = t[((c + r) % 4) * 4 + r];
                }
            }
            if round != 10 {
                for c in 0..4 {
                    let a = [s[c * 4], s[c * 4 + 1], s[c * 4 + 2], s[c * 4 + 3]];
                    s[c * 4] = xtime(a[0]) ^ (xtime(a[1]) ^ a[1]) ^ a[2] ^ a[3];
                    s[c * 4 + 1] = a[0] ^ xtime(a[1]) ^ (xtime(a[2]) ^ a[2]) ^ a[3];
                    s[c * 4 + 2] = a[0] ^ a[1] ^ xtime(a[2]) ^ (xtime(a[3]) ^ a[3]);
                    s[c * 4 + 3] = (xtime(a[0]) ^ a[0]) ^ a[1] ^ a[2] ^ xtime(a[3]);
                }
            }
            Self::add_key(&mut s, round_key);
        }
        s
    }

    pub fn decrypt_block(&self, block: &[u8; 16]) -> [u8; 16] {
        let mut s = *block;
        Self::add_key(&mut s, &self.round_keys[10]);
        for round in (0..10).rev() {
            // InvShiftRows
            let t = s;
            for c in 0..4 {
                for r in 0..4 {
                    s[((c + r) % 4) * 4 + r] = t[c * 4 + r];
                }
            }
            for b in s.iter_mut() {
                *b = INV_SBOX[*b as usize];
            }
            Self::add_key(&mut s, &self.round_keys[round]);
            if round != 0 {
                for c in 0..4 {
                    let a = [s[c * 4], s[c * 4 + 1], s[c * 4 + 2], s[c * 4 + 3]];
                    s[c * 4] = gmul(a[0], 14) ^ gmul(a[1], 11) ^ gmul(a[2], 13) ^ gmul(a[3], 9);
                    s[c * 4 + 1] = gmul(a[0], 9) ^ gmul(a[1], 14) ^ gmul(a[2], 11) ^ gmul(a[3], 13);
                    s[c * 4 + 2] = gmul(a[0], 13) ^ gmul(a[1], 9) ^ gmul(a[2], 14) ^ gmul(a[3], 11);
                    s[c * 4 + 3] = gmul(a[0], 11) ^ gmul(a[1], 13) ^ gmul(a[2], 9) ^ gmul(a[3], 14);
                }
            }
        }
        s
    }
}

const WRAP_IV: [u8; 8] = [0xA6; 8];

/// RFC 3394 AES key wrap. `plain` must be a multiple of 8 bytes, at least 16.
pub fn key_wrap(kek: &[u8; 16], plain: &[u8]) -> Option<Vec<u8>> {
    if plain.len() < 16 || plain.len() % 8 != 0 {
        return None;
    }
    let aes = Aes128::new(kek);
    let n = plain.len() / 8;
    let mut a = WRAP_IV;
    let mut r: Vec<[u8; 8]> = plain.chunks(8).map(|c| c.try_into().unwrap()).collect();
    for j in 0..6u64 {
        for (i, ri) in r.iter_mut().enumerate() {
            let mut b = [0u8; 16];
            b[..8].copy_from_slice(&a);
            b[8..].copy_from_slice(ri);
            let b = aes.encrypt_block(&b);
            let t = (n as u64) * j + (i as u64) + 1;
            a.copy_from_slice(&b[..8]);
            for (k, tb) in t.to_be_bytes().iter().enumerate() {
                a[k] ^= tb;
            }
            ri.copy_from_slice(&b[8..]);
        }
    }
    let mut out = Vec::with_capacity(plain.len() + 8);
    out.extend_from_slice(&a);
    for ri in &r {
        out.extend_from_slice(ri);
    }
    Some(out)
}

/// RFC 3394 AES key unwrap. Returns `None` if the integrity value does not match (wrong key or tampering).
pub fn key_unwrap(kek: &[u8; 16], cipher: &[u8]) -> Option<Vec<u8>> {
    if cipher.len() < 24 || cipher.len() % 8 != 0 {
        return None;
    }
    let aes = Aes128::new(kek);
    let n = cipher.len() / 8 - 1;
    let mut a: [u8; 8] = cipher[..8].try_into().unwrap();
    let mut r: Vec<[u8; 8]> = cipher[8..].chunks(8).map(|c| c.try_into().unwrap()).collect();
    for j in (0..6u64).rev() {
        for i in (0..n).rev() {
            let t = (n as u64) * j + (i as u64) + 1;
            let mut x = a;
            for (k, tb) in t.to_be_bytes().iter().enumerate() {
                x[k] ^= tb;
            }
            let mut b = [0u8; 16];
            b[..8].copy_from_slice(&x);
            b[8..].copy_from_slice(&r[i]);
            let b = aes.decrypt_block(&b);
            a.copy_from_slice(&b[..8]);
            r[i].copy_from_slice(&b[8..]);
        }
    }
    if a != WRAP_IV {
        return None;
    }
    Some(r.concat())
}

/// CCM with a 13-byte nonce (L = 2) and an 8-byte tag: exactly the parameters of 802.11 CCMP.
pub const CCM_TAG: usize = 8;
pub const CCM_NONCE: usize = 13;

fn ccm_mac(aes: &Aes128, nonce: &[u8; CCM_NONCE], aad: &[u8], msg: &[u8]) -> [u8; 16] {
    let mut b0 = [0u8; 16];
    b0[0] = (if aad.is_empty() { 0 } else { 0x40 }) | ((((CCM_TAG - 2) / 2) as u8) << 3) | 1;
    b0[1..14].copy_from_slice(nonce);
    b0[14..16].copy_from_slice(&(msg.len() as u16).to_be_bytes());
    let mut x = aes.encrypt_block(&b0);
    let feed = |data: &[u8], x: &mut [u8; 16]| {
        for chunk in data.chunks(16) {
            for (i, b) in chunk.iter().enumerate() {
                x[i] ^= b;
            }
            *x = aes.encrypt_block(x);
        }
    };
    if !aad.is_empty() {
        let mut head = Vec::with_capacity(2 + aad.len());
        head.extend_from_slice(&(aad.len() as u16).to_be_bytes());
        head.extend_from_slice(aad);
        feed(&head, &mut x);
    }
    feed(msg, &mut x);
    x
}

fn ccm_ctr(aes: &Aes128, nonce: &[u8; CCM_NONCE], counter: u16) -> [u8; 16] {
    let mut a = [0u8; 16];
    a[0] = 1;
    a[1..14].copy_from_slice(nonce);
    a[14..16].copy_from_slice(&counter.to_be_bytes());
    aes.encrypt_block(&a)
}

/// Returns ciphertext followed by the 8-byte tag.
pub fn ccm_encrypt(key: &[u8; 16], nonce: &[u8; CCM_NONCE], aad: &[u8], plain: &[u8]) -> Vec<u8> {
    let aes = Aes128::new(key);
    let mac = ccm_mac(&aes, nonce, aad, plain);
    let mut out = Vec::with_capacity(plain.len() + CCM_TAG);
    for (i, chunk) in plain.chunks(16).enumerate() {
        let s = ccm_ctr(&aes, nonce, (i + 1) as u16);
        out.extend(chunk.iter().zip(s.iter()).map(|(p, k)| p ^ k));
    }
    let s0 = ccm_ctr(&aes, nonce, 0);
    out.extend((0..CCM_TAG).map(|i| mac[i] ^ s0[i]));
    out
}

/// `data` is ciphertext followed by the tag. Returns the plaintext, or `None` if the tag does not verify.
pub fn ccm_decrypt(key: &[u8; 16], nonce: &[u8; CCM_NONCE], aad: &[u8], data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < CCM_TAG {
        return None;
    }
    let (ct, tag) = data.split_at(data.len() - CCM_TAG);
    let aes = Aes128::new(key);
    let mut plain = Vec::with_capacity(ct.len());
    for (i, chunk) in ct.chunks(16).enumerate() {
        let s = ccm_ctr(&aes, nonce, (i + 1) as u16);
        plain.extend(chunk.iter().zip(s.iter()).map(|(c, k)| c ^ k));
    }
    let mac = ccm_mac(&aes, nonce, aad, &plain);
    let s0 = ccm_ctr(&aes, nonce, 0);
    let mut diff = 0u8;
    for i in 0..CCM_TAG {
        diff |= (mac[i] ^ s0[i]) ^ tag[i];
    }
    if diff == 0 { Some(plain) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2).map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap()).collect()
    }

    #[test]
    fn sbox_is_the_standard_one() {
        assert_eq!(SBOX[0x00], 0x63);
        assert_eq!(SBOX[0x01], 0x7c);
        assert_eq!(SBOX[0x53], 0xed);
        assert_eq!(SBOX[0xff], 0x16);
        for i in 0..256 {
            assert_eq!(INV_SBOX[SBOX[i] as usize] as usize, i);
        }
    }

    #[test]
    fn fips197_appendix_c1() {
        let key: [u8; 16] = unhex("000102030405060708090a0b0c0d0e0f").try_into().unwrap();
        let pt: [u8; 16] = unhex("00112233445566778899aabbccddeeff").try_into().unwrap();
        let ct: [u8; 16] = unhex("69c4e0d86a7b0430d8cdb78070b4c55a").try_into().unwrap();
        let aes = Aes128::new(&key);
        assert_eq!(aes.encrypt_block(&pt), ct);
        assert_eq!(aes.decrypt_block(&ct), pt);
    }

    #[test]
    fn rfc3394_section_4_1() {
        let kek: [u8; 16] = unhex("000102030405060708090a0b0c0d0e0f").try_into().unwrap();
        let key = unhex("00112233445566778899aabbccddeeff");
        let wrapped = unhex("1fa68b0a8112b447aef34bd8fb5a7b829d3e862371d2cfe5");
        assert_eq!(key_wrap(&kek, &key).unwrap(), wrapped);
        assert_eq!(key_unwrap(&kek, &wrapped).unwrap(), key);
        let mut bad = wrapped.clone();
        bad[10] ^= 1;
        assert!(key_unwrap(&kek, &bad).is_none());
    }

    #[test]
    fn rfc3610_packet_vector_1() {
        let key: [u8; 16] = unhex("c0c1c2c3c4c5c6c7c8c9cacbcccdcecf").try_into().unwrap();
        let nonce: [u8; 13] = unhex("00000003020100a0a1a2a3a4a5").try_into().unwrap();
        let aad = unhex("0001020304050607");
        let plain = unhex("08090a0b0c0d0e0f101112131415161718191a1b1c1d1e");
        let expect = unhex("588c979a61c663d2f066d0c2c0f989806d5f6b61dac38417e8d12cfdf926e0");
        assert_eq!(ccm_encrypt(&key, &nonce, &aad, &plain), expect);
        assert_eq!(ccm_decrypt(&key, &nonce, &aad, &expect).unwrap(), plain);
        let mut bad = expect.clone();
        bad[3] ^= 0x80;
        assert!(ccm_decrypt(&key, &nonce, &aad, &bad).is_none());
        assert!(ccm_decrypt(&key, &nonce, b"other", &expect).is_none());
    }

    #[test]
    fn ccm_round_trips_for_every_length() {
        let key = [7u8; 16];
        let nonce = [9u8; 13];
        for len in 0..100usize {
            let plain: Vec<u8> = (0..len).map(|i| (i * 31) as u8).collect();
            let sealed = ccm_encrypt(&key, &nonce, b"hdr", &plain);
            assert_eq!(sealed.len(), len + CCM_TAG);
            assert_eq!(ccm_decrypt(&key, &nonce, b"hdr", &sealed).unwrap(), plain);
        }
    }
}
