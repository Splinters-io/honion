# What a v3 onion address actually is

An onion address is not a name that was registered; it is a *public key,
written down*. There is no registry, no authority and no lookup — the address
and the key are the same object in two encodings. That is what makes vanity
generation possible at all: to get an address you like, you keep making keys
until one of them spells something.

## The construction

From Tor's `rend-spec-v3.txt`, section 6:

```
onion_address = base32(PUBKEY ‖ CHECKSUM ‖ VERSION) ‖ ".onion"
CHECKSUM      = SHA3-256(".onion checksum" ‖ PUBKEY ‖ VERSION)[:2]
VERSION       = 0x03
```

- `PUBKEY` — 32 bytes, an Ed25519 public key (a compressed curve point).
- `CHECKSUM` — 2 bytes, guarding against typos. It is not a security feature;
  anyone can compute it.
- `VERSION` — one byte, `3`.
- `base32` — RFC 4648, **lowercase, unpadded**.

35 bytes in, 56 characters out. Every v3 address is exactly 56 characters
before the `.onion`.

Implemented in [`crates/honion-core/src/address.rs`](../crates/honion-core/src/address.rs),
and pinned against real addresses published by the Tor Project and DuckDuckGo —
which is a genuine test rather than a circular one, because `parse` recomputes
the checksum from the decoded key and rejects a mismatch. If our byte layout or
our SHA3 personalisation string were wrong, those addresses would not validate.

## The bit boundary that this project is built on

Base32 emits five bits per character, most significant bit first. So character
`i` of the address covers bits `[5i, 5i+5)` of the 35-byte body, counting bit 0
as the top bit of byte 0.

The public key occupies the first 32 bytes = **256 bits = 51.2 characters**.

Therefore:

> **The first 51 characters of an onion address are a function of the public key
> alone. The checksum cannot influence any of them.**

This is the single most important fact in the whole project. It means a search
for an address *prefix* never has to compute SHA3-256 — it can mask raw public
key bytes and compare integers. A search for a *suffix*, by contrast, would need
a SHA3-256 for every candidate, costing roughly an order of magnitude. That is
why `honion` offers prefixes and not suffixes.

The boundary is asserted at compile time in `address.rs`:

```rust
const _: () = {
    assert!(base32::prefix_bytes_needed(PREFIX_CHARS_WITHOUT_CHECKSUM) == PUBKEY_LEN);
    assert!(base32::prefix_bytes_needed(PREFIX_CHARS_WITHOUT_CHECKSUM + 1) > PUBKEY_LEN);
};
```

and demonstrated by a test that takes two keys differing only in their final
bit and shows their addresses agree on the first 51 characters.

## Why the base32 codec is its own module, used by everything

A vanity search is a search over the *output* of the base32 encoder. The GPU
does not encode anything — it masks bits. If the encoder and the bit-masking
disagreed by one position, the search would find keys whose addresses do not
match the requested pattern, and would do it silently and at full speed.

So there is exactly one definition of the bits-to-characters mapping, in
[`base32.rs`](../crates/honion-core/src/base32.rs), and everything derives from
it: address formatting, pattern compilation, and the mask/target values uploaded
to the device. The property test in
[`pattern_semantics.rs`](../crates/honion-core/tests/pattern_semantics.rs)
closes the loop by checking, over thousands of random keys and patterns, that
the compiled bit-masking predicate agrees with reading characters out of the
finished address string.

## Details worth not getting wrong

**Uppercase is not accepted.** Tor addresses are lowercase. Accepting uppercase
would make decoding many-to-one and give two spellings of one address — a
classic parser-differential hazard. `Base32Char::from_ascii` recognises only
the lowercase alphabet.

**Non-canonical encodings are rejected.** Unpadded base32 lengths are
`n mod 8 ∈ {0,2,4,5,7}`; other lengths encode a fractional byte. And the final
character's unused low bits must be zero, or the string decodes to the same
bytes as a different, canonical spelling. Both are refused, so `decode` is
injective and `encode(decode(s)) == s` for everything accepted.

**The digits are 2–7.** The base32 alphabet has no `0`, `1`, `8` or `9`. A
pattern containing them is rejected at parse time with the offset marked — you
cannot have `hon10n.onion`.
