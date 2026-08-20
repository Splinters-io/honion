# honion

A GPU vanity address generator for Tor v3 onion services.

Finds Ed25519 keys whose `.onion` addresses start with text you choose, at
**~12.5 billion candidate addresses per second** on an RTX PRO 6000 Blackwell,
measured end to end by counting keys actually written rather than by trusting
any tool's own counter. See [docs/07-benchmarks.md](docs/07-benchmarks.md) for
the method and the numbers, and
[docs/02](docs/02-gpu-architecture.md) for the four rewrites that got it there —
and the five attempts that did not.

```
$ honion search --prefix carroll --out ./keys --count 1
honion: searching for 1 pattern(s)
  carroll (35.0 bits, from --prefix)
  device compute capability 12.0, 262144 threads, 256 offsets (513 candidates per inversion, 8.6 GB device memory)
     45.26G examined    11.31G/s  elapsed  4.0s  expected  3.0s  P(found) 74.4%  hits 0
     …
carrolliuz67cgsdlpeu5e2hcrcg3rzaowhrj2dps4pwwgrkxd7xrtqd.onion
  written to ./keys/carrolliuz67cgsdlpeu5e2hcrcg3rzaowhrj2dps4pwwgrkxd7xrtqd.onion
```

Each result is a directory named for the address — `<address>.onion/`, holding
`hs_ed25519_secret_key`, `hs_ed25519_public_key` and `hostname` — which you can
point Tor's `HiddenServiceDir` at directly.
This has been tested against Tor 0.4.9.6, which loads the key and adopts the
address unmodified.

## Requirements

- An NVIDIA GPU, compute capability 5.0 or later
- The NVIDIA driver and `libnvrtc` — **no CUDA toolkit needed**; device code is
  compiled at run time
- Rust 1.85 or later

```
cargo build --release
./target/release/honion --help
```

## Use

```
honion estimate --prefix carroll            # how long will this take?
honion search   --prefix carroll --out ./keys --count 1
honion verify   ./keys/carroll…             # re-derive and check a result
```

Patterns are base32 characters (`a`–`z`, `2`–`7`), `?` for any character, and
`[abc]` / `[^abc]` for a choice. Repeat `--prefix`, or pass
`--patterns-file` — searching many patterns at once costs almost nothing and is
the cheapest way to improve your odds.

Note there is no `0`, `1`, `8` or `9` in base32. Use `o`, `l`, `s`, `z`.

## How long

| characters | expected time at 12.5 G/s |
|---|---|
| 7 | 3 seconds |
| 8 | 1.5 minutes |
| 9 | 47 minutes |
| 10 | 1.0 days |
| 11 | 33 days |

Ten characters is the practical ceiling on one card. These are *means*: the
search is memoryless, so there is a 63% chance of a result by the expected time
and 95% by three times it. The progress line reports `P(found)` so you can tell
unlucky from broken.

## How it works

An onion address *is* a public key, base32-encoded. Four facts make a fast
search possible:

1. **Consecutive valid scalars differ by 8**, because Ed25519 clamping clears
   the low three bits. So stepping to the next candidate is one point addition
   instead of a full scalar multiplication — about 250× less work.
2. **The first 51 address characters depend only on the public key**, never on
   the SHA3-256 checksum. So testing a prefix needs no hashing at all: the GPU
   masks raw key bytes and compares integers.
3. **The dual addition law gives two candidates for the price of one.** For a
   twisted Edwards curve with `a = -1`, `y(P±Q)` can be written without the
   curve constant and without projective coordinates, and the two signs share
   their multiplications. Two multiplications, two candidates — where a
   straightforward point walk needs eight for one.
4. **Inversions are batched.** Candidates come out as fractions, so Montgomery's
   trick turns 513 divisions into one inversion — and the next base point's
   `Z` rides along in the same product.

Together: about 5.5 field multiplications per address, using eight 32-bit limbs
with hardware carry chains rather than the more usual 64-bit-accumulator
representation.

Beyond that the wins were not arithmetic at all. Candidate denominators are
recomputed rather than stored, trading a multiplication for half the memory
traffic; and the running product is split in two so its dependency chain is half
as long. Both were found by ablation — at that point, removing arithmetic had
stopped helping. See [docs/02](docs/02-gpu-architecture.md).

[`docs/`](docs/) explains each of these properly, including the parts that were
harder than expected.

## Correctness

Generating a key that does not match its own address would be a silent, durable
failure, so correctness is established layer by layer against implementations
that share no code with this one:

| layer | checked against |
|---|---|
| base32 and address encoding | RFC 4648 vectors; real published `.onion` addresses |
| field arithmetic in GF(2²⁵⁵−19), both implementations | `num-bigint`, 4096 random and boundary cases |
| Edwards curve arithmetic | `curve25519-dalek` |
| the search kernel | the host reference matcher, compared for *exact* set equality |
| generated key files | pure-Python Ed25519, and real Tor |

And every hit is rebuilt from the host's own secret scalar, re-derived,
re-matched and signature-checked before anything reaches disk. A miscompiled
kernel or a bit flip in 96 GB of VRAM costs time; it cannot produce a bad key.

## Security

**No secret ever reaches the GPU.** Scalars are drawn on the host, only public
points are uploaded, and the device returns `(thread, iteration)` pairs from
which the host reconstructs the secret. The kernel contains no randomness, no
hashing and no key material — so the fact that VRAM cannot be reliably wiped
stops mattering.

A vanity address is *not* a security property. It helps humans recognise your
service; it does not authenticate it, and an attacker can generate an address
with the same prefix as easily as you did.

**If you plan to run this as a service, read
[docs/05-security-model.md](docs/05-security-model.md) first.** As it stands it
generates complete keypairs locally, which means you would learn every
customer's private key, permanently. The fix — additive split-key generation —
costs nothing and is described there, but is not implemented.

## Checking it yourself

```bash
cargo test --release --workspace     # 105 tests; GPU ones skip without a device
cargo clippy --release --workspace --all-targets

python3 cuda/gen_fe32.py | diff - cuda/fe25519_u32.cuh   # generated field
                                                         # arithmetic still
                                                         # matches its generator
python3 cuda/verify_dual_law.py      # the addition law, against the standard one
```

The field arithmetic is generated rather than hand-written, because it is 144
lines whose only content is operand indices — the first diff above is what
catches a stale header. The two functions in that file that *were* hand-written
both shipped with a transposed-index bug, caught on the first run by the
differential test.

To reproduce the benchmark, see [`bench/`](bench/).

## Documentation

| | |
|---|---|
| [00 — What a v3 onion address is](docs/00-onion-v3-address-spec.md) | the format, and the bit boundary everything depends on |
| [01 — Ed25519 vanity search](docs/01-ed25519-vanity-search.md) | why this is billions per second and not millions |
| [02 — GPU architecture](docs/02-gpu-architecture.md) | the kernel, the field arithmetic, and two compiler traps |
| [03 — Langsec design](docs/03-langsec-design.md) | the input language and the trust boundaries |
| [04 — Pattern grammar](docs/04-pattern-grammar.md) | EBNF, semantics, and every rejection |
| [05 — Security model](docs/05-security-model.md) | threat model, key handling, and the service problem |
| [06 — Performance](docs/06-performance.md) | measurements, tuning, and what was left undone |
| [07 — Benchmarks](docs/07-benchmarks.md) | measured against `mkp224o` and `prefix32`, method included |

## Layout

```
bench/    benchmark harness and raw per-run data
cuda/     device code: field arithmetic, curve arithmetic, the search kernel
crates/
  honion-core     base32, addresses, the pattern language   (no GPU, no secrets)
  honion-gpu      the kernel and its host driver            (no secrets)
  honion-keyfile  secret scalars, verification, Tor files   (no GPU)
  honion-cli      the honion binary
```

`honion-gpu` does not depend on `honion-keyfile`. That is deliberate: there is
no code path by which a secret could reach the device.

## Licence

MIT OR Apache-2.0.
