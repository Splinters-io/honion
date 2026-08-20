# Security model

## What this program is trusted to do

Produce an Ed25519 key that (a) nobody else can predict, and (b) actually
corresponds to the address written next to it. Everything below is in service of
those two properties.

## Threat model

**In scope**

- A bug in this project's own arithmetic, kernel, or encoding producing a key
  that does not match its address, or a key with less entropy than it appears to
  have.
- Hardware faults: a bit flip in 96 GB of VRAM, a marginally overclocked GPU.
- Key material leaking into places that outlive the process — logs, core dumps,
  swap.

**Out of scope**

- An attacker with code execution on this machine while the search runs. They
  can read the scalars out of host memory; nothing here prevents that.
- Traffic analysis, or any attack on Tor itself.
- Whether a vanity address is a good idea for anonymity in the first place. It
  is not a security property; a memorable prefix helps humans recognise your
  service and does nothing to authenticate it. Users still have to check all 56
  characters, and an attacker can generate their own address with the same
  prefix as easily as you did.

## Secret material never reaches the GPU

This is the central architectural decision, and it falls out of the algorithm
rather than being bolted on.

The host draws a clamped scalar `a₀` per thread from the system CSPRNG and
computes `A₀ = a₀·B` on the CPU with `curve25519-dalek`. Only `A₀` — a public
key — is uploaded. The device walks `A₀ + k·8B` and reports `(thread, k)`. The
host reconstructs the secret as `a₀[thread] + 8k` from its own memory.

The kernel therefore contains no randomness, no hashing, and no secret. It is a
pure filter over public data.

Two consequences:

- **VRAM cannot be wiped reliably**, and does not need to be. Device memory
  never holds anything worth wiping. This is a real problem for GPU crypto
  generally, and the design sidesteps it rather than mitigating it.
- **The device can be debugged, profiled and dumped freely** without leaking
  anything.

The crate structure enforces it: `honion-gpu` does not depend on
`honion-keyfile`, so there is no code path by which a secret could reach the
device.

## Where secrets do live

In `honion-keyfile`, in `SecretScalar` and `ExpandedSecretKey`, both
`ZeroizeOnDrop`. `Debug` on `SecretScalar` prints `SecretScalar(<redacted>)` and
is tested to do so — logs outlive processes, and a leaked scalar is a
permanently compromised service.

`SecretScalar` has no derived `PartialEq`, deliberately: a derived comparison
would compare secret bytes in variable time.

**Not addressed:** the scalar buffer is not `mlock`ed, so it can in principle
reach swap. On a machine with encrypted swap or none, this is moot. It is listed
here rather than silently omitted.

## Entropy

Every scalar comes from `getrandom`, which is the OS CSPRNG — `getrandom(2)` on
Linux. There is no fallback to a weaker source: if the system random source
fails, the run fails.

Scalars are **not** derived from a counter, an index, or a hash of a thread ID.
This matters more than it sounds. A natural-looking optimisation is to derive
each thread's starting scalar from a single seed plus its thread index, saving a
CSPRNG call per thread. That would make every key in a run predictable from any
one of them. `honion` draws each starting scalar independently, and re-draws all
of them every launch.

## Why there is no SHA-512

Normally an Ed25519 key comes from a 32-byte seed: `SHA-512(seed)` gives 64
bytes, the first 32 clamp into the scalar and the rest become the nonce prefix.

A vanity search moves through scalar space directly and lands on `a₀ + 8k`. No
seed produces that scalar, because SHA-512 cannot be inverted. So the output is
necessarily an expanded key — which is exactly what Tor's
`hs_ed25519_secret_key` stores, so nothing is lost.

Given that, running SHA-512 at all would be theatre. `honion` draws the clamped
scalar straight from the CSPRNG.

**Is that sound?** Yes. Clamping is a fixed bit-mask, so hashing a random seed
and clamping produces a value uniform over `{x : x ≡ 0 mod 8, 2²⁵⁴ ≤ x < 2²⁵⁵}`
— assuming SHA-512 output is indistinguishable from random, which is the
assumption Ed25519 already makes. Clamping 32 CSPRNG bytes produces a value
uniform over exactly that same set, without the assumption. The nonce half is 32
further independent CSPRNG bytes; its only requirement is to be secret and fixed
per key, which independent randomness satisfies at least as well as a hash does.

## Every result is verified before it is written

The GPU is treated as an untrusted accelerator. Its output is a *claim*.
`VerifiedKey::verify` checks, entirely on the CPU:

1. `a₀ + 8k` is still a clamped scalar — this is where the overflow invariant is
   enforced rather than assumed.
2. The public key, re-derived with `curve25519-dalek`, matches at least one
   pattern under the host reference matcher. Not the device's `pattern_id`,
   which is advisory: the answer is recomputed.
3. The address is built and survives a parse round trip, which recomputes the
   SHA3-256 checksum from scratch.
4. The key produces a signature that verifies against the public key.

Only then does a `VerifiedKey` exist, and `write_service_dir` accepts nothing
else. This costs a fraction of a millisecond, once per hit, against billions of
candidates per second — so it is free, and it means a miscompiled kernel, a bit
flip, or a mistake in this project's own field arithmetic cannot produce a key
file that disagrees with its hostname.

Verification failure is reported loudly and aborts the run. It is never an
ordinary condition to skip past: it means the GPU and the CPU disagree, which is
a bug or a hardware fault.

## On disk

Directory `0700`, files `0600`, written to a temporary name and renamed into
place — atomic on any POSIX filesystem, so an interrupted run leaves either no
file or a complete one, never a truncated secret key.

An existing directory for the same address is never overwritten. The key inside
it cannot be recovered if destroyed.

`honion verify <dir>` re-derives the address from the stored secret key and
checks it against the stored `hostname`, so a directory can be audited long
after it was produced — including one made by other tools.

## Running this as a service

`honion` currently generates complete keypairs locally, which is right for
generating your own address and wrong for generating someone else's. If you run
it as a service, **you learn every customer's private key**, permanently. You
could impersonate or seize their service at any point in the future, and so
could anyone who compromises the machine or obtains its logs or backups.

The fix is well known and costs nothing here: **additive split-key generation**.

1. The customer generates a keypair locally and sends only the public key `P`.
2. The service searches for an offset `s` such that `P + s·B` matches the
   pattern — the identical kernel, started from `P` instead of a random point.
3. The service returns `s`, which is useless on its own.
4. The customer computes the final secret `a + s` locally.

The service never learns the secret. The search cost is unchanged; only the
starting point differs, so `honion`'s architecture already accommodates it.

This is not implemented. It was considered and deliberately deferred, and it
should be revisited before any of this is exposed to anyone else's keys.
