# The pattern language

## Grammar

```ebnf
pattern   = atom , { atom } ;              (* 1 to 51 atoms *)
atom      = literal | wildcard | class ;
literal   = "a".."z" | "2".."7" ;          (* the base32 alphabet *)
wildcard  = "?" ;
class     = "[" , [ "^" ] , literal , { literal } , "]" ;
```

A pattern matches the **start** of an onion address. Atom `i` constrains
address character `i`: a literal admits one character, `?` admits all 32, and
`[…]` admits those listed — or, with `^`, all *except* those listed.

Recognised by [`pattern/parse.rs`](../crates/honion-core/src/pattern/parse.rs);
compiled by [`pattern/compile.rs`](../crates/honion-core/src/pattern/compile.rs).

## The alphabet

Base32 is `a`–`z` and `2`–`7`. There is no `0`, `1`, `8` or `9`, and no
uppercase. So `hon10n` is not a pattern anyone can search for — the parser says
so, with the offset marked:

```
--prefix "hon[i1]on": unexpected byte 0x31 at offset 5: expected a base32 character (a-z, 2-7) or ']'
  hon[i1]on
       ^
```

Common substitutions that *do* work, since the letters are all available:
`l` for `1`, `o` for `0`, `s` for `5`, `z` for `2`.

## Why 51 characters is the limit

An address is `base32(pubkey ‖ checksum ‖ version)`. The public key is 32 bytes
= 256 bits = 51.2 base32 characters, so characters 0–50 depend only on the key
while character 51 onward involves the SHA3-256 checksum.

A 52-character pattern could not be tested by masking public-key bytes; it would
need a hash per candidate. Rather than accept such a pattern and be slow, the
language does not contain it. See
[00-onion-v3-address-spec.md](00-onion-v3-address-spec.md).

In practice this is not a constraint anyone meets: eleven characters already
takes months.

## Difficulty

Each atom independently admits `n` of 32 values, so a uniformly random address
matches with probability `∏ nᵢ/32`. In bits:

```
difficulty = Σ (5 − log₂ nᵢ)
```

A literal costs 5 bits, a wildcard 0, a two-member class 4, `[^a]` about 0.05.

`honion estimate` reports this without searching:

```
$ honion estimate --prefix carroll --prefix "hon[i2]on"
  carroll                    35.0 bits  expected       7.0s   [--prefix]
  hon[i2]on                  29.0 bits  expected      110ms   [--prefix]

combined difficulty : 29.0 bits (528.60M expected trials)
```

Searching several patterns at once is nearly free — the cost is one binary
search per distinct mask shape — so extra patterns are almost pure gain.

## What compilation produces

Each pattern becomes a **prefilter**, a 64-bit `(mask, target)` over the first
eight key bytes, plus **residuals** for anything the prefilter cannot express.

The prefilter covers the first 12 characters and holds every position that
admits exactly one character. Residuals hold multi-character classes and any
position past character 12; they are checked only on candidates that already
passed the prefilter.

Wildcards contribute to neither — they constrain nothing — but they still
occupy a position, so later atoms keep their character indices.

```
carroll     →  mask covers 7 characters (35 bits), no residuals: prefilter exact
a[bc]d      →  mask covers characters 0 and 2 (10 bits), one residual at 1
?b          →  mask covers character 1 only (5 bits), no residuals
[q]         →  identical to `q`; a one-member class folds into the prefilter
```

## Rejected inputs

Each of these fails with a specific error, not a generic one — a parser that
rejects for the wrong reason is still a parser whose language you do not know.

| input | error |
|---|---|
| `""` | `Empty` |
| 52 or more atoms | `TooLong` |
| `carRoll` | `UnexpectedByte` at offset 3 |
| `hon10n` | `UnexpectedByte` at offset 3 |
| `ab[cd` | `UnterminatedClass` opened at 2 |
| `ab]cd` | `UnmatchedClassClose` at 2 |
| `ab[]`, `ab[^]` | `EmptyClass` at 2 |
| `[a[b]]` | `UnexpectedByte` at 2, in class context |
| `[a?b]` | `UnexpectedByte` at 2 — `?` is an atom, not a class member |
| `[^abcdefghijklmnopqrstuvwxyz234567]` | `UnsatisfiableClass` — matches nothing, ever |

Offsets are byte offsets, including for multi-byte UTF-8, because a byte offset
is what a caller needs to point at the input.

## Searching many patterns

`--prefix` may be repeated, and `--patterns-file` reads one pattern per line
with `#` comments and blank lines ignored. Both sources are combined, and one
bad pattern anywhere fails the run before any work begins.

There is a floor on how weak the *prefilter* may be — not the pattern, the
prefilter. A pattern like `?????????????abcdefgh` is hard to satisfy yet places
no constraint in the first eight bytes, so the device would forward every
candidate to the host. `honion` refuses, and says why:

```
these patterns give the GPU prefilter only 0.0 bits of selectivity, but 20 are
needed. The device would forward candidates faster than the host can check them.
This happens when a pattern's fixed characters all sit past position 12, or when
it is mostly wildcards and character classes; anchoring more literal characters
near the start fixes it.
```
