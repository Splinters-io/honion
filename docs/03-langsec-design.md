# Langsec, and what it means for a key generator

Language-theoretic security starts from an observation about where software
actually fails: not usually in its algorithms, but at its boundaries, where
input is turned into structure. Programs that scatter recognition through the
code — a `split` here, a length check there, an assumption three functions
later — end up with an *ad hoc parser* whose accepted language nobody knows,
including its authors. The prescription is to define the input language
formally, recognise it completely and in one place, and let everything
downstream be total.

Two features of this program make that unusually worth doing.

**A parsing bug is expensive but silent.** If a pattern is mis-compiled, the
search does not crash. It runs at full speed, examining billions of keys
against the wrong predicate, and finds nothing. On a nine-character prefix that
is hours of GPU time before anyone suspects the pattern rather than the odds.

**The output is permanent.** An onion address is an identity that may be
published, linked to and trusted for years. A key file that does not match its
own hostname is not a bug you fix in the next release.

## The six rules, and where each one lives

### 1. One formally specified input language

The pattern grammar is written as EBNF in
[04-pattern-grammar.md](04-pattern-grammar.md) and implemented by a single
hand-written recogniser in
[`pattern/parse.rs`](../crates/honion-core/src/pattern/parse.rs). There is no
regular-expression engine, no `split`, and no ad hoc scanning of user text
anywhere else in the workspace.

The one place text is treated leniently — skipping blank lines and `#` comments
in a pattern file — is defined in
[`patterns.rs`](../crates/honion-cli/src/patterns.rs), operates on whole lines,
and is the only such rule in the program.

### 2. Full recognition before processing

Every pattern, from every source, is parsed and compiled before a single byte
of GPU memory is allocated or a single secret scalar is drawn. One malformed
pattern fails the whole run, with the byte offset and a caret:

```
$ honion estimate --prefix "hon[i1]on"
Error: --prefix "hon[i1]on": unexpected byte 0x31 at offset 5: expected a base32 character (a-z, 2-7) or ']'
  hon[i1]on
       ^
```

There is no partial acceptance and no best-effort mode. A pattern file with one
bad line on line 12 is rejected as a file; the first eleven patterns are not
quietly searched for.

### 3. Parse, don't validate

Invariants live in types with private constructors, so holding a value *is* the
proof:

| type | what holding one proves |
|---|---|
| `Base32Char` | the byte is in the lowercase base32 alphabet |
| `CharClass` | the set of admitted characters is non-empty |
| `Pattern` | 1–51 atoms, every atom satisfiable |
| `OnionAddress` | 56 canonical characters, checksum and version correct |
| `SecretScalar` | clamped: low three bits clear, bit 254 set, bit 255 clear |
| `VerifiedKey` | key, public key and address agree, and the key can sign |

`VerifiedKey` is the one that matters most. It cannot be constructed except by
`VerifiedKey::verify` returning `Ok`, and `write_service_dir` takes one. There
is therefore no code path that writes an unverified key — not as a matter of
discipline, but as a matter of what the type system permits.

An empty `CharClass` is worth singling out. `[^abcdefghijklmnopqrstuvwxyz234567]`
excludes the entire alphabet, so no address could ever match and the search
would run forever. It is rejected at parse time, because "runs forever" is a
result the type system can rule out.

### 4. The device is not a parser

The GPU receives fixed-size integers: masks, sorted targets, offset arrays, and
32-byte compressed points. It contains no strings, reads no length out of the
data it is processing, allocates nothing, and has no data-dependent control flow
beyond the search itself.

Everything it sees is built on the host by
[`tables.rs`](../crates/honion-gpu/src/tables.rs) from an already-recognised
pattern set, in ordinary testable Rust.

The device source itself is embedded with `include_str!` and its `#include`
directives resolve against a closed list. The set of text that can reach the
device compiler is fixed when the binary is built.

### 5. Bounded computation from input

No input causes unbounded work or allocation. Patterns are capped at 51 atoms —
not an arbitrary limit but exactly the number of address characters determined
by the public key, so a longer pattern is not merely refused, it is *not in the
language*. Character classes compile to 32-bit sets rather than expanding
combinatorially. The prefilter-selectivity check refuses pattern sets that would
forward candidates faster than the host can verify them.

### 6. One encoder, one decoder, proven inverse

A single base32 implementation, in
[`base32.rs`](../crates/honion-core/src/base32.rs), used by address formatting,
pattern compilation and the mask computation alike. `decode` rejects uppercase,
impossible lengths and non-canonical trailing bits, so it is injective and
`encode(decode(s)) == s` for everything it accepts.

## The claim that ties it together

Rules 1–6 are means. The end is a single property, established by property test
over thousands of random keys and patterns in
[`pattern_semantics.rs`](../crates/honion-core/tests/pattern_semantics.rs):

> A key passes the compiled bit-masking predicate **if and only if** the address
> derived from that key, spelled out in base32, literally begins with characters
> the pattern admits.

The test's reference side is deliberately naive: it walks characters of the
finished address string and consults each atom directly, sharing no code with
the compiler. Agreement between two implementations that different is evidence.
Agreement between an implementation and itself is not.

Everything else — the GPU kernel, the prefilter, the batching — is an
optimisation of that predicate, and is tested for agreement with it.

## Where the discipline stops

Langsec is about inputs. It says nothing about whether the arithmetic is right,
and the arithmetic is where a project like this is most likely to be wrong. That
is handled separately, by differential testing against independent
implementations at every layer, and by verifying every result before it is
written. See [02-gpu-architecture.md](02-gpu-architecture.md) and
[05-security-model.md](05-security-model.md).
