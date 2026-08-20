//! Report the PTX instruction mix of the search kernel.
//!
//! Written while diagnosing why the kernel was slower than a competitor's. It
//! showed 56% of operations were 64-bit — each costing two 32-bit operations on
//! this hardware — which is what motivated the 8x32-limb field arithmetic.
//! Kept because that number is worth being able to re-check after any change to
//! `cuda/fe25519*.cuh`.
fn main() {
    let ptx = honion_gpu::nvrtc::compile(honion_gpu::nvrtc::sources::SEARCH, (12,0),
        &[("HALF","256".into()), ("FE_RADIX32","1".into())]).unwrap();
    let body: String = ptx.lines()
        .skip_while(|l| !l.contains(".entry honion_search"))
        .take_while(|l| !l.contains(".entry honion_walk_dump"))
        .collect::<Vec<_>>().join("\n");
    let mut counts = std::collections::BTreeMap::new();
    for line in body.lines() {
        let t = line.trim();
        if let Some(op) = t.split_whitespace().next() {
            if op.starts_with(|c: char| c.is_ascii_lowercase()) && t.ends_with(';') {
                *counts.entry(op.trim_end_matches(';').to_string()).or_insert(0usize) += 1;
            }
        }
    }
    let total: usize = counts.values().sum();
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    println!("total PTX ops in honion_search: {total}");
    let s64: usize = v.iter().filter(|(o,_)| o.ends_with("64")).map(|(_,n)| n).sum();
    println!("64-bit ops: {s64} ({:.0}% of all) -- each costs two 32-bit ops on this hardware", s64 as f64/total as f64*100.0);
    println!();
    for (op, n) in v.iter().take(10) { println!("{n:>7}  {op}"); }
}
