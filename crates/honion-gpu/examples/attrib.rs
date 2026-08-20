//! Attribute PTX instructions in honion_search to the source lines that emitted
//! them, using NVRTC line info. Guessing where kernel time goes has already
//! been wrong twice in this project; this measures it.
fn main() {
    let src = honion_gpu::nvrtc::sources::SEARCH;
    let ptx = honion_gpu::nvrtc::compile_with_options(
        src, (12, 0),
        &[("HALF", "256".into()), ("FE_RADIX32", "1".into())],
        &["--generate-line-info".to_string()],
    ).unwrap();
    // Map file ids to names from the .file directives.
    let mut files = std::collections::HashMap::new();
    for line in ptx.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix(".file") {
            let mut it = rest.split_whitespace();
            if let (Some(id), Some(name)) = (it.next(), it.next()) {
                files.insert(id.to_string(), name.trim_matches('"').to_string());
            }
        }
    }
    // Walk honion_search, tracking the current .loc and counting instructions.
    let mut counts: std::collections::BTreeMap<(String, String), usize> = Default::default();
    let mut cur = ("?".to_string(), "?".to_string());
    let mut inside = false;
    let mut total = 0usize;
    for line in ptx.lines() {
        let t = line.trim();
        if t.contains(".entry honion_search") { inside = true; continue; }
        if inside && t.contains(".entry ") && !t.contains("honion_search") { break; }
        if !inside { continue; }
        if let Some(rest) = t.strip_prefix(".loc") {
            let mut it = rest.split_whitespace();
            let f = it.next().unwrap_or("?").to_string();
            let l = it.next().unwrap_or("?").to_string();
            cur = (files.get(&f).cloned().unwrap_or(f), l);
            continue;
        }
        if t.ends_with(';') && t.starts_with(|c: char| c.is_ascii_lowercase()) && !t.starts_with('.') {
            *counts.entry(cur.clone()).or_insert(0) += 1;
            total += 1;
        }
    }
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    println!("total instructions in honion_search: {total}");
    println!("{:>7}  {:>5}  source line", "count", "%");
    for ((file, line), n) in v.iter().take(18) {
        println!("{n:>7}  {:>4.1}%  {}:{}", *n as f64 / total as f64 * 100.0, file, line);
    }
}
