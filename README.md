# hopgrepX — Disk-Aware Log Search for Large Sorted Logs

![GitHub stars](https://img.shields.io/github/stars/harshavardhanreddy06/HopgrepX)
![License](https://img.shields.io/github/license/harshavardhanreddy06/HopgrepX)

**Avoid full scans on large logs. Reduce disk I/O by orders of magnitude under cold-cache workloads.**

Traditional tools like `grep` and `ripgrep` optimize CPU throughput but still require scanning the entire file. When logs do not fit in memory, disk I/O dominates runtime and full scans become prohibitively expensive. 

**hopgrepX** exploits key ordering (timestamps, numeric IDs, or strings) to avoid full file scans and drastically reduce disk reads.

---

## Motivation: Why grep is the wrong abstraction for large logs

Most log search tools treat log files as unstructured text and rely on
sequential scans. This works well when data is cached in memory, but breaks
down under cold-cache conditions where the operating system must repeatedly
fetch data from disk.

In practice, many real-world log files are naturally sorted by a primary key
such as timestamp or request identifier. hopgrepX leverages this ordering to
locate relevant regions using a small number of random disk probes followed
by a bounded sequential scan, instead of reading the entire file.

### Why This Matters

Most log search tools (grep/ripgrep) are optimized for CPU throughput but still scan the entire file. On large logs that don’t fit in memory, disk I/O dominates runtime. **hopgrepX minimizes disk accesses by exploiting key ordering to avoid full scans.**

This creates an O(log N) search complexity versus O(N) scan complexity, which is critical for GB/TB-scale logs.

## Benchmarks

Tests performed on a MacBook Pro (M-series). `grep` times scale linearly with file size, while `hopgrepX` remains sub-second.

| Tool | Log Size | Query Type | Time |
| :--- | :--- | :--- | :--- |
| **grep** | 700 MB | Exact Match (End of File) | 8.53s |
| **hopgrepX** | 700 MB | Exact Match (End of File) | **0.49s** |
| **grep** | 10 GB | Exact Match (End of File) | > 120s (Est) |
| **hopgrepX** | 10 GB | Exact Match (End of File) | **0.20s** |

---

## Installation

hopgrepX is a standalone Python script with no external dependencies.

**Requirements**: Python 3.6 or higher.

1. Clone the repository or download `hopgrepX.py`:
   ```bash
   git clone https://github.com/yourusername/hopgrepX.git
   cd hopgrepX
   ```

2. Make the script executable:
   ```bash
   chmod +x hopgrepX.py
   ```

3. (Optional) Symlink to your bin directory for global access:
   ```bash
   ln -s $(pwd)/hopgrepX.py /usr/local/bin/hopgrepX
   ```

### Quick Run

Verify performance instantly:

```bash
# Clear caches to simulate cold start
# macOS:
sudo purge 
# Linux:
# sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"

# Run benchmark
time ./hopgrepX.py huge.log --eq "2025-01-01 00:00:00"
```

---

## High-level design

hopgrepX follows a multi-stage search pipeline:

1. **Macro search (k-ary probing)**  
   The file is sampled at multiple offsets to quickly narrow the search
   window containing the target key.

2. **Micro refinement**  
   The macro phase is repeated on a smaller region until the window is
   reduced to a few hundred kilobytes.

3. **Bounded sequential scan**  
   A short sequential scan is performed within the refined window to find
   exact matches.

4. **Adaptive ZoneMap (optional)**  
   Observed key ranges are recorded as lightweight summaries in a `.hop.idx`
   sidecar file to accelerate future searches.

This design minimizes disk I/O while remaining robust to non-uniform key
distributions and variable-length log records.

---

## Supported query types

hopgrepX supports multiple query modes on the **first column**, which must be
sorted:

- **Exact match** (`--eq`)
- **Range queries** (`--range`)
- **Prefix queries** (`--prefix`, string-based)
- **Multiple exact matches** (`--multi`)
- **Substring search** (default behavior, similar to `grep`)

Additional filtering can be applied using a SQL-like `--where` clause.

---

## Example usage

```bash
# Exact timestamp search
./hopgrepX.py app.log --eq "2024-01-15 10:30:00"

# Range query with filtering
./hopgrepX.py system.log \
  --range "2024-01-01" "2024-01-02" \
  --where "level='ERROR'"

# Prefix search (string-based)
./hopgrepX.py access.log --prefix "192.168"

# JSON output for pipelines
./hopgrepX.py app.log --eq 12345 --json | jq .
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
