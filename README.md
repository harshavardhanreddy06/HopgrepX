# HopGrepX

HopGrepX is a high-performance, production-ready log search tool designed for rapidly querying large, sorted log files. It combines adaptive indexing (ZoneMap) with optimized K-ary search algorithms to deliver instant results for timestamp, numeric, and string-based queries.

**Version**: 1.0.0 (Production Release)

## 🚀 Features

- **Blazing Fast Search**: Uses K-ary search on sorted files to find data without scanning the whole file.
- **Adaptive Indexing**: Automatically learns file distribution and creates lightweight sidecar indices (`.hop.idx`), making subsequent searches 10-100x faster.
- **Multi-File Parallelism**: Native support for searching multiple files in parallel using wildcards (`*.log`) or comma-separated lists.
- **Smart Querying**:
    - **Equality**: Exact matches (`--eq`)
    - **Range**: Time/Number ranges (`--range`)
    - **Prefix**: String-based prefix matching (`--prefix`)
    - **Multi-Match**: Search for multiple exact keys at once (`--multi`)
    - **Implicit Substring**: Behaves like standard grep if no mode is specified.
- **SQL-Like Filtering**: Powerful `--where` clause for complex nested logic (`AND`, `OR`, `NOT`, `( )`).
- **JSON Output**: Structural output support (`--json`) for pipeline integration.
- **Auto-Detection**: Automatically detects key types (Timestamp, Number, or String).

## 📋 Requirements

- Python 3.6+
- No external dependencies (standard library only).

## 🔧 Installation

Simply download the `hopgrepX.py` script and make it executable:

```bash
curl -O https://raw.githubusercontent.com/yourusername/hopgrepX/main/hopgrepX.py
chmod +x hopgrepX.py
```

## 📖 Usage

### Basic Syntax

```bash
./hopgrepX.py <files> [options]
```

### File Specifications

HopGrepX supports flexible file arguments:
- **Single File**: `app.log`
- **Wildcards**: `logs/*.log`, `data-2024-*.txt`
- **Lists**: `access.log,error.log` (comma-separated)
- **Mixed**: `current.log,archive/2023-*.log`

### Search Modes

| Flag | Description | Example |
|------|-------------|---------|
| `(none)` | Substring search (linear scan or smart scan) | `./hopgrepX.py app.log "Connection failed"` |
| `--eq` | Exact key match | `./hopgrepX.py app.log --eq "2024-01-01 12:00:00"` |
| `--range` | Range search [start, end] | `./hopgrepX.py app.log --range 100 200` |
| `--prefix` | String prefix match | `./hopgrepX.py app.log --prefix "2024-01"` |
| `--multi` | Multiple exact matches | `./hopgrepX.py app.log --multi "ID:555,ID:999"` |

### Advanced Filtering (`--where`)

Filter results using fields extracted from the log line. Fields are auto-detected (`key=value`) or accessible by column index (`col:N`).

```bash
# Filter by extracted field 'level'
./hopgrepX.py app.log --where "level='ERROR'"

# Filter by column index 2 and 3
./hopgrepX.py app.log --where "col:2='CRITICAL' AND col:3=500"

# Complex logic
./hopgrepX.py app.log --where "(status=500 OR latency>1000) AND env='PROD'"
```

## ⚡ Performance & Indexing

1.  **First Run**: Performs a K-ary search or full scan. Simultaneously builds a sparse index in memory.
2.  **Indexing**: On completion (for single-file runs), saves a `.hop.idx` sidecar file properly handling file rotation/updates.
3.  **Subsequent Runs**: Loads the `.hop.idx` to jump directly to relevant file regions, drastically reducing I/O.
    *   *Note: Index updates are skipped during parallel multi-file searches for safety.*

## 💡 Examples

**1. Find specific error logs in a time range:**
```bash
./hopgrepX.py system.log \
  --range "2023-12-01 00:00:00" "2023-12-01 23:59:59" \
  --where "level='ERROR'"
```

**2. Search integer IDs across all archive logs:**
```bash
./hopgrepX.py "archive/data-*.log" --eq 987654321
```

**3. Pipe JSON output to jq:**
```bash
./hopgrepX.py app.log --prefix "Warn" --json | jq .
```

**4. Numeric Prefix Search:**
Note that `--prefix` is string-based.
```bash
./hopgrepX.py access.log --prefix "192.168"
# Matches: 192.168.0.1, 192.168.1.55
# Does NOT match: 1920.1.1.1
```
