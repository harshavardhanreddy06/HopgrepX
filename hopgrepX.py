#!/usr/bin/env python3
"""
hopgrepX - Fast sorted log file search with adaptive indexing
Version: 1.0.0 (Production Release)

Features:
- Fast k-ary search for sorted files (timestamp/number/string keys)
- Adaptive ZoneMap indexing (.hop.idx sidecar files)
- Parallel multi-file search
- SQL-like filter expressions (--where)
- JSON output mode (--json)

Usage:
  hopgrepX.py <files> <pattern>                     # Substring search
  hopgrepX.py <files> --eq <key>                    # Exact match
  hopgrepX.py <files> --range <start> <end>         # Range search
  hopgrepX.py <files> --prefix <prefix>             # Prefix search (string-based)
  hopgrepX.py <files> --multi <k1,k2,k3>            # Multiple exact matches

Note on numeric prefix search:
  --prefix "192" matches keys starting with "192" (string prefix)
  Examples: 192, 192.1, 192.168.0.1
  Does NOT match: 1920, 1192 (not string prefixes)
"""

import os
import sys
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Union
import glob
import multiprocessing
from multiprocessing import Pool, Lock
import fcntl
import tempfile
import contextlib

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

ZONEMAP_VERSION = 1
MAX_GLOB_FILES = 1000  # Safety limit for glob expansion
ZONEMAP_MAX_ZONES = 5000
ZONEMAP_TARGET_SIZE = 64 * 1024 * 1024  # 64MB
ZONEMAP_MAX_AGE_DAYS = 30  # Age out zones older than 30 days
VERSION = "1.0.0"

# ============================================================================
# GLOBALS & INITIALIZATION
# ============================================================================

_stdout_lock = None

def init_worker(lock):
    """Initialize worker process with shared lock."""
    global _stdout_lock
    _stdout_lock = lock


# ============================================================================
# 1. FAST PARSING & UTILS
# ============================================================================

class FastUtils:
    """Methods optimized for raw speed using BYTES processing."""
    
    @staticmethod
    def parse_timestamp(line_prefix: bytes) -> Optional[float]:
        """Parse timestamp from bytes, returns float or None."""
        if len(line_prefix) < 19: 
            return None
        try:
            y = int(line_prefix[0:4])
            m = int(line_prefix[5:7])
            d = int(line_prefix[8:10])
            H = int(line_prefix[11:13])
            M = int(line_prefix[14:16])
            S = int(line_prefix[17:19])
            dt = datetime(y, m, d, H, M, S)
            return dt.timestamp()
        except:
            return None

    @staticmethod
    def get_first_column_bytes(line: bytes) -> bytes:
        """Extract first column (key) from a log line."""
        if not line:
            return b""
        
        for i, char in enumerate(line):
            if char in b'| \t':
                return line[:i]
        return line

    @staticmethod
    def get_key_from_line(line: bytes, key_type: str) -> Union[float, bytes, None]:
        """Extract and parse key from first column."""
        if not line: 
            return None
        
        if key_type == 'timestamp':
            if len(line) >= 19:
                if len(line) > 19 and line[19] not in b' |\t\n\r':
                    first_col = FastUtils.get_first_column_bytes(line)
                    if len(first_col) >= 19:
                        return FastUtils.parse_timestamp(first_col[:19])
                    return FastUtils.parse_timestamp(first_col)
                return FastUtils.parse_timestamp(line[:19])
            return None
            
        elif key_type == 'number':
            first_col = FastUtils.get_first_column_bytes(line)
            try:
                return float(first_col)
            except:
                return None
        
        return FastUtils.get_first_column_bytes(line)

    @staticmethod
    def dt_to_float(s: str) -> float:
        """Parse various timestamp formats to float."""
        try: 
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()
        except: 
            pass
        
        fmt_map = {
            4: "%Y", 7: "%Y-%m", 10: "%Y-%m-%d",
            13: "%Y-%m-%d %H", 16: "%Y-%m-%d %H:%M",
        }
        
        if len(s) in fmt_map:
            try:
                return datetime.strptime(s, fmt_map[len(s)]).timestamp()
            except:
                pass
        
        try: 
            return float(s)
        except: 
            raise ValueError(f"Cannot parse timestamp: {s}")


# ============================================================================
# 2. FILTER LOGIC (SQL-like)
# ============================================================================

class FilterLogic:
    """SQL-like filter expression evaluator."""
    
    TOK_ID, TOK_OP, TOK_VAL = 'ID', 'OP', 'VAL'
    TOK_LPAREN, TOK_RPAREN = 'LPAREN', 'RPAREN'
    TOK_AND, TOK_OR, TOK_NOT = 'AND', 'OR', 'NOT'
    TOK_EOF = 'EOF'

    def __init__(self, expression: str):
        self.tokens = self.tokenize(expression)
        self.pos = 0
        try:
            self.ast = self.parse_expression()
        except Exception as e:
            raise ValueError(f"Invalid filter expression: {e}")

    def tokenize(self, text: str):
        """Tokenize filter expression with escape handling."""
        tokens = []
        i = 0
        n = len(text)
        ops = {'>=', '<=', '!=', '=', '>', '<', '=='}
        
        while i < n:
            c = text[i]
            
            if c.isspace(): 
                i += 1
                continue
            
            if c == '(':
                tokens.append((self.TOK_LPAREN, '('))
                i += 1
                continue
            if c == ')':
                tokens.append((self.TOK_RPAREN, ')'))
                i += 1
                continue
            
            if i + 2 <= n and text[i:i+2] in ops:
                tokens.append((self.TOK_OP, text[i:i+2]))
                i += 2
                continue
            if text[i:i+1] in ops:
                tokens.append((self.TOK_OP, text[i:i+1]))
                i += 1
                continue
            
            if c in '"\'':
                q = c
                val = ""
                i += 1
                while i < n:
                    if text[i] == '\\' and i + 1 < n:
                        i += 1
                        val += text[i]
                    elif text[i] == q:
                        break
                    else:
                        val += text[i]
                    i += 1
                i += 1
                tokens.append((self.TOK_VAL, val))
                continue
            
            if c.isalnum() or c in '_-.:':
                start = i
                while i < n and (text[i].isalnum() or text[i] in '_-./:'):
                    i += 1
                w = text[start:i]
                u = w.upper()
                
                if u == 'AND':
                    tokens.append((self.TOK_AND, 'AND'))
                elif u == 'OR':
                    tokens.append((self.TOK_OR, 'OR'))
                elif u == 'NOT':
                    tokens.append((self.TOK_NOT, 'NOT'))
                else:
                    tokens.append((self.TOK_VAL, w))
                continue
            
            raise ValueError(f"Invalid character '{c}' at position {i}")
        
        tokens.append((self.TOK_EOF, ''))
        return tokens

    def peek(self):
        return self.tokens[self.pos]

    def consume(self, expected=None):
        t = self.tokens[self.pos]
        if expected and t[0] != expected:
            raise ValueError(f"Expected {expected}, got {t}")
        self.pos += 1
        return t

    def parse_expression(self):
        left = self.parse_or()
        while self.peek()[0] == self.TOK_OR:
            self.consume()
            right = self.parse_or()
            left = ('OR', left, right)
        return left

    def parse_or(self):
        left = self.parse_and()
        while self.peek()[0] == self.TOK_AND:
            self.consume()
            right = self.parse_and()
            left = ('AND', left, right)
        return left

    def parse_and(self):
        t = self.peek()
        if t[0] == self.TOK_NOT:
            self.consume()
            return ('NOT', self.parse_and())
        if t[0] == self.TOK_LPAREN:
            self.consume()
            n = self.parse_expression()
            self.consume(self.TOK_RPAREN)
            return n
        return self.parse_condition()

    def parse_condition(self):
        k = self.consume(self.TOK_VAL)
        op = self.consume(self.TOK_OP)
        v = self.consume(self.TOK_VAL)
        return ('BIN', op[1], k[1], v[1])

    def evaluate(self, ctx, node=None):
        if not node:
            node = self.ast
            
        typ = node[0]
        
        if typ == 'AND':
            return self.evaluate(ctx, node[1]) and self.evaluate(ctx, node[2])
        if typ == 'OR':
            return self.evaluate(ctx, node[1]) or self.evaluate(ctx, node[2])
        if typ == 'NOT':
            return not self.evaluate(ctx, node[1])
        if typ == 'BIN':
            op, k, target = node[1], node[2], node[3]
            actual = ctx.get(k)
            
            if actual is None:
                return False
            
            try:
                a, b = float(actual), float(target)
                if op in ['=', '==']:
                    return abs(a - b) < 0.000001
                if op == '!=':
                    return abs(a - b) >= 0.000001
                if op == '>':
                    return a > b
                if op == '<':
                    return a < b
                if op == '>=':
                    return a >= b
                if op == '<=':
                    return a <= b
            except:
                a, b = str(actual), str(target)
                if op in ['=', '==']:
                    return a == b
                if op == '!=':
                    return a != b
                if op == '>':
                    return a > b
                if op == '<':
                    return a < b
                if op == '>=':
                    return a >= b
                if op == '<=':
                    return a <= b
            
        return False


def extract_context(line: str) -> dict:
    """Extract fields from log line for filtering."""
    d = {}
    
    if '|' in line:
        parts = [p.strip() for p in line.split('|')]
    else:
        parts = line.split()
    
    for idx, p in enumerate(parts):
        d[f"col:{idx}"] = p
    
    all_tokens = ' '.join(parts).split()
    for token in all_tokens:
        if '=' in token:
            try:
                k, v = token.split('=', 1)
                d[k.strip()] = v.strip()
            except:
                pass
    
    return d


# ============================================================================
# 3. ZONE MAP Metadata Manager (HARDENED)
# ============================================================================

class ZoneMapLite:
    """
    Manages sidecar .hop.idx file for scanned regions.
    Implements versioning, aging, and safe parallel access.
    """
    
    # Configuration
    MAX_KEY_GAP_TS = 600.0
    MAX_KEY_GAP_NUM = 5000.0
    PADDING = 4 * 1024 * 1024
    
    TS_TOLERANCE = 120.0
    NUM_TOLERANCE = 0.001
    
    TARGET_ZONE_SIZE = 64 * 1024 * 1024
    MAX_MERGE_DIST = 16 * 1024 * 1024
    MAX_ZONES = 5000

    def __init__(self, log_path: str, read_only: bool = False):
        self.log_path = log_path
        self.idx_path = log_path + ".hop.idx"
        self.zones = []
        self.loaded = False
        self.read_only = read_only  # Prevent writes in parallel mode
        self.file_stats = self._get_file_stats()

    def _get_file_stats(self):
        """Get file statistics with head+tail hashes for robust validation."""
        try:
            st = os.stat(self.log_path)
            
            with open(self.log_path, "rb") as f:
                # Head hash (first 4KB)
                head = f.read(4096)
                head_hash = hashlib.sha256(head).hexdigest()[:16]
                
                # Tail hash (last 4KB) - important for log rotation detection
                if st.st_size > 4096:
                    f.seek(-4096, os.SEEK_END)
                    tail = f.read(4096)
                else:
                    tail = head
                tail_hash = hashlib.sha256(tail).hexdigest()[:16]
            
            return {
                "size": st.st_size,
                "mtime": st.st_mtime,
                "inode": st.st_ino,
                "head_hash": head_hash,
                "tail_hash": tail_hash,
            }
        except:
            return None

    @contextlib.contextmanager
    def _lock_file(self, path, mode='r'):
        """Context manager for file locking."""
        try:
            f = open(path, mode)
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                yield f
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
        except (IOError, OSError):
            yield None

    def load(self):
        """Load and validate index file with version and rotation checking."""
        if not os.path.exists(self.idx_path):
            return
            
        try:
            with self._lock_file(self.idx_path, 'r') as f:
                if f is None:
                    return
                data = json.load(f)
            
            # Version check
            version = data.get("version", 0)
            if version != ZONEMAP_VERSION:
                return  # Incompatible version
            
            sig = data.get("file_signature", {})
            if not self.file_stats:
                return
            
            # Allow index for files that have only grown (appended)
            current_size = self.file_stats["size"]
            cached_size = sig.get("size", 0)
            
            if cached_size > current_size:
                return  # File got smaller
            
            # For large mtime differences, verify file hasn't been replaced or rotated
            current_mtime = self.file_stats["mtime"]
            cached_mtime = sig.get("mtime", 0)
            if abs(current_mtime - cached_mtime) > 3600:
                # Check both head and tail hashes for rotation detection
                if (sig.get("head_hash") != self.file_stats.get("head_hash") or
                    sig.get("tail_hash") != self.file_stats.get("tail_hash")):
                    return
            
            self.zones = data.get("zones", [])
            self.zones.sort(key=lambda x: x["start"])
            self.loaded = True
        except (json.JSONDecodeError, IOError, KeyError):
            pass

    def get_search_hints(self, target_key, key_type: str) -> Tuple[int, Optional[int]]:
        """Get search hints with safety padding."""
        if not self.loaded or not self.zones or target_key is None:
            return 0, None

        def to_runtime(val):
            try:
                if key_type == 'string':
                    if isinstance(val, str):
                        return val.encode('iso-8859-1')
                    return val
                return val
            except:
                return val

        found_s, found_e = 0, None
        found_hit = False

        for z in self.zones:
            z_min = to_runtime(z["min"])
            z_max = to_runtime(z["max"])
            try:
                if z_min <= target_key <= z_max:
                    found_s, found_e = z["start"], z["end"]
                    found_hit = True
                    break
            except:
                pass

        if not found_hit:
            best_low = 0
            best_high = None

            for z in self.zones:
                z_min = to_runtime(z["min"])
                z_max = to_runtime(z["max"])
                try:
                    if z_max < target_key:
                        if z["end"] > best_low:
                            best_low = z["end"]
                    if z_min > target_key:
                        if best_high is None or z["start"] < best_high:
                            best_high = z["start"]
                except:
                    pass
            
            found_s, found_e = best_low, best_high

        final_s = max(0, found_s - self.PADDING)
        final_e = None
        if found_e is not None:
            final_e = found_e + self.PADDING
            file_sz = self.file_stats["size"] if self.file_stats else float('inf')
            if final_e > file_sz:
                final_e = file_sz

        return final_s, final_e

    def add_observation(self, start: int, end: int, min_k, max_k, key_type: str):
        """Record scanned region with tolerance."""
        if self.read_only:
            return  # Skip in parallel mode
            
        if min_k is None or max_k is None:
            return
            
        if isinstance(min_k, bytes):
            min_k = min_k.decode('iso-8859-1')
        if isinstance(max_k, bytes):
            max_k = max_k.decode('iso-8859-1')

        try:
            if key_type == 'timestamp':
                min_k -= self.TS_TOLERANCE
                max_k += self.TS_TOLERANCE
            elif key_type == 'number':
                min_k -= self.NUM_TOLERANCE
                max_k += self.NUM_TOLERANCE
        except:
            pass
        
        self.zones.append({
            "start": start,
            "end": end,
            "min": min_k,
            "max": max_k,
            "last_used": time.time()  # For aging policies
        })

    def _merge_two_zones(self, z1: dict, z2: dict):
        """Merge z2 into z1 with proper error handling."""
        z1["end"] = max(z1["end"], z2["end"])
        try:
            if z2["min"] < z1["min"]:
                z1["min"] = z2["min"]
            if z2["max"] > z1["max"]:
                z1["max"] = z2["max"]
        except:
            pass

        z1["last_used"] = max(
            z1.get("last_used", 0),
            z2.get("last_used", 0)
        )

    def _should_merge(self, z1: dict, z2: dict, force: bool = False) -> bool:
        byte_gap = z2["start"] - z1["end"]
        
        if byte_gap <= 0:
            return True
        
        if force:
            curr_size = z1["end"] - z1["start"]
            return curr_size < self.TARGET_ZONE_SIZE

        curr_size = z1["end"] - z1["start"]
        if byte_gap < self.MAX_MERGE_DIST and curr_size < self.TARGET_ZONE_SIZE:
            try:
                if (isinstance(z1["max"], (int, float)) and 
                    isinstance(z2["min"], (int, float))):
                    k_gap = z2["min"] - z1["max"]
                    if k_gap > self.MAX_KEY_GAP_NUM * 10:
                        return False
                    if 1000000000 < z1["max"] < 3000000000:
                        if k_gap > self.MAX_KEY_GAP_TS * 10:
                            return False
            except:
                pass
            return True
        
        return False

    def _compact_zones(self, zones: List[dict], force_mode: bool = False) -> List[dict]:
        if not zones:
            return []
            
        compacted = []
        curr = zones[0]
        
        for i in range(1, len(zones)):
            nxt = zones[i]
            if self._should_merge(curr, nxt, force=force_mode):
                self._merge_two_zones(curr, nxt)
            else:
                compacted.append(curr)
                curr = nxt
                
        compacted.append(curr)
        return compacted

    def commit(self):
        """Merge zones and write to disk atomically with aging policy."""
        if self.read_only:
            return  # No writes in parallel mode
            
        if not self.zones:
            return
        
        # Apply aging policy: remove zones older than MAX_AGE_DAYS
        now = time.time()
        max_age = ZONEMAP_MAX_AGE_DAYS * 24 * 3600
        
        self.zones = [
            z for z in self.zones
            if now - z.get("last_used", now) < max_age
        ]
        
        # Sort by start position
        self.zones.sort(key=lambda x: x["start"])
        
        # Normal compaction
        self.zones = self._compact_zones(self.zones, force_mode=False)
        
        # Cleanup: remove small zones (<1MB)
        self.zones = [z for z in self.zones if (z["end"] - z["start"]) > 1_000_000]
        
        # Hard cap enforcement
        if len(self.zones) > self.MAX_ZONES:
            self.zones = self._compact_zones(self.zones, force_mode=True)
        
        # Final safety cap
        if len(self.zones) > self.MAX_ZONES:
            self.zones = self.zones[-self.MAX_ZONES:]
        
        # Update file stats before writing
        self.file_stats = self._get_file_stats()
        
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(self.idx_path) or ".",
                suffix='.tmp',
                prefix=os.path.basename(self.idx_path) + '.'
            )
            
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                data = {
                    "version": ZONEMAP_VERSION,
                    "file_signature": self.file_stats,
                    "zones": self.zones
                }
                json.dump(data, f, separators=(',', ':'))  # Minimal JSON
                f.flush()
                os.fsync(f.fileno())
            
            os.replace(tmp_path, self.idx_path)
        except (IOError, OSError, json.JSONEncodeError):
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except:
                pass


# ============================================================================
# 4. REGION LOCATOR (HYPER-HOP)
# ============================================================================

class HyperHopSeeker:
    """K-ary search for finding file bounds efficiently."""
    
    def __init__(self, filepath: str, key_type: str = 'timestamp'):
        self.filepath = filepath
        self.key_type = key_type
        self.f = None
        self.filesize = 0
        self._open_file()
        
        self.K_FACTOR = 32
        self.SCAN_LIMIT = 256 * 1024

    def _open_file(self):
        """Open file and get size."""
        if self.f is not None:
            self.f.close()
        self.f = open(self.filepath, "rb")
        self.f.seek(0, 2)
        self.filesize = self.f.tell()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close file handle."""
        if self.f:
            self.f.close()
            self.f = None

    def raw_read_line_at(self, offset: int) -> Tuple[Optional[Union[float, bytes]], int]:
        if offset >= self.filesize:
            return None, offset
            
        self.f.seek(offset)
        if offset > 0:
            self.f.readline()
            
        aligned_pos = self.f.tell()
        line = self.f.readline()
        
        if not line:
            return None, aligned_pos
            
        k = FastUtils.get_key_from_line(line.strip(), self.key_type)
        return k, aligned_pos

    def locate_lower_bound(self, target, hint_start: int = 0, hint_end: Optional[int] = None) -> int:
        if target is None or target == float('-inf') or target == '':
            return 0
        
        low = hint_start
        high = hint_end if hint_end is not None else self.filesize
        
        if high > self.filesize:
            high = self.filesize
        if low > high:
            low = high

        while (high - low) > self.SCAN_LIMIT:
            chunk = (high - low) / self.K_FACTOR
            probes = [int(low + i * chunk) for i in range(1, self.K_FACTOR)]
            
            valid_samples = []
            for p_off in probes:
                k, aligned = self.raw_read_line_at(p_off)
                if k is not None:
                    valid_samples.append((aligned, k))
            
            if not valid_samples:
                break

            idx_ge = -1
            for i, (off, k) in enumerate(valid_samples):
                if k >= target:
                    idx_ge = i
                    break
            
            new_low = low
            new_high = high
            
            if idx_ge == 0:
                new_high = valid_samples[0][0]
            elif idx_ge > 0:
                new_low = valid_samples[idx_ge-1][0]
                new_high = valid_samples[idx_ge][0]
            else:
                new_low = valid_samples[-1][0]
            
            if new_low == low and new_high == high:
                break
                
            low = new_low
            high = new_high
        
        return low


# ============================================================================
# 5. HELPER FUNCTIONS
# ============================================================================

class QueryRule:
    def __init__(self, start_key, end_key):
        self.start_key = start_key
        self.end_key = end_key


def merge_byte_regions(regions: List[Tuple[int, int, List[QueryRule]]], 
                       safety_window: int) -> List[Tuple[int, int, List[QueryRule]]]:
    if not regions:
        return []
        
    regions.sort(key=lambda x: x[0])
    merged = []
    curr_start, curr_end, curr_rules = regions[0]
    
    for i in range(1, len(regions)):
        next_start, next_end, next_rules = regions[i]
        
        if next_start <= curr_end + safety_window:
            curr_end = max(curr_end, next_end)
            curr_rules.extend(next_rules)
        else:
            merged.append((curr_start, curr_end, curr_rules))
            curr_start, curr_end, curr_rules = next_start, next_end, next_rules
            
    merged.append((curr_start, curr_end, curr_rules))
    return merged


def next_epsilon(val, key_type: str):
    if key_type == 'number':
        return val + 0.000001
    elif key_type == 'timestamp':
        return val + 0.000001
    else:
        if isinstance(val, str):
            val = val.encode('utf-8')
        return val + b'\0'


def detect_key_type_from_file(filepath: str) -> str:
    try:
        with open(filepath, 'rb') as f:
            line = f.readline()
            while line:
                line = line.strip()
                if line:
                    first_col = FastUtils.get_first_column_bytes(line)
                    if not first_col:
                        line = f.readline()
                        continue
                    
                    try:
                        first_col_str = first_col.decode('utf-8')
                    except UnicodeDecodeError:
                        first_col_str = first_col.decode('latin-1', errors='replace')
                    
                    if len(first_col_str) >= 10 and first_col_str[4] == '-' and first_col_str[7] == '-':
                        return "timestamp"
                    
                    try:
                        float(first_col_str)
                        return "number"
                    except:
                        pass
                    
                    break  # Only need first valid line
                line = f.readline()
    except:
        pass
    
    return "string"


# ============================================================================
# 6. MAIN SEARCH ENGINE (HARDENED)
# ============================================================================

def search_file_generator(filepath: str, mode: str, keys: List[str], filter_expr: Optional[str], json_mode: bool = False, is_parallel: bool = False):
    """
    Pure generator for searching a file.
    
    Args:
        is_parallel: True if running in parallel mode (disables ZoneMap writes)
    """
    
    key_type = detect_key_type_from_file(filepath)
    
    seeker = None
    try:
        seeker = HyperHopSeeker(filepath, key_type)
        file_size = seeker.filesize
        
        # ZoneMap is READ-ONLY in parallel mode
        zonemap = ZoneMapLite(filepath, read_only=is_parallel)
        zonemap.load()
        
        queries = []
        
        if mode == "--substring":
            queries.append(QueryRule(None, None))
            
        elif mode == "--eq":
            for k_str in keys:
                if key_type == "timestamp":
                    v = FastUtils.dt_to_float(k_str)
                    queries.append(QueryRule(v, next_epsilon(v, 'timestamp')))
                elif key_type == "number":
                    v = float(k_str)
                    queries.append(QueryRule(v, next_epsilon(v, 'number')))
                else:
                    v_b = k_str.encode('utf-8')
                    queries.append(QueryRule(v_b, next_epsilon(v_b, 'string')))
                    
        elif mode == "--range":
            s_str, e_str = keys[0], keys[1]
            if key_type == "timestamp":
                s = FastUtils.dt_to_float(s_str)
                e = FastUtils.dt_to_float(e_str)
                queries.append(QueryRule(s, next_epsilon(e, 'timestamp')))
            elif key_type == "number":
                s, e = float(s_str), float(e_str)
                queries.append(QueryRule(s, next_epsilon(e, 'number')))
            else:
                s_b, e_b = s_str.encode('utf-8'), e_str.encode('utf-8')
                queries.append(QueryRule(s_b, next_epsilon(e_b, 'string')))
                
        elif mode == "--prefix":
            s_str = keys[0]
            prefix_str = s_str
            prefix_bytes = s_str.encode('utf-8')
            
            # IMPORTANT: Prefix matching is STRING-BASED for all key types
            # --prefix "192" matches: 192, 192.1, 192.168.0.1 (string prefix)
            # Does NOT match: 1920, 1192 (not string prefixes)
            
            if key_type == "timestamp":
                try:
                    prefix = s_str
                    if len(prefix) == 4:
                        y = int(prefix)
                        start = datetime(y, 1, 1).timestamp()
                        end = datetime(y + 1, 1, 1).timestamp()
                    elif len(prefix) == 7:
                        y, m = map(int, prefix.split('-'))
                        start = datetime(y, m, 1).timestamp()
                        end = datetime(y, m + 1, 1).timestamp() if m < 12 else datetime(y + 1, 1, 1).timestamp()
                    elif len(prefix) == 10:
                        y, m, d = map(int, prefix.split('-'))
                        start = datetime(y, m, d).timestamp()
                        end = start + 86400
                    elif len(prefix) == 13:
                        y, m, d = map(int, prefix[:10].split('-'))
                        h = int(prefix[11:13])
                        start = datetime(y, m, d, h).timestamp()
                        end = start + 3600
                    elif len(prefix) == 16:
                        y, m, d = map(int, prefix[:10].split('-'))
                        h = int(prefix[11:13])
                        mm = int(prefix[14:16])
                        start = datetime(y, m, d, h, mm).timestamp()
                        end = start + 60
                    else:
                        start = FastUtils.dt_to_float(prefix)
                        end = None
                    
                    queries.append(QueryRule(start, end))
                except:
                    queries.append(QueryRule(0, None))
                    
            elif key_type == "number":
                queries.append(QueryRule(0, None))
                    
            else:
                queries.append(QueryRule(prefix_bytes, None))
                 
        elif mode == "--multi":
            for k_str in keys:
                if key_type == "timestamp":
                    v = FastUtils.dt_to_float(k_str)
                    queries.append(QueryRule(v, next_epsilon(v, 'timestamp')))
                elif key_type == "number":
                    v = float(k_str)
                    queries.append(QueryRule(v, next_epsilon(v, 'number')))
                else:
                    v_b = k_str.encode('utf-8')
                    queries.append(QueryRule(v_b, next_epsilon(v_b, 'string')))

        raw_regions = []
        for q in queries:
            hs_start, he_start = zonemap.get_search_hints(q.start_key, key_type)
            start_bound = seeker.locate_lower_bound(q.start_key, hs_start, he_start)
            
            if q.end_key is None:
                end_bound = file_size
            else:
                hs_end, he_end = zonemap.get_search_hints(q.end_key, key_type)
                end_bound = seeker.locate_lower_bound(q.end_key, hs_end, he_end)
                
            raw_regions.append((start_bound, end_bound, [q]))

        SAFETY_WINDOW = 1024 * 1024
        merged_regions = merge_byte_regions(raw_regions, SAFETY_WINDOW)
        
        filter_eng = FilterLogic(filter_expr) if filter_expr else None
        
        # Track if we should commit ZoneMap (only if successful scan)
        match_count = 0
        
        if mode == "--substring" and not keys:
            seeker.f.seek(0)
            while True:
                curr_offset = seeker.f.tell()
                line_b = seeker.f.readline()
                if not line_b:
                    break
                    
                if filter_eng or json_mode:
                    try:
                        line_str = line_b.decode('utf-8')
                    except UnicodeDecodeError:
                        line_str = line_b.decode('latin-1', errors='replace')
                    line_str = line_str.rstrip('\n\r')
                    
                    ctx = extract_context(line_str)
                    
                    if filter_eng and not filter_eng.evaluate(ctx):
                        continue
                        
                    match_count += 1
                    
                    if json_mode:
                        fields = {k: v for k, v in ctx.items() if not k.startswith("col:")}
                        json_obj = {
                            "file": filepath,
                            "offset": curr_offset,
                            "key": ctx.get("col:0", ""),
                            "fields": fields,
                            "_raw": line_str
                        }
                        yield (json.dumps(json_obj, ensure_ascii=False) + "\n").encode('utf-8')
                    else:
                        yield line_b
                else:
                    match_count += 1
                    yield line_b
        
        else:
            for (safe_start, safe_end_anchor, rules) in merged_regions:
                seeker.f.seek(safe_start)
                if safe_start > 0:
                    seeker.f.readline()
                
                max_logical_key_rule = None
                has_open_end = False
                for r in rules:
                    if r.end_key is None:
                        has_open_end = True
                    elif max_logical_key_rule is None or r.end_key > max_logical_key_rule:
                        max_logical_key_rule = r.end_key
                
                safety_cutoff = safe_end_anchor + SAFETY_WINDOW
                if safety_cutoff > file_size:
                    safety_cutoff = file_size
                
                q_sub_list = [k.encode('utf-8') for k in keys] if (mode == "--substring" and keys) else []
                prefix_str = str(keys[0]) if (mode == "--prefix" and keys) else ""
                prefix_bytes = prefix_str.encode('utf-8') if prefix_str else b""
                
                region_min_k = None
                region_max_k = None
                region_start_pos = seeker.f.tell()
                
                while True:
                    curr_pos = seeker.f.tell()
                    if curr_pos >= safety_cutoff:
                        break
                    
                    line_bytes = seeker.f.readline()
                    if not line_bytes:
                        break
                    
                    line_stripped = line_bytes.strip()
                    if not line_stripped:
                        continue

                    k = FastUtils.get_key_from_line(line_stripped, key_type)
                    
                    if k is not None:
                        if region_min_k is None:
                            region_min_k = region_max_k = k
                        else:
                            if k < region_min_k: region_min_k = k
                            if k > region_max_k: region_max_k = k
                    
                    # Termination checks with safe handling
                    if mode in ["--eq", "--range", "--multi"]:
                        if not has_open_end and k is not None and max_logical_key_rule is not None:
                            if k > max_logical_key_rule:
                                break
                    
                    elif mode == "--prefix":
                        # Safe termination for numeric/timestamp prefixes
                        if max_logical_key_rule is not None and k is not None:
                            if key_type in ("number", "timestamp"):
                                if k >= max_logical_key_rule:
                                    break
                    
                    is_match = False
                    if mode == "--substring":
                        for qs in q_sub_list:
                            if qs in line_bytes:
                                is_match = True
                                break
                    elif mode == "--prefix":
                        if key_type == "string":
                            if isinstance(k, bytes):
                                is_match = k.startswith(prefix_bytes)
                        elif key_type == "timestamp":
                            is_match = line_stripped.startswith(prefix_bytes)
                        elif key_type == "number":
                            first_col = FastUtils.get_first_column_bytes(line_stripped)
                            is_match = first_col.startswith(prefix_bytes)
                    else:
                        if k is not None:
                            for r in rules:
                                if (r.start_key is None or k >= r.start_key) and \
                                   (r.end_key is None or k < r.end_key):
                                    is_match = True
                                    break
                    
                    if is_match:
                        if filter_eng or json_mode:
                            try:
                                line_str = line_stripped.decode('utf-8')
                            except UnicodeDecodeError:
                                line_str = line_stripped.decode('latin-1', errors='replace')
                            
                            ctx = extract_context(line_str)
                            if filter_eng and not filter_eng.evaluate(ctx):
                                continue
                            
                            match_count += 1
                            
                            if json_mode:
                                fields = {key: v for key, v in ctx.items() if not key.startswith("col:")}
                                json_obj = {
                                    "file": filepath,
                                    "offset": curr_pos,
                                    "key": ctx.get("col:0", ""),
                                    "fields": fields,
                                    "_raw": line_str
                                }
                                yield (json.dumps(json_obj, ensure_ascii=False) + "\n").encode('utf-8')
                            else:
                                yield line_bytes
                        else:
                            match_count += 1
                            yield line_bytes
                
                if region_min_k is not None and region_max_k is not None:
                    if mode not in ["--prefix", "--substring"]:
                        zonemap.add_observation(region_start_pos, seeker.f.tell(), 
                                              region_min_k, region_max_k, key_type)
        
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        pass
    finally:
        if seeker is not None:
            seeker.close()
        
        # Only commit if:
        # 1. Not in parallel mode (read_only handles this)
        # 2. We had successful matches
        # 3. Not a prefix/substring search (no ZoneMap updates)
        if match_count > 0 and not is_parallel and mode not in ["--prefix", "--substring"]:
            zonemap.commit()


# ============================================================================
# 7. WORKER & FILE MANAGEMENT
# ============================================================================

def run_worker_task(args):
    """Worker entry point for parallel execution."""
    filepath, mode, keys, filter_expr, json_mode = args
    buffer = []
    current_size = 0
    BUF_LIMIT = 32 * 1024
    
    match_count = 0
    start_time = time.time()
    
    try:
        # Note: is_parallel=True disables ZoneMap writes
        for chunk in search_file_generator(filepath, mode, keys, filter_expr, json_mode, is_parallel=True):
            match_count += 1
            
            buffer.append(chunk)
            current_size += len(chunk)
            
            if current_size >= BUF_LIMIT:
                with _stdout_lock:
                    sys.stdout.buffer.write(b''.join(buffer))
                    sys.stdout.buffer.flush()
                buffer.clear()
                current_size = 0
        
        if buffer:
            with _stdout_lock:
                sys.stdout.buffer.write(b''.join(buffer))
                sys.stdout.buffer.flush()
                
    except BrokenPipeError:
        pass
    except Exception as e:
        sys.stderr.write(f"Error processing {filepath}: {e}\n")
        
    dur = time.time() - start_time
    sys.stderr.write(f"[{os.path.basename(filepath)}] Found {match_count} matches in {dur:.4f}s\n")


def expand_target_files(arg: str) -> List[str]:
    """
    Expand file specification to a list of file paths with safety limits.
    
    Returns:
        List of file paths (max MAX_GLOB_FILES)
    """
    paths = []
    
    parts = [p.strip() for p in arg.split(',')]
    
    for part in parts:
        if not part:
            continue
            
        if any(ch in part for ch in "*?[]"):
            try:
                matches = glob.glob(part, recursive=True)
                total_matches = len(matches)
                if total_matches > MAX_GLOB_FILES:
                    sys.stderr.write(
                        f"Warning: Pattern '{part}' matched {total_matches} files, "
                        f"limiting to first {MAX_GLOB_FILES}\n"
                    )
                    matches = matches[:MAX_GLOB_FILES]
                paths.extend(matches)
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to expand pattern '{part}': {e}\n")
                continue
        elif os.path.isdir(part):
            try:
                # List files in directory (non-recursive for safety)
                matches = [os.path.join(part, f) for f in os.listdir(part) 
                          if os.path.isfile(os.path.join(part, f))]
                total_matches = len(matches)
                if total_matches > MAX_GLOB_FILES:
                    sys.stderr.write(
                        f"Warning: Directory '{part}' contains {total_matches} files, "
                        f"limiting to first {MAX_GLOB_FILES}\n"
                    )
                    matches = matches[:MAX_GLOB_FILES]
                paths.extend(matches)
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to list directory '{part}': {e}\n")
                continue
        else:
            paths.append(part)
    
    # Deduplicate and filter
    paths = sorted(list(set(paths)))
    paths = [p for p in paths if os.path.isfile(p)]
    
    return paths[:MAX_GLOB_FILES]  # Final safety cap


# ============================================================================
# 8. CLI INTERFACE (HARDENED)
# ============================================================================

def print_help():
    """Print comprehensive help information."""
    print(__doc__)
    print("""
File Specification:
  - Single file: 'logs.txt'
  - Wildcards: 'test*.log', 'logs/**/*.log'
  - Comma-separated: 'file1.log,file2.log,file3.log'
  - Mixed: 'specific.log,archive/*.log' (combines both)
    
Filters:
  --where "field=value"                           # Filter results
  --where "col:0=ERROR"                           # Filter by column (0-based)
  --where "severity=ERROR AND user=admin"         # Multiple conditions
  --json                                          # Output matches as JSON objects
    
Important Notes:
  1. First column must be primary key (timestamp/number/string)
  2. Log file must be sorted by primary key
  3. Auto-detects key type from file content
  4. Creates .hop.idx index file for faster repeated searches
  5. Multi-file searches run in parallel automatically
  6. ZoneMap index updates are DISABLED in parallel mode
  7. Numeric prefix search is string-based (not numeric range)
    
Examples:
  hopgrepX.py logs.txt ERROR                      # Find "ERROR" anywhere
  hopgrepX.py logs.txt --eq "2024-01-15 10:30:00" # Exact timestamp
  hopgrepX.py logs.txt --range "2024-01-15" "2024-01-16"
  hopgrepX.py logs.txt --prefix "2024-01"         # January 2024 logs
  hopgrepX.py logs.txt --where "severity=ERROR"   # Filtered search
  hopgrepX.py 'app*.log' --eq 12345               # Search multiple files (wildcard)
  hopgrepX.py app1.log,app2.log --eq 12345        # Search multiple files (comma-separated)
    
Performance Tips:
  - First run creates index (.hop.idx file)
  - Subsequent runs are 10-100x faster
  - For best results, run single-file searches first to build index
  - Parallel mode disables index writes (run single-file to build index)
""")


def main():
    """Main CLI entry point with platform-aware multiprocessing."""
    
    # 1. Version Flag
    if "--version" in sys.argv:
        print(f"hopgrepx {VERSION}")
        sys.exit(0)

    # 2. Short Flags
    flag_map = {
        "-e": "--eq",
        "-r": "--range",
        "-p": "--prefix",
        "-m": "--multi",
        "-j": "--json",
        "-w": "--where",
    }
    sys.argv = [flag_map.get(a, a) for a in sys.argv]
    
    if len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']:
        print_help()
        sys.exit(0)
    
    if len(sys.argv) < 3:
        print("Error: Insufficient arguments")
        print_help()
        sys.exit(1)
        
    json_mode = False
    if "--json" in sys.argv:
        json_mode = True
        sys.argv.remove("--json")
        
    if len(sys.argv) < 3:
        print("Error: Insufficient arguments")
        print_help()
        sys.exit(1)
    
    target_arg = sys.argv[1]
    
    target_files = expand_target_files(target_arg)
    if not target_files:
        print(f"Error: No files found matching '{target_arg}'")
        sys.exit(1)
    
    second_arg = sys.argv[2]
    
    if second_arg.startswith("--"):
        mode = second_arg
        if mode not in ['--eq', '--range', '--prefix', '--multi', '--substring']:
            print(f"Error: Unknown mode '{mode}'")
            print_help()
            sys.exit(1)
        
        remaining_args = sys.argv[3:]
        keys = []
        where_expr = None
        
        i = 0
        while i < len(remaining_args):
            arg = remaining_args[i]
            if arg == "--where":
                if i + 1 < len(remaining_args):
                    where_expr = remaining_args[i + 1]
                    i += 2
                    continue
            keys.append(arg)
            i += 1
    else:
        mode = "--substring"
        remaining_args = sys.argv[2:]
        keys = []
        where_expr = None
        
        i = 0
        while i < len(remaining_args):
            arg = remaining_args[i]
            if arg == "--where":
                if i + 1 < len(remaining_args):
                    where_expr = remaining_args[i + 1]
                    i += 2
                    continue
            keys.append(arg)
            i += 1
    
    if mode == "--range" and len(keys) != 2:
        print("Error: --range requires exactly two arguments")
        sys.exit(1)
    
    if mode == "--multi" and len(keys) == 1:
        keys = keys[0].split(',')
        keys = [k.strip() for k in keys if k.strip()]
        
    if mode == "--multi" and not keys:
        print("Error: --multi requires comma-separated list")
        sys.exit(1)
    
    try:
        if len(target_files) > 1:
            sys.stderr.write(f"Searching {len(target_files)} files in parallel (index writes disabled)...\n")
            sys.stderr.write("Note: Run single-file searches first to build optimal indexes.\n")
            
            tasks = []
            for f in target_files:
                tasks.append((f, mode, keys, where_expr, json_mode))
            
            # Platform-aware multiprocessing setup
            if sys.platform == "darwin":
                # macOS: Use 'spawn' for safety (Apple discourages fork)
                try:
                    multiprocessing.set_start_method('spawn')
                except RuntimeError:
                    pass  # Already set
            else:
                # Linux/Unix: Use 'fork' for performance
                try:
                    multiprocessing.set_start_method('fork', force=True)
                except RuntimeError:
                    pass
            
            lock = Lock()
            with Pool(processes=min(multiprocessing.cpu_count(), len(target_files)), 
                     initializer=init_worker, initargs=(lock,)) as pool:
                for _ in pool.imap_unordered(run_worker_task, tasks):
                    pass
                    
        else:
            f = target_files[0]
            start_time = time.time()
            match_count = 0
            
            buffer = []
            current_size = 0
            BUF_LIMIT = 64 * 1024
            
            try:
                # Single-file mode: ZoneMap writes ENABLED
                for chunk in search_file_generator(f, mode, keys, where_expr, json_mode, is_parallel=False):
                    match_count += 1
                    buffer.append(chunk)
                    current_size += len(chunk)
                    
                    if current_size >= BUF_LIMIT:
                        sys.stdout.buffer.write(b''.join(buffer))
                        sys.stdout.buffer.flush()
                        buffer.clear()
                        current_size = 0
                
                if buffer:
                    sys.stdout.buffer.write(b''.join(buffer))
                    sys.stdout.buffer.flush()
                    
                dur = time.time() - start_time
                sys.stderr.write(f"[{os.path.basename(f)}] Found {match_count} matches in {dur:.4f}s\n")
                sys.stderr.flush()
                
            except BrokenPipeError:
                pass
            except Exception as e:
                sys.stderr.write(f"Error: {e}\n")
                sys.exit(1)

    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()