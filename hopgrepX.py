import os
import sys
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Union

############################################################
# === 1. FAST PARSING & UTILS ===
############################################################

class FastUtils:
    """Methods optimized for raw speed using BYTES processing."""
    
    @staticmethod
    def parse_timestamp(line_prefix: bytes) -> Optional[float]:
        """Parse timestamp from bytes, returns float or None."""
        if len(line_prefix) < 19: 
            return None
        try:
            # Direct slicing on bytes - fastest method
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
        """Get first column (before space, pipe, or tab)."""
        if not line: 
            return b""
        
        # Fast loop - find first delimiter
        for i, char in enumerate(line):
            if char in b' |\t':
                return line[:i]
        return line

    @staticmethod
    def get_key_from_line(line: bytes, key_type: str) -> Union[float, bytes, None]:
        """Extract and parse key from first column."""
        if not line: 
            return None
        
        if key_type == 'timestamp':
            # Handle timestamp edge cases properly
            if len(line) >= 19:
                # Check if position 19 has a delimiter
                if len(line) > 19 and line[19] not in b' |\t\n\r':
                    # Not clean timestamp, extract first column
                    first_col = FastUtils.get_first_column_bytes(line)
                    if len(first_col) >= 19:
                        return FastUtils.parse_timestamp(first_col[:19])
                    return FastUtils.parse_timestamp(first_col)
                # Clean 19-char timestamp
                return FastUtils.parse_timestamp(line[:19])
            return None
            
        elif key_type == 'number':
            first_col = FastUtils.get_first_column_bytes(line)
            try:
                return float(first_col)
            except:
                return None
        
        # String type
        return FastUtils.get_first_column_bytes(line)

    @staticmethod
    def dt_to_float(s: str) -> float:
        """Parse various timestamp formats to float."""
        # Try full timestamp first
        try: 
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()
        except: 
            pass
        
        # Try other common formats
        fmt_map = {
            4: "%Y",                    # YYYY
            7: "%Y-%m",                 # YYYY-MM
            10: "%Y-%m-%d",            # YYYY-MM-DD
            13: "%Y-%m-%d %H",         # YYYY-MM-DD HH  # FIXED: Added missing %d
            16: "%Y-%m-%d %H:%M",      # YYYY-MM-DD HH:MM
        }
        
        if len(s) in fmt_map:
            try:
                return datetime.strptime(s, fmt_map[len(s)]).timestamp()
            except:
                pass
        
        # Last resort: try as float
        try: 
            return float(s)
        except: 
            raise ValueError(f"Cannot parse timestamp: {s}")

############################################################
# === 2. FILTER LOGIC (SQL-like) ===
############################################################

class FilterLogic:
    """SQL-like filter expression evaluator."""
    
    # Token types
    TOK_ID, TOK_OP, TOK_VAL = 'ID', 'OP', 'VAL'
    TOK_LPAREN, TOK_RPAREN = 'LPAREN', 'RPAREN'
    TOK_AND, TOK_OR, TOK_NOT = 'AND', 'OR', 'NOT'
    TOK_EOF = 'EOF'

    def __init__(self, expression: str):
        self.tokens = self.tokenize(expression)
        self.pos = 0
        self.ast = self.parse_expression()

    def tokenize(self, text: str):
        """Tokenize filter expression."""
        tokens = []
        i = 0
        n = len(text)
        ops = {'>=', '<=', '!=', '=', '>', '<', '=='}
        
        while i < n:
            c = text[i]
            
            # Skip whitespace
            if c.isspace(): 
                i += 1
                continue
            
            # Parentheses
            if c == '(':
                tokens.append((self.TOK_LPAREN, '('))
                i += 1
                continue
            if c == ')':
                tokens.append((self.TOK_RPAREN, ')'))
                i += 1
                continue
            
            # Operators (2-char first)
            if i + 2 <= n and text[i:i+2] in ops:
                tokens.append((self.TOK_OP, text[i:i+2]))
                i += 2
                continue
            if text[i:i+1] in ops:
                tokens.append((self.TOK_OP, text[i:i+1]))
                i += 1
                continue
            
            # Quoted strings
            if c in '"\'':
                q = c
                val = ""
                i += 1
                while i < n and text[i] != q:
                    val += text[i]
                    i += 1
                i += 1
                tokens.append((self.TOK_VAL, val))
                continue
            
            # Identifiers and values
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
            
            i += 1
        
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
        k = self.consume()
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
            
            # Try numeric comparison first
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
                # String comparison
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
    
    # Handle both pipe and space delimiters
    if '|' in line:
        parts = line.split('|')
    else:
        parts = line.split()
    
    # Positional columns (0-based)
    for idx, p in enumerate(parts):
        d[f"col:{idx}"] = p.strip()
    
    # Key-value pairs from all tokens
    all_tokens = ' '.join(parts).split()
    for token in all_tokens:
        if '=' in token:
            try:
                k, v = token.split('=', 1)
                d[k.strip()] = v.strip()
            except:
                pass
    
    return d


############################################################
# === 3. ZONE MAP Metadata Manager ===
############################################################

class ZoneMapLite:
    """
    Manages sidecar .hop.idx file for scanned regions.
    Maps byte_ranges -> key_ranges for faster searches.
    """
    
    # Configuration
    MAX_KEY_GAP_TS = 600.0      # 10 minutes
    MAX_KEY_GAP_NUM = 5000.0    # Numeric gap
    PADDING = 4 * 1024 * 1024   # 4MB safety padding
    
    # Tolerance for out-of-order logs
    TS_TOLERANCE = 120.0        # 2 minutes
    NUM_TOLERANCE = 0.001       # Numeric tolerance
    
    # Compaction targets
    TARGET_ZONE_SIZE = 64 * 1024 * 1024  # 64 MB
    MAX_MERGE_DIST = 16 * 1024 * 1024    # Merge if gap < 16MB
    MAX_ZONES = 5000                     # Hard cap

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.idx_path = log_path + ".hop.idx"
        self.zones = []
        self.loaded = False
        self.file_stats = self._get_file_stats()

    def _get_file_stats(self):
        """Get file statistics for validation."""
        try:
            st = os.stat(self.log_path)
            
            # Compute content hash for robustness
            head_hash = ""
            try:
                with open(self.log_path, "rb") as f:
                    chunk = f.read(4096)
                    head_hash = hashlib.md5(chunk).hexdigest()
            except:
                pass
            
            return {
                "size": st.st_size,
                "mtime": st.st_mtime,
                "inode": st.st_ino,
                "head_hash": head_hash
            }
        except:
            return None

    def load(self):
        """Load and validate index file."""
        if not os.path.exists(self.idx_path):
            return
            
        try:
            with open(self.idx_path, 'r') as f:
                data = json.load(f)
            
            sig = data.get("file_signature", {})
            if not self.file_stats:
                return
            
            # Basic validation
            if (sig.get("size") != self.file_stats["size"] or
                abs(sig.get("mtime", 0) - self.file_stats["mtime"]) > 0.1):
                return

            # Hardened validation
            if ("inode" in sig and sig["inode"] != self.file_stats["inode"]):
                return
            if ("head_hash" in sig and sig["head_hash"] != self.file_stats["head_hash"]):
                return
            
            self.zones = data.get("zones", [])
            self.zones.sort(key=lambda x: x["start"])
            self.loaded = True
        except:
            pass  # Corrupted index, ignore

    def get_search_hints(self, target_key, key_type: str) -> Tuple[int, Optional[int]]:
        """Get search hints with safety padding."""
        if not self.loaded or not self.zones or target_key is None:
            return 0, None

        def to_runtime(val):
            """Convert JSON value back to runtime type."""
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

        # 1. Direct hit search
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
            # 2. Neighbor search
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

        # Apply safety padding
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
        # Skip invalid observations
        if min_k is None or max_k is None:
            return
            
        # Normalize keys for JSON
        if isinstance(min_k, bytes):
            min_k = min_k.decode('iso-8859-1')
        if isinstance(max_k, bytes):
            max_k = max_k.decode('iso-8859-1')

        # Apply tolerances
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
            "max": max_k
        })

    def _merge_two_zones(self, z1: dict, z2: dict):
        """Merge z2 into z1."""
        z1["end"] = max(z1["end"], z2["end"])
        try:
            if z2["min"] < z1["min"]:
                z1["min"] = z2["min"]
            if z2["max"] > z1["max"]:
                z1["max"] = z2["max"]
        except:
            pass

    def _should_merge(self, z1: dict, z2: dict, force: bool = False) -> bool:
        """Decide if zones should be merged."""
        byte_gap = z2["start"] - z1["end"]
        
        # Strictly overlapping or touching
        if byte_gap <= 0:
            return True
        
        # Force merge for compaction
        if force:
            curr_size = z1["end"] - z1["start"]
            return curr_size < self.TARGET_ZONE_SIZE

        # Smart merge
        curr_size = z1["end"] - z1["start"]
        if byte_gap < self.MAX_MERGE_DIST and curr_size < self.TARGET_ZONE_SIZE:
            # Check key continuity
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
        """Compact zones list."""
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
        """Merge zones and write to disk atomically."""
        if not self.zones:
            return
        
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
        
        # Atomic write
        data = {
            "file_signature": self.file_stats,
            "zones": self.zones
        }
        
        try:
            tmp_path = self.idx_path + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(data, f)
            os.replace(tmp_path, self.idx_path)
        except:
            pass


############################################################
# === 4. REGION LOCATOR (HYPER-HOP) ===
############################################################

class HyperHopSeeker:
    """K-ary search for finding file bounds efficiently."""
    
    def __init__(self, filepath: str, key_type: str = 'timestamp'):
        self.f = open(filepath, "rb")
        self.f.seek(0, 2)
        self.filesize = self.f.tell()
        self.key_type = key_type
        
        self.K_FACTOR = 32
        self.SCAN_LIMIT = 256 * 1024  # 256KB

    def close(self):
        self.f.close()

    def raw_read_line_at(self, offset: int) -> Tuple[Optional[Union[float, bytes]], int]:
        """Read line at offset and parse its key."""
        if offset >= self.filesize:
            return None, offset
            
        self.f.seek(offset)
        if offset > 0:
            self.f.readline()  # Align to line boundary
            
        aligned_pos = self.f.tell()
        line = self.f.readline()
        
        if not line:
            return None, aligned_pos
            
        # Parse key from line
        k = FastUtils.get_key_from_line(line.strip(), self.key_type)
        return k, aligned_pos

    def locate_lower_bound(self, target, hint_start: int = 0, hint_end: Optional[int] = None) -> int:
        """Find lower bound for target key using k-ary search."""
        if target is None or target == float('-inf') or target == '':
            return 0
        
        low = hint_start
        high = hint_end if hint_end is not None else self.filesize
        
        # Clamp values
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

            # Find first sample >= target
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


############################################################
# === 5. DATA TYPES & HELPER FUNCTIONS ===
############################################################

class QueryRule:
    """Represents a search query rule."""
    
    def __init__(self, start_key, end_key):
        self.start_key = start_key
        self.end_key = end_key


def merge_byte_regions(regions: List[Tuple[int, int, List[QueryRule]]], 
                       safety_window: int) -> List[Tuple[int, int, List[QueryRule]]]:
    """Merge overlapping byte regions."""
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
    """Get next epsilon value for exclusive upper bound."""
    if key_type == 'number':
        return val + 0.000001
    elif key_type == 'timestamp':
        return val + 0.000001
    else:
        if isinstance(val, str):
            val = val.encode('utf-8')
        return val + b'\0'


def detect_key_type_from_file(filepath: str) -> str:
    """Detect key type by sampling file content."""
    sample_lines = []
    
    try:
        with open(filepath, 'rb') as f:
            for _ in range(100):  # Sample first 100 lines
                line = f.readline()
                if not line:
                    break
                sample_lines.append(line)
    except:
        pass
    
    if not sample_lines:
        return "string"
    
    # NEW: Use first non-empty line for detection, not majority
    for line in sample_lines:
        line = line.strip()
        if not line:
            continue
            
        # Try timestamp first (most specific)
        if len(line) >= 19:
            try:
                # Check timestamp pattern
                if line[4] == 45 and line[7] == 45 and line[10] == 32:  # '-', '-', ' '
                    FastUtils.parse_timestamp(line[:19])  # Validate
                    return "timestamp"
            except:
                pass
        
        # Extract first column
        first_col = FastUtils.get_first_column_bytes(line)
        
        # Try number
        try:
            float(first_col)
            return "number"
        except:
            pass
    
    # Default to string if no specific type detected
    return "string"


############################################################
# === 6. MAIN SEARCH ENGINE ===
############################################################

def run_search(filepath: str, mode: str, keys: List[str], filter_expr: Optional[str], json_mode: bool = False):
    """Main search function."""
    
    # 1. Detect key type from file
    key_type = detect_key_type_from_file(filepath)
    
    # Fallback: check first key if file detection fails
    if keys and key_type == "string":
        k0 = str(keys[0]).strip()
        if len(k0) >= 10 and k0[4] == '-' and k0[7] == '-':
            key_type = "timestamp"

    # 2. Initialize components
    seeker = HyperHopSeeker(filepath, key_type)
    file_size = seeker.filesize
    zonemap = ZoneMapLite(filepath)
    zonemap.load()
    
    # 3. Convert request to query rules
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
        
        if key_type == "timestamp":
            # Generate ranges for timestamp prefixes
            try:
                prefix = s_str
                if len(prefix) == 4:  # YYYY
                    y = int(prefix)
                    start = datetime(y, 1, 1).timestamp()
                    end = datetime(y + 1, 1, 1).timestamp()
                elif len(prefix) == 7:  # YYYY-MM
                    y, m = map(int, prefix.split('-'))
                    start = datetime(y, m, 1).timestamp()
                    end = datetime(y, m + 1, 1).timestamp() if m < 12 else datetime(y + 1, 1, 1).timestamp()
                elif len(prefix) == 10:  # YYYY-MM-DD
                    y, m, d = map(int, prefix.split('-'))
                    start = datetime(y, m, d).timestamp()
                    end = start + 86400  # +1 day
                elif len(prefix) == 13:  # YYYY-MM-DD HH
                    y, m, d = map(int, prefix[:10].split('-'))
                    h = int(prefix[11:13])
                    start = datetime(y, m, d, h).timestamp()
                    end = start + 3600  # +1 hour
                elif len(prefix) == 16:  # YYYY-MM-DD HH:MM
                    y, m, d = map(int, prefix[:10].split('-'))
                    h = int(prefix[11:13])
                    mm = int(prefix[14:16])
                    start = datetime(y, m, d, h, mm).timestamp()
                    end = start + 60  # +1 minute
                else:
                    # Fallback to open-ended search
                    start = FastUtils.dt_to_float(prefix)
                    end = None
                
                queries.append(QueryRule(start, end))
            except:
                queries.append(QueryRule(0, None))
                
        elif key_type == "number":
            # FIXED: No numeric range expansion - use open-ended query
            # Prefix matching will be done via string comparison
            queries.append(QueryRule(0, None))
                
        else:  # string
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

    # 4. Locate regions using ZoneMap hints
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

    # 5. Merge regions
    SAFETY_WINDOW = 1024 * 1024  # 1MB
    merged_regions = merge_byte_regions(raw_regions, SAFETY_WINDOW)
    
    # 6. Execute scan
    filter_eng = FilterLogic(filter_expr) if filter_expr else None
    total_matches = 0
    start_time = time.time()
    
    # Output buffer for batch writes
    output_buffer = []
    BUFFER_SIZE = 1000
    
    try:
        if mode == "--substring" and not keys:
            # Full file scan for substring
            seeker.f.seek(0)
            while True:
                curr_offset = seeker.f.tell()
                line_b = seeker.f.readline()
                if not line_b:
                    break
                    
                if filter_eng or json_mode:
                    line_str = line_b.decode(errors='ignore').strip()
                    ctx = extract_context(line_str)
                    
                    if filter_eng and not filter_eng.evaluate(ctx):
                        continue
                        
                    if json_mode:
                        # Structured JSON output
                        fields = {k: v for k, v in ctx.items() if not k.startswith("col:")}
                        json_obj = {
                            "file": filepath,
                            "offset": curr_offset,
                            "key": ctx.get("col:0", ""),
                            "fields": fields,
                            "_raw": line_str
                        }
                        output_buffer.append((json.dumps(json_obj) + "\n").encode('utf-8'))
                    else:
                        output_buffer.append(line_b)
                else:
                    output_buffer.append(line_b)
                
                total_matches += 1
                
                if len(output_buffer) >= BUFFER_SIZE:
                    sys.stdout.buffer.write(b''.join(output_buffer))
                    output_buffer.clear()
        
        else:
            # Region-based scanning
            for (safe_start, safe_end_anchor, rules) in merged_regions:
                seeker.f.seek(safe_start)
                if safe_start > 0:
                    seeker.f.readline()  # Align
                
                # Determine termination conditions
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
                
                # Pre-compute prefix values
                q_sub_list = [k.encode('utf-8') for k in keys] if (mode == "--substring" and keys) else []
                prefix_str = str(keys[0]) if (mode == "--prefix" and keys) else ""
                prefix_bytes = prefix_str.encode('utf-8') if prefix_str else b""
                
                # Track for ZoneMap
                region_min_k = None
                region_max_k = None
                region_start_pos = seeker.f.tell()
                
                # Progress tracking
                last_progress = 0
                lines_scanned = 0
                max_lines_without_match = 10000  # Safety limit
                
                while True:
                    curr_pos = seeker.f.tell()
                    if curr_pos >= file_size:
                        break
                    
                    # Progress reporting for large files
                    if file_size > 100_000_000:
                        if curr_pos - last_progress > file_size // 100:
                            sys.stderr.write(f"\rProgress: {curr_pos/file_size:.1%}")
                            last_progress = curr_pos
                    
                    line_bytes = seeker.f.readline()
                    if not line_bytes:
                        break
                    
                    lines_scanned += 1
                    line_stripped = line_bytes.strip()
                    
                    # Get key for ZoneMap and eq/range/multi queries
                    k = FastUtils.get_key_from_line(line_stripped, key_type)
                    
                    # Update ZoneMap stats (if applicable)
                    if k is not None:
                        if region_min_k is None:
                            region_min_k = region_max_k = k
                        else:
                            if k < region_min_k:
                                region_min_k = k
                            if k > region_max_k:
                                region_max_k = k
                    
                    # --- TERMINATION CHECKS ---
                    terminate = False
                    
                    if mode not in ["--substring", "--prefix"]:
                        # Eq/Range/Multi: stop when passed max key
                        if not has_open_end and k is not None and max_logical_key_rule is not None:
                            if k > max_logical_key_rule:
                                terminate = True
                        
                        # Safety fallback
                        if (not terminate and not has_open_end and 
                            curr_pos > safety_cutoff and k is not None):
                            if k >= max_logical_key_rule:
                                terminate = True
                    
                    # For prefix searches, ONLY terminate on:
                    # 1. EOF (curr_pos >= file_size)
                    # 2. Safety window exhaustion (curr_pos >= safety_cutoff)
                    # NO NUMERIC/TIMESTAMP EARLY TERMINATION LOGIC
                    
                    elif mode == "--prefix":
                        # Safety: too many lines without matches (warning only)
                        if lines_scanned > max_lines_without_match and total_matches == 0:
                            sys.stderr.write(f"\nWarning: Scanned {max_lines_without_match} lines without matches\n")
                            # Don't terminate, just log warning
                    
                    # Always terminate on EOF or safety cutoff
                    if curr_pos >= safety_cutoff:
                        terminate = True
                    
                    if terminate:
                        break
                    
                    # --- MATCH CHECKING ---
                    is_match = False
                    
                    if mode == "--substring":
                        for qs in q_sub_list:
                            if qs in line_bytes:
                                is_match = True
                                break
                            
                    elif mode == "--prefix":
                        if key_type == "string":
                            # Check first column only
                            if isinstance(k, bytes):
                                is_match = k.startswith(prefix_bytes)
                                
                        elif key_type == "timestamp":
                            # Check raw line start
                            is_match = line_stripped.startswith(prefix_bytes)
                            
                        elif key_type == "number":
                            # FIXED: Simple string prefix matching on first column
                            first_col_bytes = FastUtils.get_first_column_bytes(line_stripped)
                            try:
                                first_col_str = first_col_bytes.decode('utf-8', errors='ignore')
                                is_match = first_col_str.startswith(prefix_str)
                            except:
                                pass
                    
                    else:  # Eq/Range/Multi
                        if k is None:
                            pass
                        else:
                            for r in rules:
                                if ((r.start_key is None or k >= r.start_key) and
                                    (r.end_key is None or k < r.end_key)):
                                    is_match = True
                                    break
                    
                    # --- OUTPUT HANDLING ---
                    if is_match:
                        if filter_eng or json_mode:
                            line_str = line_bytes.decode(errors='ignore').strip()
                            ctx = extract_context(line_str)
                            
                            if filter_eng and not filter_eng.evaluate(ctx):
                                continue
                                
                            if json_mode:
                                # Structured JSON output
                                fields = {k: v for k, v in ctx.items() if not k.startswith("col:")}
                                json_obj = {
                                    "file": filepath,
                                    "offset": curr_pos,
                                    "key": ctx.get("col:0", ""),
                                    "fields": fields,
                                    "_raw": line_str
                                }
                                output_buffer.append((json.dumps(json_obj) + "\n").encode('utf-8'))
                            else:
                                output_buffer.append(line_bytes)
                        else:
                            output_buffer.append(line_bytes)
                        
                        total_matches += 1
                        
                        # Flush buffer if full
                        if len(output_buffer) >= BUFFER_SIZE:
                            sys.stdout.buffer.write(b''.join(output_buffer))
                            output_buffer.clear()
                
                # End of region - update ZoneMap (only for bounded queries)
                region_end_pos = seeker.f.tell()
                if region_min_k is not None and region_max_k is not None:
                    # FIXED: Don't update ZoneMap for prefix or substring searches
                    if mode not in ["--prefix", "--substring"]:
                        zonemap.add_observation(region_start_pos, region_end_pos, 
                                              region_min_k, region_max_k, key_type)
        
        # Flush any remaining output
        if output_buffer:
            sys.stdout.buffer.write(b''.join(output_buffer))
            
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        pass
    
    # 7. Save metadata and cleanup
    zonemap.commit()
    seeker.close()
    
    # Final statistics
    dur = time.time() - start_time
    sys.stderr.write(f"\nFound {total_matches} matches in {dur:.4f}s\n")


############################################################
# === 7. CLI INTERFACE ===
############################################################

def print_help():
    """Print help information."""
    print("""
Fast Log Search (hopgrepX) - Search sorted log files efficiently
    
Usage:
  hopgrepX.py <file> <pattern>                    # Substring search
  hopgrepX.py <file> --eq <key>                   # Exact match
  hopgrepX.py <file> --range <start> <end>        # Range search
  hopgrepX.py <file> --prefix <prefix>            # Prefix search
  hopgrepX.py <file> --multi <k1,k2,k3>           # Multiple exact matches
    
Filters:
  --where "field=value"                           # Filter results
  --where "col:0=ERROR"                           # Filter by column (0-based)
  --where "severity=ERROR AND user=admin"         # Multiple conditions
  --json                                          # Output matches as JSON objects
    
Examples:
  hopgrepX.py logs.txt ERROR                      # Find "ERROR" anywhere
  hopgrepX.py logs.txt --eq "2024-01-15 10:30:00" # Exact timestamp
  hopgrepX.py logs.txt --range "2024-01-15" "2024-01-16"
  hopgrepX.py logs.txt --prefix "2024-01"         # January 2024 logs
  hopgrepX.py logs.txt --where "severity=ERROR"   # Filtered search
    
Notes:
  - First column must be primary key (timestamp/number/string)
  - Log file must be sorted by primary key
  - Auto-detects key type from file content
  - Creates .hop.idx index file for faster repeated searches
""")


def main():
    """Main CLI entry point."""
    
    # Show help if requested
    if len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']:
        print_help()
        sys.exit(0)
    
    # Check minimum arguments
    if len(sys.argv) < 3:
        print("Error: Insufficient arguments")
        print_help()
        sys.exit(1)
        
    # Pre-parse flags
    json_mode = False
    if "--json" in sys.argv:
        json_mode = True
        sys.argv.remove("--json")
        
    # Re-check len after flag removal
    if len(sys.argv) < 3:
        print("Error: Insufficient arguments")
        print_help()
        sys.exit(1)
    
    filepath = sys.argv[1]
    second_arg = sys.argv[2]
    
    # Parse arguments based on whether second arg is a mode flag
    if second_arg.startswith("--"):
        # Explicit mode
        mode = second_arg
        if mode not in ['--eq', '--range', '--prefix', '--multi', '--substring']:
            print(f"Error: Unknown mode '{mode}'")
            print_help()
            sys.exit(1)
        
        # Parse remaining arguments
        remaining_args = sys.argv[3:]
        keys = []
        where_expr = None
        
        i = 0
        while i < len(remaining_args):
            arg = remaining_args[i]
            if arg == "--where":
                if i + 1 < len(remaining_args):
                    if where_expr:
                        where_expr += " AND " + remaining_args[i + 1]
                    else:
                        where_expr = remaining_args[i + 1]
                    i += 2
                    continue
            keys.append(arg)
            i += 1
    else:
        # Implicit substring mode
        mode = "--substring"
        remaining_args = sys.argv[2:]
        keys = []
        where_expr = None
        
        i = 0
        while i < len(remaining_args):
            arg = remaining_args[i]
            if arg == "--where":
                if i + 1 < len(remaining_args):
                    if where_expr:
                        where_expr += " AND " + remaining_args[i + 1]
                    else:
                        where_expr = remaining_args[i + 1]
                    i += 2
                    continue
            keys.append(arg)
            i += 1
    
    # Validate arguments
    if mode == "--range" and len(keys) != 2:
        print("Error: --range requires exactly two arguments")
        sys.exit(1)
    
    if mode == "--multi" and len(keys) == 1:
        # Split comma-separated list
        keys = keys[0].split(',')
        keys = [k.strip() for k in keys if k.strip()]
        
    if mode == "--multi" and not keys:
        print("Error: --multi requires comma-separated list")
        sys.exit(1)
    
    # Run the search
    try:
        run_search(filepath, mode, keys, where_expr, json_mode)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()