from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

from .config import BIGWIG_DIR, EPIGENETIC_EXTENSIONS, EPIG_AGGREGATION, EXPECTED_BIGWIGS
from .epi_seq import single_interval_features

_COORD_RE = re.compile(r'(chr[\w]+|\b[0-9XYM]+)\s*:\s*([0-9,]+)\s*-\s*([0-9,]+)', re.IGNORECASE)

def _parse_id_region(id_str: str) -> Optional[Tuple[str, int, int]]:
    m = _COORD_RE.search(id_str)
    if not m:
        return None
    chrom = m.group(1)
    if not chrom.lower().startswith("chr"):
        chrom = "chr" + chrom
    start = int(m.group(2).replace(",", ""))
    end   = int(m.group(3).replace(",", ""))
    if start > 0:
        start -= 1  # convert to 0-based, half-open
    return chrom.lower(), start, end

def _collect_bigwigs() -> List[Tuple[str, Optional[Path]]]:
    existing = {}
    if BIGWIG_DIR.exists():
        for p in BIGWIG_DIR.iterdir():
            if p.is_file() and p.suffix.lower() in {".bw", ".bigwig"}:
                existing[p.stem] = p
    ordered = EXPECTED_BIGWIGS[:] if EXPECTED_BIGWIGS else sorted(existing.keys())
    return [(base, existing.get(base)) for base in ordered]

def _predeclare_zero_feats(basenames: List[str]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for base in basenames:
        for ext in EPIGENETIC_EXTENSIONS:
            feats[f"{base}_{int(ext)}"] = 0.0
    return feats

def epigenetic_features(row, logger=None) -> Dict[str, float]:
    items = _collect_bigwigs()
    basenames = [b for b, _ in items]
    feats = _predeclare_zero_feats(basenames)

    parsed = _parse_id_region(str(row.get("ID", "")))
    if not parsed:
        return feats

    chrom, abs_start_input, _abs_end_input = parsed
    try:
        off_start = int(row.get("Start")); off_end = int(row.get("End"))
    except Exception:
        return feats

    abs_start = abs_start_input + off_start
    abs_end   = abs_start_input + off_end

    present_paths = [p for (_b, p) in items if p is not None]
    if present_paths:
        try:
            vals = single_interval_features(
                bw_paths=present_paths,
                chrom=chrom, start0=abs_start, end0=abs_end,
                extensions=EPIGENETIC_EXTENSIONS, agg=EPIG_AGGREGATION
            )
            for base, path in items:
                for ext in EPIGENETIC_EXTENSIONS:
                    key = f"{base}_{int(ext)}"
                    if path is not None:
                        feats[key] = float(vals.get(key, 0.0))
        except Exception as e:
            if logger: logger.warning(f"EPI fail for {row.get('ID')}: {e}")
    return feats