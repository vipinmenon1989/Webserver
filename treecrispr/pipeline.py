from __future__ import annotations
from typing import List, Tuple, Dict
import pandas as pd

from .config import MAX_SEQ_LEN
from .features_seq import seq_features_for, reverse_complement
from .features_epi import epigenetic_features
from .models import load_models, score_with_models

# ---- PAM scanner ----
def _scan_30mers(seq: str) -> List[Tuple[int,int,str,str]]:
    """
    Return list of (start, end, strand, pam_label) for 30-mers that satisfy:
      + strand: window[25:27] == 'GG'  (NGG at positions 25..26)
      - strand: window[3:5]  == 'CC'  (CCN at positions 3..4), report as '-' and use RC for features
    """
    s = seq.upper().replace("U","T")
    out = []
    n = len(s)
    for i in range(0, max(0, n-30)+1):
        w = s[i:i+30]
        if len(w) < 30: break
        if w[25:27] == "GG":
            out.append((i, i+30, "+", "NGG"))
        if w[3:5] == "CC":
            out.append((i, i+30, "-", "CCN"))
    return out

def build_candidates(fasta_id: str, seq: str) -> pd.DataFrame:
    """Return candidates DataFrame with core columns."""
    seq = seq.upper().replace("U","T")
    rows = []
    for start, end, strand, pam in _scan_30mers(seq):
        win = seq[start:end]
        rc  = win if strand == "+" else reverse_complement(win)
        rows.append({
            "ID": fasta_id,
            "Start": start,
            "End": end,
            "Strand": "+" if strand == "+" else "-",
            "Sequence": win,                 # 30nt as found in input
            "ReverseComplement": rc,         # if -, RC; if +, same as Sequence
            "PAM": pam
        })
    return pd.DataFrame(rows)

# REPLACE your old build_features() with this
def compute_features_only(df_base: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    Returns ONLY numeric features (seq + epi) for scoring.
    Does NOT include metadata columns like Start/End/ID.
    """
    if df_base.empty:
        return pd.DataFrame(index=df_base.index)

    feat_rows: List[Dict[str, float]] = []
    for _, row in df_base.iterrows():
        # Feature sequence: original for +, RC for -
        feat_seq = row["Sequence"] if row["Strand"] == "+" else row["ReverseComplement"]
        fseq = seq_features_for(feat_seq, "+" if row["Strand"] == "+" else "-")
        fepi = epigenetic_features(row, logger=logger)
        fseq.update(fepi)
        feat_rows.append(fseq)

    F = pd.DataFrame(feat_rows).fillna(0.0)
    return F

def run_full_pipeline(records: List[Tuple[str,str]], logger=None, model_dir=None) -> pd.DataFrame:
    """
    records: list of (id, seq)
    model_dir: Path to the chosen mode's model directory (i or a)
    """
    # build all candidates
    all_rows = []
    for rid, seq in records:
        if len(seq) > MAX_SEQ_LEN:
            continue
        cands = build_candidates(rid, seq)
        all_rows.append(cands)
    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=[
        "ID","Start","End","Strand","Sequence","ReverseComplement","PAM"
    ])
    # features
   # after
# base candidates (metadata)
    base_df = df  # same DataFrame from build_candidates, untouched

# features ONLY (numeric)
    F = compute_features_only(base_df, logger=logger)

# quick sanity log (should be ~688)
    if logger:
        n_num = F.select_dtypes(include="number").shape[1]
        logger.info(f"[features_only] numeric_cols={n_num} (expect ~688)")

    # scoring uses ONLY features
    models = load_models(model_dir, logger=logger) if model_dir else {}
    scores = score_with_models(F, models, model_dir=model_dir, logger=logger) if models else pd.DataFrame(index=F.index)

    # final output for UI/downloads: base metadata + scores (no raw features)
    out = pd.concat([base_df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    return out 