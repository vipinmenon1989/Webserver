from pathlib import Path
from flask import Flask, render_template, request, flash, redirect, url_for, send_file
from io import StringIO
import os, uuid
import pandas as pd
from Bio import SeqIO
import numpy as np
from flask import send_from_directory, abort
import pandas as pd
from treecrispr.plots import generate_boxplot_and_stats
from flask import abort, send_from_directory, send_file
from treecrispr.config import ALLOWED_EXT, MAX_SEQ_LEN, UPLOAD_DIR, RESULTS_DIR, MODEL_DIR_I, MODEL_DIR_A
from treecrispr.pipeline import run_full_pipeline

app = Flask(__name__)
app.secret_key = "supersecret"

def parse_fasta_text(text: str):
    recs=[]
    handle=StringIO(text)
    for rec in SeqIO.parse(handle,"fasta"):
        s=str(rec.seq).upper().replace("U","T")
        if len(s)>MAX_SEQ_LEN: raise ValueError(f"{rec.id} exceeds {MAX_SEQ_LEN} bp")
        recs.append((rec.id, s))
    return recs

def parse_fasta_file(path: str):
    recs=[]
    for rec in SeqIO.parse(path,"fasta"):
        s=str(rec.seq).upper().replace("U","T")
        if len(s)>MAX_SEQ_LEN: raise ValueError(f"{rec.id} exceeds {MAX_SEQ_LEN} bp")
        recs.append((rec.id, s))
    return recs

@app.route("/", methods=["GET","POST"])
def index():
    if request.method=="POST":
        # 1) must have FASTA (pasted or file)
        all_records=[]
        pasted=request.form.get("fasta_text","").strip()
        if pasted:
            try: all_records+=parse_fasta_text(pasted)
            except Exception as e: flash(str(e),"danger"); return redirect(url_for("index"))
        f=request.files.get("fasta_file")
        if f and f.filename:
            ext=os.path.splitext(f.filename)[1].lower()
            if ext not in ALLOWED_EXT:
                flash("Unsupported file type (allowed: .fa, .fasta, .fna)","danger"); return redirect(url_for("index"))
            p = UPLOAD_DIR / f.filename
            f.save(p)
            try: all_records+=parse_fasta_file(p)
            except Exception as e: flash(str(e),"danger"); return redirect(url_for("index"))
        if not all_records:
            flash("Please paste or upload at least one FASTA sequence.","warning")
            return redirect(url_for("index"))

        # 2) require mode selection
        mode = request.form.get("mode")  # "i" or "a"
        if mode not in ("i","a"):
            flash("Please choose TreeCRISPRi or TreeCRISPRa.","warning")
            return redirect(url_for("index"))
        model_dir = MODEL_DIR_I if mode=="i" else MODEL_DIR_A

        # 3) run pipeline
        df = run_full_pipeline(all_records, logger=app.logger, model_dir=model_dir)

        # 4) cache result
        token = str(uuid.uuid4())[:8]
        outcsv = RESULTS_DIR / f"results_{token}.csv"
        df.to_csv(outcsv, index=False)

        # columns for table
        base_cols = ["ID","Start","End","Strand","Sequence","ReverseComplement","PAM"]
        score_cols = [c for c in df.columns if c not in base_cols and not df[c].dtype=='O']
        columns = base_cols + score_cols
        rows = df[columns].to_dict(orient="records")
        return render_template("results.html", columns=columns, rows=rows, token=token)
    return render_template("index.html")
@app.route("/workflow")
def workflow():
    # Look for static/img/workflow.png (or .jpg)
    static_dir = Path(current_app.static_folder or "static")
    candidates = [
        static_dir / "img" / "workflow.png",
        static_dir / "img" / "workflow.jpg",
        static_dir / "img" / "workflow.jpeg",
        static_dir / "workflow.png",
        static_dir / "workflow.jpg",
        static_dir / "workflow.jpeg",
    ]
    for p in candidates:
        if p.exists():
            # pass the relative path so template can url_for('static', filename=...)
            rel = p.relative_to(static_dir).as_posix()
            return render_template("workflow.html", workflow_image=rel)
    # no image yet — render placeholder text
    return render_template("workflow.html", workflow_image=None)

@app.route("/download/<token>.<fmt>", endpoint="download_result")
def download(token, fmt):
    path = RESULTS_DIR / f"results_{token}.csv"
    if not path.exists():
        flash("Result not found.","danger"); return redirect(url_for("index"))
    df = pd.read_csv(path)
    if fmt=="csv":   return send_file(path, as_attachment=True)
    if fmt=="txt":   return send_file(path, as_attachment=True, download_name=f"results_{token}.txt")
    if fmt=="xlsx":
        xlsx = RESULTS_DIR / f"results_{token}.xlsx"
        df.to_excel(xlsx, index=False)
        return send_file(xlsx, as_attachment=True)
    if fmt=="tsv":
        tsv = RESULTS_DIR / f"results_{token}.tsv"
        df.to_csv(tsv, sep="\t", index=False)
        return send_file(tsv, as_attachment=True)
    flash("Unsupported format.","warning"); return redirect(url_for("index"))

# === Plot page: shows boxplot + ridgeline + rank heatmap + dominance heatmap + summary/table ===
@app.route("/boxplot/<token>")
def boxplot_view(token):
    csv_path = RESULTS_DIR / f"results_{token}.csv"
    if not csv_path.exists():
        abort(404)

    df = pd.read_csv(csv_path)
    out_dir = RESULTS_DIR / f"results_{token}_box"
    out = generate_boxplot_and_stats(df, out_dir)

    stats_df = out["stats"] if isinstance(out.get("stats"), pd.DataFrame) else pd.DataFrame()
    show_table, rows = False, []
    if not stats_df.empty:
        sdf = stats_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["pvalue"])
        sig = sdf[sdf["pvalue"] <= 0.05].copy().sort_values(
            ["Editor1", "Editor2", "pvalue"], kind="mergesort"
        )
        if not sig.empty:
            rows = sig[["Editor1","Editor2","pvalue","Better","cliffs_delta","cliffs_mag","hl_shift"]].to_dict(orient="records")
            show_table = True

    return render_template(
        "boxplot.html",
        token=token,
        have_box=bool(out.get("box_png")),
        have_ridge=bool(out.get("ridge_png")),
        have_tau=bool(out.get("tau_png")),
        have_dom=bool(out.get("dom_png")),
        box_name=os.path.basename(out.get("box_png") or ""),
        ridge_name=os.path.basename(out.get("ridge_png") or ""),
        tau_name=os.path.basename(out.get("tau_png") or ""),
        dom_name=os.path.basename(out.get("dom_png") or ""),
        show_table=show_table,
        table_rows=rows,
        ranking_text=out.get("ranking_text", "")
    )

# Serve plot images
@app.route("/boxplot_img/<token>/<kind>")
def boxplot_image(token, kind):
    img_dir = RESULTS_DIR / f"results_{token}_box"
    fname_map = {
        "box":  "scores_boxplot.png",
        "ridge":"scores_ridgeline.png",
        "tau":  "scores_rank_heatmap.png",
        "dom":  "scores_dominance_heatmap.png",
    }
    fname = fname_map.get(kind)
    if not fname:
        abort(404)
    return send_from_directory(str(img_dir), fname, as_attachment=False)

# Downloads: images and stats CSV
@app.route("/boxplot_download/<token>/<kind>")
def boxplot_download(token, kind):
    out_dir = RESULTS_DIR / f"results_{token}_box"
    fname_map = {
        "box":  "scores_boxplot.png",
        "ridge":"scores_ridgeline.png",
        "tau":  "scores_rank_heatmap.png",
        "dom":  "scores_dominance_heatmap.png",
        "csv":  "pairwise_stats.csv",
    }
    fname = fname_map.get(kind)
    if not fname:
        abort(404)
    path = out_dir / fname
    if not path.exists():
        abort(404)
    return send_file(str(path), as_attachment=True)

# optional diags
@app.route("/diag/epi")
def diag_epi():
    from treecrispr.features_epi import _collect_bigwigs
    from treecrispr.config import EPIGENETIC_EXTENSIONS, BIGWIG_DIR
    items = _collect_bigwigs()
    have = [b for b,p in items if p is not None]
    missing = [b for b,p in items if p is None]
    return "<pre>" + "\n".join([
        f"BIGWIG_DIR: {BIGWIG_DIR} (exists={BIGWIG_DIR.exists()})",
        f"Extensions: {EPIGENETIC_EXTENSIONS}",
        f"Found   ({len(have)}): {', '.join(have)}",
        f"Missing ({len(missing)}): {', '.join(missing) or '-'}",
    ]) + "</pre>"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8386, debug=True)

