"""Check the report's key numbers against the saved result files, and optionally compare a fresh run.

Usage (from the repository root, CPU only, needs numpy, pandas and scipy):

    python scripts/verify_results.py
        Recomputes the headline numbers of the report from the saved CSV/JSON files and prints PASS/FAIL
        for each one.

    python scripts/verify_results.py --compare path/to/new/outputs/05b
        Compares the CSV files in a folder you downloaded from a fresh Kaggle run (for example
        /kaggle/working/outputs/05b) with the saved files of the same name, row by row, and reports the largest
        difference in every numeric column.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1 = os.path.join(ROOT, "stage1_degrade_restore_detect", "results")
S23 = os.path.join(ROOT, "stage2_3_restoring_for_the_machine", "results")

results = []


def check(name, value, expected, tol, where):
    ok = abs(value - expected) <= tol
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:<62} {value:>9.4f}  (report: {expected}, from {where})")


def check_true(name, ok, detail, where):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name:<62} {detail:>9}  (from {where})")


def stage1():
    print("\n== Stage 1: Degrade -> Restore -> Detect ==")
    d = pd.read_csv(f"{S1}/05b/sweep.csv")
    m = d.set_index(["degradation", "severity", "restorer"])["map"]
    clean = m[("clean", 0.0, "none")]
    check("clean mAP@[.5:.95], YOLOv8n, 300 COCO images", clean, 0.387, 0.0005, "05b/sweep.csv")

    def R(deg, sev, rest):
        return (m[(deg, sev, rest)] - m[(deg, sev, "none")]) / (clean - m[(deg, sev, "none")])

    check("DnCNN recovered share R, Gaussian sigma = 25", R("gaussian", 25.0, "dncnn"), 0.77, 0.005, "05b/sweep.csv")
    check("DnCNN recovered share R, Gaussian sigma = 50", R("gaussian", 50.0, "dncnn"), 0.72, 0.005, "05b/sweep.csv")
    check("BM3D mAP, Gaussian sigma = 50", m[("gaussian", 50.0, "bm3d")], 0.297, 0.0005, "05b/sweep.csv")
    check("low light 0.8: gamma -> DnCNN mAP", m[("low_light", 0.8, "gamma>dncnn")], 0.242, 0.0005, "05b/sweep.csv")
    check("low light 0.8: DnCNN -> gamma mAP", m[("low_light", 0.8, "dncnn>gamma")], 0.196, 0.0005, "05b/sweep.csv")
    r = d[d.degradation != "clean"]
    check("Spearman PSNR vs mAP over the 64 restored runs", spearmanr(r.psnr, r["map"]).correlation, 0.76, 0.005,
          "05b/sweep.csv")
    e = pd.read_csv(f"{S1}/04/dncnn_eval.csv").groupby("sigma")["psnr_dncnn"].mean()
    for s, v in [(15, 33.76), (25, 31.05), (50, 27.69)]:
        check(f"DnCNN PSNR (dB) on the 96 held-out BSD500 images, sigma = {s}", e[s], v, 0.01, "04/dncnn_eval.csv")


def stage2():
    print("\n== Stage 2: inside the detector (notebook 07) ==")
    sc = json.load(open(f"{S23}/07/scorecard_07.json"))
    check("clean mAP reproduced in stage 2", sc["clean_map"], 0.387, 0.0006, "07/scorecard_07.json")
    check("noise ceiling (Spearman between two noise seeds)", sc["noise_ceiling"], 0.74, 0.005, "07/scorecard_07.json")
    n = sum(sc["verdicts"].values())
    check_true("8 of the 13 stage 2 hypotheses supported", n == 8 and len(sc["verdicts"]) == 13, f"{n}/13",
               "07/scorecard_07.json")
    rt = pd.read_csv(f"{S23}/07/recovery_table.csv")
    g = rt[(rt.degradation == "gaussian") & (rt.restorer == "dncnn")].set_index("severity")
    check("DnCNN pixel-error removed, min over sigma 15/25/50", g.R_pixel.min(), 0.82, 0.006, "07/recovery_table.csv")
    check("DnCNN pixel-error removed, max over sigma 15/25/50", g.R_pixel.max(), 0.93, 0.006, "07/recovery_table.csv")
    check("DnCNN neck damage removed, min over sigma 15/25/50", g.R_neck.min(), 0.58, 0.006, "07/recovery_table.csv")
    check("DnCNN neck damage removed, max over sigma 15/25/50", g.R_neck.max(), 0.62, 0.006, "07/recovery_table.csv")
    pick = json.load(open(f"{S23}/07/layers_for_partB.json"))
    check_true("layer chosen for stage 3 = cosine at layer 21", pick["metric"] == "cos" and pick["layers"] == ["21"],
               f"{pick['metric']}@{','.join(pick['layers'])}", "07/layers_for_partB.json")


def stage3():
    print("\n== Stage 3: denoiser trained on detector features (notebook 08) ==")
    r = pd.read_csv(f"{S23}/08/runs.csv")
    t = r[(r.split == "test") & (r.detector == "yolov8n") & (r.sigma == 50)].set_index("model")["map"]
    check("YOLOv8n test mAP, sigma = 50, control (lambda = 0)", t["lam_0"], 0.302, 0.0005, "08/runs.csv")
    check("YOLOv8n test mAP, sigma = 50, lambda* = 1", t["lam_1"], 0.312, 0.0005, "08/runs.csv")
    check("YOLOv8n test mAP, sigma = 50, lambda = 0.3", t["lam_0.3"], 0.317, 0.0005, "08/runs.csv")
    ls = json.load(open(f"{S23}/08/lambda_star.json"))
    check_true("lambda* chosen on the validation images = 1", ls["lambda_star"] == "lam_1", ls["lambda_star"],
               "08/lambda_star.json")
    sc = json.load(open(f"{S23}/08/scorecard_08.json"))
    n = sum(sc["verdicts"].values())
    check_true("2 of the 4 stage 3 hypotheses supported", n == 2 and len(sc["verdicts"]) == 4, f"{n}/4",
               "08/scorecard_08.json")


def compare(new_dir, tol):
    """Compare every CSV in new_dir with the saved file of the same name (searched under both results folders)."""
    saved = {os.path.basename(p): p for p in glob.glob(f"{S1}/**/*.csv", recursive=True) +
             glob.glob(f"{S23}/**/*.csv", recursive=True)}
    stage = os.path.basename(os.path.normpath(new_dir)).replace("_quick", "")
    worst_all = 0.0
    for p in sorted(glob.glob(os.path.join(new_dir, "*.csv"))):
        name = os.path.basename(p)
        ref_path = next((s for s in saved.values() if s.endswith(os.path.join(stage, name))), saved.get(name))
        if ref_path is None:
            print(f"skip  {name}: no saved file with this name")
            continue
        new, ref = pd.read_csv(p), pd.read_csv(ref_path)
        keys = [c for c in ref.columns if not pd.api.types.is_numeric_dtype(ref[c]) or c in ("severity", "sigma", "image", "image_id")]
        keys = [k for k in keys if k in new.columns]
        num = [c for c in ref.columns if c in new.columns and c not in keys and pd.api.types.is_numeric_dtype(ref[c])]
        m = ref.merge(new, on=keys, suffixes=("_saved", "_new")) if keys else None
        if m is None or len(m) == 0:
            print(f"skip  {name}: could not match rows (key columns {keys})")
            continue
        print(f"\n{name}: {len(m)} of {len(ref)} saved rows matched on {keys}")
        for c in num:
            a, b = m[f"{c}_saved"].to_numpy(float), m[f"{c}_new"].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            diff = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0
            if c in ("seconds",):
                continue
            worst_all = max(worst_all, diff) if c in ("map", "map50", "ap") else worst_all
            print(f"  {'OK  ' if diff <= tol else 'DIFF'} {c:<14} max |new - saved| = {diff:.6f}")
    print(f"\nlargest mAP difference: {worst_all:.6f} (tolerance {tol})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", metavar="DIR", help="folder of CSVs from a fresh run to compare with the saved ones")
    ap.add_argument("--tol", type=float, default=0.002, help="tolerance for --compare (default 0.002)")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare, a.tol)
        sys.exit(0)
    stage1()
    stage2()
    stage3()
    print(f"\n{sum(results)} of {len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
