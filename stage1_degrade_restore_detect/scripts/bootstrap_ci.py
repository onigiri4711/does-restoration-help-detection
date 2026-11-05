# Paired bootstrap CIs from notebook 05 per-image AP. Run from the repo root: python scripts/bootstrap_ci.py results/05
import pandas as pd, numpy as np, sys
from scipy.stats import spearmanr
A = "results/05/"
pi = pd.read_csv(A + "per_image_ap.csv"); sw = pd.read_csv(A + "detection_sweep.csv")
ids = sorted(pi.image_id.unique()); n = len(ids); assert n == 300
B = 10000; rng = np.random.default_rng(12345); IDX = rng.integers(0, n, size=(B, n))
def v(d, s, r): return pi[(pi.degradation == d) & (pi.severity == s) & (pi.restorer == r)].set_index("image_id").loc[ids, "ap"].to_numpy()
clean = v("clean", 0.0, "none")
conds = [("gaussian", 25.0), ("gaussian", 50.0), ("jpeg", 10.0), ("low_light", 0.8)]
runs = []
for d, s in conds:
    for r in sorted(pi[(pi.degradation == d) & (pi.severity == s)].restorer.unique()):
        x = v(d, s, r); bs = x[IDX].mean(1)
        m = sw[(sw.degradation == d) & (sw.severity == s) & (sw.restorer == r)]["map"]
        runs.append(dict(cond=f"{d} {s:g}", restorer=r, coco_map=float(m.iloc[0]), img_avg_ap=x.mean(),
                         ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5)))
runs = pd.DataFrame(runs)
print(f"image-averaged AP vs COCO mAP over {len(runs)} runs: Spearman {spearmanr(runs.img_avg_ap, runs.coco_map).correlation:.3f}, "
      f"Pearson {np.corrcoef(runs.img_avg_ap, runs.coco_map)[0,1]:.3f}")
print(runs.round(3).to_string(index=False))

rows = []
for d, s in conds:
    none = v(d, s, "none")
    rs = sorted(pi[(pi.degradation == d) & (pi.severity == s)].restorer.unique())
    for i, a in enumerate(rs):
        for b in rs[i+1:]:
            for (x, y) in [(a, b)]:
                dif = v(d, s, x) - v(d, s, y); bs = dif[IDX].mean(1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                rows.append(dict(cond=f"{d} {s:g}", a=x, b=y, diff=dif.mean(), ci_low=lo, ci_high=hi,
                                 a_wins=(dif > 1e-9).mean(), ties=(np.abs(dif) <= 1e-9).mean(), b_wins=(dif < -1e-9).mean(),
                                 clear=bool(lo > 0 or hi < 0)))
    # share of lost AP recovered, with CI (ratio of means on the same resample)
    for r in rs:
        if r == "none": continue
        x = v(d, s, r)
        bs = (x[IDX].mean(1) - none[IDX].mean(1)) / (clean[IDX].mean(1) - none[IDX].mean(1))
        pt = (x.mean() - none.mean()) / (clean.mean() - none.mean())
        rows.append(dict(cond=f"{d} {s:g}", a=r, b="recovered%", diff=100*pt, ci_low=100*np.percentile(bs, 2.5), ci_high=100*np.percentile(bs, 97.5)))
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
runs.to_csv(sys.argv[1] + "/ci_runs.csv", index=False); out.to_csv(sys.argv[1] + "/ci_pairs.csv", index=False)

# PSNR vs AP within each condition, per image: does a restorer's PSNR gain over 'none' predict its AP gain?
print("\nper-image: Spearman between PSNR gain and AP gain over no restoration")
for d, s in conds:
    none = pi[(pi.degradation == d) & (pi.severity == s) & (pi.restorer == "none")].set_index("image_id").loc[ids]
    for r in sorted(pi[(pi.degradation == d) & (pi.severity == s)].restorer.unique()):
        if r == "none": continue
        x = pi[(pi.degradation == d) & (pi.severity == s) & (pi.restorer == r)].set_index("image_id").loc[ids]
        print(f"  {d} {s:g} {r:9s}: rho = {spearmanr(x.psnr - none.psnr, x.ap - none.ap).correlation:+.2f}")
