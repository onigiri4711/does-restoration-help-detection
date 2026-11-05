# Reproducing the results

Notebooks 01–06 and 05b are saved with their outputs; notebooks 07 and 08 are saved without outputs, but everything
they produced is in `stage2_3…/results/07/` and `results/08/`, and the executed versions are on Kaggle. So nothing has
to be re-run to check a number. This page is for re-running the notebooks and getting the same outputs.

## 0. Quick check (CPU, about a minute)

```bash
pip install numpy pandas scipy
python scripts/verify_results.py                       # 23 PASS/FAIL checks against the report
cd stage2_3_restoring_for_the_machine
python tests/test_analysis.py                          # stage 2 analysis code
python tests/test_analysis8.py                         # stage 3 analysis code
```

## 1. One-time setup on Kaggle

1. Upload this whole repository as a **Kaggle dataset** (Kaggle → Datasets → New Dataset → upload the folder, or the
   zip). Call it, for example, `ipa-restoration-detection-code`. It contains everything the notebooks look for:
   `degradations.py`, the tuned filter settings (`best_classical.json`), the DnCNN checkpoint (`dncnn_last.pt`), the
   fixed 300-image subset (`subset_ids.json`) and the `rftm/` package. The notebooks find these files anywhere under
   `/kaggle/input`, so the folder names do not matter.
2. For each notebook: **File → Import Notebook**, choose the `.ipynb` from this repository, then in the right-hand
   panel set **Accelerator = GPU T4** (notebooks 04–08; the saved runs used T4 x2) and **Internet = On**, and **Add Input** as listed below.
3. Run with **Save Version → Save & Run All (Commit)**. Outputs appear under `/kaggle/working/outputs/<notebook>/`
   (notebooks 01–04 write to `/kaggle/working/`). Download that folder.
4. Compare the new outputs with the saved ones:

   ```bash
   python scripts/verify_results.py --compare path/to/downloaded/outputs/05b
   ```

   It matches rows on their key columns (damage, severity, restorer, …) and prints the largest difference in every
   numeric column.

Notebooks with a `QUICK` switch in their first cells can be run with `QUICK = True` first (a few minutes, a subset of
images) to check that everything is attached correctly; the full results need `QUICK = False`.

## 2. Notebook by notebook

| # | Notebook (folder) | Add Input on Kaggle | Settings | Writes | Compare with |
|---|---|---|---|---|---|
| 01 | `01_degradations_and_baselines` (stage 1) | BSDS500 dataset | CPU | `baselines.csv`, `best_classical.json`, … | `stage1…/results/01/` |
| 02 | `02_jpeg_from_scratch` (stage 1) | BSDS500 dataset (or set `BSD_VAL` to a local copy of `images/val`) | CPU, runs on a laptop | figures, printed tables | outputs saved in the notebook |
| 03 | `03_numpy_conv_gradcheck` (stage 1) | BSDS500 dataset (or set `BSD_TRAIN` and `BSD_VAL`) | CPU, runs on a laptop; with PyTorch installed it adds a PyTorch comparison that the saved run skipped | printed gradient-check errors | outputs saved in the notebook |
| 04 | `04_dncnn_training` (stage 1) | none (downloads BSDS500 with `kagglehub`) | GPU T4, Internet on; `QUICK = False` | `dncnn_last.pt`, `dncnn_eval.csv`, `loss.png` | `stage1…/results/04/` |
| 05 | `05_detection_eval` (stage 1) | the code dataset from step 1 | GPU T4, Internet on | `outputs/05/` | `stage1…/results/05/` |
| 05b | `05b_quick_wins` (stage 1) | the code dataset + BSDS500 | GPU T4, Internet on; ~9 h (BM3D on CPU) | `outputs/05b/` | `stage1…/results/05b/` |
| 06 | `06_denoiser_analysis` (stage 1) | the code dataset + BSDS500 | GPU T4, Internet on | `outputs/06/` | `stage1…/results/06/` |
| 07 | `07_feature_analysis` (stage 2) | the code dataset (this is enough on its own: it contains the DnCNN checkpoint and the `rftm/` code) | GPU T4, Internet on; ~4 h | `outputs/07/` | `stage2_3…/results/07/` |
| 08 | `08_task_driven_dncnn` (stage 3) | same as 07 | GPU T4, Internet on; ~6.5 h | `outputs/08/` | `stage2_3…/results/08/` |

Order: 01 → 04 → 05 → 05b → 06 → 07 → 08 (02 and 03 are independent). Because the trained checkpoint and every
intermediate result are already in this repository, any single notebook from 05 onward can be re-run on its own.

## 3. What "the same outputs" means here

* **Damage is seeded per image.** Every damaged image is generated from a random generator seeded by
  (seed, image index, damage index, severity index), so a re-run produces the identical noisy, JPEG or low-light
  image, and every restorer sees the same input.
* **Fixed image sets.** The 300 COCO val2017 test images are listed in
  `stage2_3…/prior/05/subset_ids.json`; the stage 3 training and validation split is in
  `stage2_3…/results/08/splits.json`. The 100 BSD500 validation images are split 4 (filter tuning) / 96 (testing).
* **Built-in checks.** Notebook 05b re-computes notebook 05's shared runs, and notebook 07 checks the subset ids, the
  clean mAP (0.387) and the per-image AP of every shared run against the stage 1 results before it continues.
* **Expected differences.** Detection, BM3D and all analysis steps are deterministic. GPU training (notebooks 04 and
  08) uses fixed seeds, but cuDNN kernels are not bit-exact across GPU types or library versions, so a retrained
  model can differ in the third or fourth decimal of mAP. To reproduce the reported detection numbers exactly, use the
  saved checkpoints (`stage1…/results/04/dncnn_last.pt`, `stage2_3…/results/08/checkpoints/`). Newer releases of
  `ultralytics` or `torchvision` may also move mAP slightly; the runs used the Kaggle GPU image at the time (October–November 2026).
