# Does Restoration Really Help Detection? A Deep Dive

**Course project, Image Processing and Applications (Section IPA 9)** · Shourja Banerjee · Roll No. 24155423

Image restoration is usually judged by PSNR, but restored images are increasingly read by object detectors rather than
people. This project asks whether restoration helps a pretrained detector (YOLOv8n), whether PSNR picks the restorer
that helps most, and what happens to the damage inside the detector. It has three stages:

| Stage | Folder | Notebooks | Question |
|---|---|---|---|
| **1 · Degrade → Restore → Detect** | [`stage1_degrade_restore_detect/`](stage1_degrade_restore_detect) | 01–06, 05b | Does restoration help YOLOv8n under Gaussian noise, JPEG and low light, and does PSNR pick the best restorer? |
| **2 · Inside the detector** | [`stage2_3_restoring_for_the_machine/`](stage2_3_restoring_for_the_machine) | 07 | Where does the damage enter YOLOv8n's features, and how much of it do restorers repair? |
| **3 · Denoiser trained on detector features** | [`stage2_3_restoring_for_the_machine/`](stage2_3_restoring_for_the_machine) | 08 | Does fine-tuning DnCNN with a YOLO feature loss improve detection, also on detectors never used in training? |

📄 **Report:** [`report/Report_IPA.pdf`](report/Report_IPA.pdf) (Word version: [`report/Report_IPA.docx`](report/Report_IPA.docx))

## Key results

| | Result | Saved in |
|---|---|---|
| Clean YOLOv8n mAP@[.5:.95] on the 300-image COCO subset | 0.387 | `stage1…/results/05b/sweep.csv` |
| DnCNN on Gaussian noise, σ = 25 / 50 | recovers 77% / 72% of the lost mAP; BM3D is statistically tied | `stage1…/results/05b/` |
| Low light, ℓ = 0.8 | gamma → DnCNN 0.242 mAP vs DnCNN → gamma 0.196, although the second has higher PSNR | `stage1…/results/05b/sweep.csv` |
| PSNR vs mAP over 64 restored runs | Spearman 0.76 overall, but 63 of 144 low-light restorer pairs reversed | `stage1…/results/05b/sweep.csv` |
| Inside YOLOv8n (stage 2) | DnCNN removes 82–93% of the pixel error but only 58–62% of the neck feature damage | `stage2_3…/results/07/` |
| Feature-loss fine-tuning (stage 3), σ = 50 | YOLOv8n 0.302 (control) → 0.312 at λ* = 1 (chosen on validation) and 0.317 at λ = 0.3 | `stage2_3…/results/08/runs.csv` |

Pre-registered hypotheses: 5 for stage 1, 13 for stage 2 (8 supported), 4 for stage 3 (2 supported). Full wording is
in `stage1…/paper/paper.pdf` and [`stage2_3…/PLAN.md`](stage2_3_restoring_for_the_machine/PLAN.md).

## Check the numbers in 1 minute (CPU, no data needed)

```bash
pip install numpy pandas scipy
python scripts/verify_results.py
```

This recomputes the report's headline numbers from the saved result files and prints PASS/FAIL for each
(23 checks). Notebooks 01–06 and 05b are saved **with their outputs**, so their tables and figures can be inspected
without re-running anything. Notebooks 07 and 08 are saved without outputs; their results are in
`stage2_3…/results/07/` and `results/08/`, and the executed versions are on Kaggle (links below).

## Re-run everything

The notebooks were run on **Kaggle (GPU T4, Internet ON)**. Every run is seeded, so a re-run gives the same
outputs. Step-by-step instructions for each notebook (what to attach, which settings, how long it takes and which
file to compare) are in **[`REPRODUCE.md`](REPRODUCE.md)**; the datasets are listed in **[`DATA.md`](DATA.md)**.

### Kaggle notebooks

The executed Kaggle versions (with all outputs) of every notebook that needs a GPU or Kaggle data. Run times are
those of the saved Kaggle versions (the GPU runs used T4 x2).

| # | Notebook | Kaggle link | GPU | Run time on Kaggle |
|---|---|---|---|---|
| 01 | Degradations and baselines | [kaggle.com/code/shourjabanerjee/1-degradations](https://www.kaggle.com/code/shourjabanerjee/1-degradations) | no | 1 h 04 min |
| 02 | JPEG from scratch | not run on Kaggle (CPU only; run locally, see REPRODUCE.md) | no | – |
| 03 | NumPy convolution and gradient check | not run on Kaggle (CPU only; run locally, see REPRODUCE.md) | no | – |
| 04 | DnCNN training | [kaggle.com/code/shourjabanerjee/04-training-a-blind-dncnn](https://www.kaggle.com/code/shourjabanerjee/04-training-a-blind-dncnn) | T4 | 3 h 54 min |
| 05 | Detection evaluation | [kaggle.com/code/shourjabanerjee/05-does-restoration-help-detection](https://www.kaggle.com/code/shourjabanerjee/05-does-restoration-help-detection) | T4 | 19 min |
| 05b | Bootstrap CIs, BM3D, low-light pipelines | [kaggle.com/code/shourjabanerjee/notebook-05b](https://www.kaggle.com/code/shourjabanerjee/notebook-05b) | T4 | 8 h 51 min (`QUICK = True`: ~10 min) |
| 06 | Denoiser analysis | [kaggle.com/code/shourjabanerjee/notebook-06](https://www.kaggle.com/code/shourjabanerjee/notebook-06) | T4 | 1.5 min |
| 07 | Feature analysis (stage 2) | [kaggle.com/code/shourjabanerjee/notebook-7](https://www.kaggle.com/code/shourjabanerjee/notebook-7) | T4 | 4 h 07 min (`QUICK = True`: ~10 min) |
| 08 | Task-driven DnCNN (stage 3) | [kaggle.com/code/shourjabanerjee/notebook-8](https://www.kaggle.com/code/shourjabanerjee/notebook-8) | T4 | 6 h 23 min (`QUICK = True` first) |

### Datasets and weights

| What | Where | Used by |
|---|---|---|
| BSDS500 (500 images) | Kaggle: [balraj98/berkeley-segmentation-dataset-500-bsds500](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500) | 01–04, 05b, 06 |
| COCO val2017 (5,000 images + annotations) | downloaded by the notebooks from [cocodataset.org](https://cocodataset.org/#download) | 05–08 |
| Trained DnCNN checkpoint | `stage1…/results/04/dncnn_last.pt`, also on Kaggle: [shourjabanerjee/dcnn-last](https://www.kaggle.com/datasets/shourjabanerjee/dcnn-last) | 05–08 |
| YOLOv8n / YOLOv8s weights | downloaded automatically by `ultralytics` | 05–08 |
| Faster R-CNN R50-FPN v2 weights | downloaded automatically by `torchvision` | 08 |

## Repository layout

```
README.md                       this file
REPRODUCE.md                    how to re-run each notebook and compare its outputs
DATA.md                         datasets, the fixed 300-image subset and the stage 3 split
requirements.txt                Python packages for all notebooks
report/                         the course report (PDF and Word)
scripts/verify_results.py       CPU check of the report's numbers; --compare checks a fresh run
stage1_degrade_restore_detect/  stage 1: notebooks 01–06 and 05b, degradations.py, results/, figures/, paper/
stage2_3_restoring_for_the_machine/
                                stages 2 and 3: notebooks 07 and 08, the rftm/ package, PLAN.md
                                (hypotheses), prior/ (stage 1 results it builds on), results/07, results/08, tests/
```

Each stage folder keeps its own README with more detail. The two folders are copies of the original repositories
[onigiri4711/degrade-restore-detect](https://github.com/onigiri4711/degrade-restore-detect) and
[onigiri4711/restoring-for-the-machine](https://github.com/onigiri4711/restoring-for-the-machine).

## Use of AI assistance

An AI assistant (Claude, Anthropic) helped plan the studies, scaffold and debug the notebooks, write analysis code and
draft parts of the report. All experiments were run by the author on Kaggle. Details are in the AI-assistance section
at the end of the report.
