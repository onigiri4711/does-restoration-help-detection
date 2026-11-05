# Data

No damaged photographs were collected: every damaged image is generated from a clean image by the seeded functions
in `stage1_degrade_restore_detect/degradations.py` (an identical copy is `stage2_3…/rftm/degradations.py`).

## Datasets

| Dataset | What it is | Where to get it | How it is used |
|---|---|---|---|
| **BSDS500** (Arbeláez et al., 2011) | 500 natural photos: 200 train, 100 val, 200 test | Kaggle: [balraj98/berkeley-segmentation-dataset-500-bsds500](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500) | The 400 train + test images train DnCNN (stage 1). The 100 val images are held out: 4 tune the classical filters, 96 measure restoration (PSNR, SSIM). They contain the CBSD68 benchmark images. |
| **COCO val2017** (Lin et al., 2014) | 5,000 photos with box labels for 80 object classes | [images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip) and [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) (the notebooks download both) | A fixed random 300-image subset (2,035 objects) is the detection test set for all three stages. 4,700 of the remaining images train (4,200) and validate (500) the stage 3 denoisers. |

## Fixed image lists

| File | Contents |
|---|---|
| `stage2_3_restoring_for_the_machine/prior/05/subset_ids.json` | COCO image ids of the 300-image test subset (used by notebooks 05–08) |
| `stage2_3_restoring_for_the_machine/results/08/splits.json` | stage 3 `train` / `val` / `test` image ids |

DnCNN's training images do not overlap the 96 BSD500 test images (checked by file name and pixel hash in notebook
04), and the stage 3 training images exclude the 300 test images.

## Models and weights

| Model | Source | File |
|---|---|---|
| DnCNN (17 layers, 558,403 parameters), trained from scratch | notebook 04 | `stage1_degrade_restore_detect/results/04/dncnn_last.pt`; also Kaggle dataset [shourjabanerjee/dcnn-last](https://www.kaggle.com/datasets/shourjabanerjee/dcnn-last) |
| Fine-tuned DnCNNs (λ = 0, 0.03, 0.1, 0.3, 1, 3 and the neck variant) | notebook 08 | `stage2_3_restoring_for_the_machine/results/08/checkpoints/*.pt` |
| YOLOv8n, YOLOv8s (COCO weights) | Ultralytics, downloaded automatically | – |
| Faster R-CNN ResNet-50-FPN v2 (COCO weights) | torchvision, downloaded automatically | – |
