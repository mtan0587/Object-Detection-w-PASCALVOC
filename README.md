# ELEC5304 — Assignment 3: PASCAL VOC 2007 Object Detection

This project fine-tunes **YOLOv8m** on the PASCAL VOC 2007 benchmark dataset to perform
multi-class object detection across 20 everyday categories. The goal is to train a model from
COCO-pretrained weights using a two-phase freeze/unfreeze strategy, evaluate it with mAP@0.5,
and analyse where and why the model fails.

**Final result: mAP@0.5 = 0.866** on the VOC 2007 test set (~35 min training on consumer hardware).

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Hardware and Environment](#2-hardware-and-environment)
3. [Requirements and Installation](#3-requirements-and-installation)
4. [The Dataset — PASCAL VOC 2007](#4-the-dataset--pascal-voc-2007)
5. [Model Architecture — YOLOv8m](#5-model-architecture--yolov8m)
6. [Training Strategy](#6-training-strategy)
7. [Directory Structure](#7-directory-structure)
8. [How to Run](#8-how-to-run)
9. [Results](#9-results)
10. [Troubleshooting](#10-troubleshooting)
11. [References](#11-references)

---

## 1. What This Project Does

Object detection is the task of simultaneously locating and classifying all objects in an image —
outputting a bounding box and class label for each instance. This is harder than image
classification (single label, no localisation) and harder than semantic segmentation (per-pixel
labels, no instance separation).

This project uses **transfer learning**: rather than training from random weights (which would
require hundreds of thousands of images and days of compute), we start from a model already
trained on MS-COCO and adapt it to the 20-class VOC vocabulary. We customise:

- **The backbone freeze schedule** — the backbone is locked for the first 15 epochs so only the
  detection head adapts, preventing the pretrained features from being destroyed early in training.
- **The optimiser and LR schedule** — AdamW with cosine decay and a 3-epoch warmup.
- **The augmentation pipeline** — mosaic, mixup, copy-paste, and HSV jitter specifically tuned
  for VOC's small training set (2,501 images).
- **The loss weights** — box/cls/dfl weights adjusted for VOC's 20-class structure vs COCO's 80.

The completed notebook (`Assignment_3-1.ipynb`) covers the full pipeline end-to-end: data
preparation, training, evaluation, visualisation, and written error analysis.

---

## 2. Hardware and Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM | 6 GB GDDR6 |
| CPU | (Intel/AMD laptop, 4 WSL2 processors allocated) |
| RAM (WSL2) | 6 GB (capped via `.wslconfig`) |
| OS | Ubuntu on WSL2 (Windows 11) |
| CUDA | 11.8 |
| Python | 3.12 |

> The 6 GB VRAM is a hard constraint. `batch=16` at `imgsz=512` fits comfortably. Increasing
> to `imgsz=640` or `batch=32` will cause OOM. The `.wslconfig` memory cap prevents the WSL2
> kernel from crashing when PyTorch pre-allocates GPU memory alongside system RAM.

---

## 3. Requirements and Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / WSL
# .venv\Scripts\activate           # Windows PowerShell

# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Ultralytics (YOLOv8 framework) and notebook dependencies
pip install ultralytics matplotlib pandas pillow jupyter
```

**WSL2 memory cap (required before first training run):**

Create or edit `%USERPROFILE%\.wslconfig` in Windows with:
```ini
[wsl2]
memory=6GB
processors=4
swap=4GB
```
Then run `wsl --shutdown` in PowerShell and reopen WSL. Without this, the WSL2 kernel will
consume all available system RAM and crash mid-training. Cell 24 in the notebook creates this
file automatically if it does not already exist.

---

## 4. The Dataset — PASCAL VOC 2007

PASCAL VOC 2007 (Visual Object Classes Challenge 2007) is one of the most widely used benchmarks
in object detection research. It consists of real-world photographs scraped from Flickr, manually
annotated with bounding boxes and class labels.

### Statistics

| Split | Images | Annotated objects |
|---|---|---|
| Train | 2,501 | ~6,300 |
| Val | 2,510 | ~6,400 |
| Test | 4,952 | ~12,000 |
| **Total** | **9,963** | **~24,700** |

### The 20 Classes

```
aeroplane   bicycle    bird       boat       bottle
bus         car        cat        chair      cow
diningtable dog        horse      motorbike  person
pottedplant sheep      sofa       train      tvmonitor
```

These span a wide range of scales, aspect ratios, and frequencies in the dataset. `person` is by
far the most common class (~4,500 test instances); `sheep` has fewer than 250. This imbalance
affects per-class AP and is one of the reasons harder classes like `bottle` and `pottedplant`
score lower.

### Annotation Format

Each image has a corresponding XML file in `Annotations/`. The notebook parses these and converts
them to YOLO normalised-centre format (`cx cy w h`, all divided by image width/height) stored in
`VOCdevkit/VOC2007/labels/`. Difficult objects (flagged in the XML with `<difficult>1</difficult>`)
are skipped — 5,998 such objects are excluded from training and evaluation.

### How the Data is Sourced

The notebook downloads the data automatically (cells 5–6) from the official Oxford servers:
- `VOCtrainval_06-Nov-2007.tar` (~450 MB) — train + val images and annotations
- `VOCtest_06-Nov-2007.tar`     (~430 MB) — test images and annotations

These are extracted to `VOCdevkit/VOC2007/`. A symlink directory `voc_dataset/` is then built
(cell 26) into the structure Ultralytics expects (`train/images/`, `val/images/`, `test/images/`).

---

## 5. Model Architecture — YOLOv8m

We chose **YOLOv8m** (You Only Look Once v8, medium variant) for its balance of accuracy and
speed on 6 GB VRAM. YOLO-family models are **single-stage detectors**: they process the entire
image in one forward pass and directly predict bounding boxes and class probabilities — making
them significantly faster than two-stage detectors like Faster R-CNN.

| Variant | Params | COCO mAP@0.5 | Why not used |
|---|---|---|---|
| YOLOv8n | 3.2 M | 37.3 | Too small for 20 diverse VOC classes |
| YOLOv8s | 11.2 M | 44.9 | Moderate capacity |
| **YOLOv8m** | **25.9 M** | **50.2** | ✅ Best accuracy/speed on RTX 4050 |
| YOLOv8l | 43.7 M | 52.9 | OOM risk at batch 16 on 6 GB VRAM |

### Architecture Diagram

```
                    ┌─────────────────────────────────────────────────────────┐
                    │            INPUT  512 × 512 × 3  (RGB)                  │
                    └────────────────────────┬────────────────────────────────┘
                                             │
          ╔══════════════════════════════════▼═══════════════════════════════════╗
          ║              BACKBONE — CSPDarknet  (layers 0 – 9)                   ║
          ║                                                                      ║
          ║  Conv(3×3, s=2) → 256×256×48                                         ║
          ║       │                                                              ║
          ║  Conv(3×3, s=2) → 128×128×96                                         ║
          ║       │                                                              ║
          ║  C2f  (2 bottlenecks) ──────────────────────────────► P3  64×64×192  ║
          ║       │                                                    (stride 8)║
          ║  Conv(3×3, s=2) → 64×64×192                                          ║
          ║       │                                                              ║
          ║  C2f  (4 bottlenecks) ──────────────────────────────► P4  32×32×384  ║
          ║       │                                                    (stride16)║
          ║  Conv(3×3, s=2) → 32×32×384                                          ║
          ║       │                                                              ║
          ║  C2f  (4 bottlenecks)                                                ║
          ║       │                                                              ║
          ║  SPPF (5×5 MaxPool ×3 concat) ─────────────────────► P5  16×16×576   ║
          ║                                                            (stride32)║
          ╚══════════════════════════════════════════════════════════════════════╝
                     P3 │            P4 │            P5 │
                        │               │               │
          ╔═════════════▼═══════════════▼═══════════════▼════════════════════════╗
          ║              NECK — PAN-FPN  (layers 10 – 21)                        ║
          ║                                                                      ║
          ║  FPN (top-down):  propagates deep semantics to fine-resolution maps  ║
          ║    P5 ──Upsample──► concat(P4) ──C2f──► P4'  32×32                   ║
          ║    P4'──Upsample──► concat(P3) ──C2f──► P3'  64×64                   ║
          ║                                                                      ║
          ║  PAN (bottom-up): propagates fine spatial detail to coarse maps      ║
          ║    P3'──Conv(s=2)──► concat(P4')──C2f──► P4'' 32×32                  ║
          ║    P4''─Conv(s=2)──► concat(P5) ──C2f──► P5'' 16×16                  ║
          ╚══════════════════════════════════════════════════════════════════════╝
                     P3' │           P4'' │           P5'' │
                  64×64×192       32×32×384        16×16×576
                         │                │                │
          ╔═════════════▼═══════════════▼═══════════════▼═══════════════════════╗
          ║         HEAD — Decoupled Anchor-Free  (layer 22)                    ║
          ║                                                                     ║
          ║  Branch S (stride  8, 64×64):  detects small objects   (< 32 px)   ║
          ║  Branch M (stride 16, 32×32):  detects medium objects  (32–96 px)  ║
          ║  Branch L (stride 32, 16×16):  detects large objects   (> 96 px)   ║
          ║                                                                     ║
          ║  Each branch:                                                       ║
          ║    ┌─ Box branch  (DFL) ─► Δ(cx, cy, w, h) per cell                ║
          ║    └─ Cls branch  (BCE) ─► 20 class scores per cell                ║
          ╚═════════════════════════════════════════════════════════════════════╝
                                       │
                    ┌──────────────────▼─────────────────────┐
                    │  NMS  (conf > 0.25,  IoU > 0.45)       │
                    └──────────────────┬─────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────┐
                    │  Final detections: class, bbox, score  │
                    └────────────────────────────────────────┘
```

### Key Components Explained

**Backbone (CSPDarknet, layers 0–9):** Extracts hierarchical visual features through a series of
strided convolutions and C2f (Cross-Stage-Partial with 2 bottlenecks) blocks. Each stride-2
convolution halves spatial resolution while doubling channel depth, building up increasingly
abstract feature representations. SPPF (Spatial Pyramid Pooling-Fast) at the end captures
multi-scale context within the deepest feature map.

**Neck (PAN-FPN, layers 10–21):** Fuses features from all three backbone scales. The FPN
(Feature Pyramid Network) top-down path upsamples deep coarse features and merges them with
shallower fine-resolution features, giving smaller feature maps access to high-level semantic
information. The PAN (Path Aggregation Network) bottom-up path then re-propagates fine spatial
details back up, resulting in a bidirectional fusion that benefits both small and large object
detection.

**Head (Decoupled Anchor-Free, layer 22):** Three parallel detection branches, one per scale.
Each branch uses separate sub-networks for box regression and class prediction (decoupled head).
Box coordinates are predicted via DFL (Distribution Focal Loss), which models the box boundary
as a probability distribution rather than a single point — producing more accurate and calibrated
boxes without hand-crafted anchor priors.

### YOLOv8m vs Its Predecessors

| Feature | YOLOv5 | **YOLOv8m (ours)** |
|---|---|---|
| Detection head | Anchor-based (9 anchors) | **Anchor-free (DFL)** |
| Backbone blocks | CSP bottleneck | **C2f (improved gradient flow)** |
| Box loss | GIoU | **CIoU + DFL** |
| Class loss | BCEWithLogits | **Sigmoid BCE, decoupled** |
| Head design | Coupled cls+box | **Decoupled cls/box** |
| Default optimiser | SGD | **AdamW (our choice)** |

---

## 6. Training Strategy

### Why Transfer Learning?

VOC 2007 has only 2,501 training images. Training YOLOv8m (25.9 M parameters) from random
weights on a dataset this small would require 200–300+ epochs and would almost certainly
overfit. Instead, we start from **COCO-pretrained weights** (`yolov8m.pt`): the model already
understands edges, textures, and shapes from 118,000 COCO images. All 20 VOC classes are
subsets of COCO's 80 classes, so the pretrained feature representations are highly relevant.

### Phase 1 — Frozen Backbone (Epochs 1–15)

In the first phase, **backbone layers 0–9 are frozen** (no gradient updates). Only the neck and
detection head are trained. This is important because:

- The head's weights are initialised for COCO's 80 classes and need to be re-mapped to VOC's 20.
  Allowing the backbone to update before the head stabilises can corrupt the pretrained features.
- A high learning rate (0.001) can be used safely on the head without risking the backbone.
- A 3-epoch warmup linearly ramps the LR from near-zero to 0.001, preventing large gradient
  spikes at the start of training.

> **Implementation note:** Ultralytics' built-in `freeze=N` incorrectly freezes the DFL layer,
> which causes a gradient error during training. The notebook uses a custom `on_train_start`
> callback that manually freezes layers 0–9 while explicitly leaving DFL trainable.

### Phase 2 — Full Fine-Tune (Epochs 16–30)

All layers are unfrozen and the entire network is fine-tuned at **10× lower learning rate**
(0.0001). The lower LR prevents the now-stable head from overwriting the pretrained backbone
features with large gradient updates.

### Summary of Training Configuration

| Setting | Phase 1 | Phase 2 |
|---|---|---|
| Epochs | 15 | 15 |
| Frozen layers | Backbone (0–9) | None |
| Learning rate | 0.001 | 0.0001 |
| LR schedule | Cosine decay | Cosine decay |
| Warmup epochs | 3 | 0 |
| Optimiser | AdamW | AdamW |
| Batch size | 16 | 16 |
| Input resolution | 512 × 512 | 512 × 512 |
| Workers | 2 | 2 |

### Data Augmentation

With only 2,501 training images, augmentation is critical for preventing overfitting:

| Augmentation | Setting | Purpose |
|---|---|---|
| **Mosaic** | 1.0 (always on) | Tiles 4 images per sample — effectively 4× scene diversity and forces small-object detection |
| **Mixup** | 0.1 | Blends two image/label pairs — reduces overconfidence and smooths decision boundaries |
| **Copy-Paste** | 0.1 | Pastes object instances across images — boosts rare classes like `sheep` and `boat` |
| **HSV Jitter** | h=0.015, s=0.7, v=0.4 | Randomises hue, saturation, and brightness — forces the model to rely on shape/texture rather than colour |
| **Horizontal Flip** | 0.5 | Doubles orientation diversity at zero cost |
| **Dropout (head)** | 0.1 | Randomly drops neurons in the detection head — additional regularisation |

### Loss Function

| Term | Weight | What it penalises |
|---|---|---|
| `box` (CIoU) | 8.0 | Inaccurate bounding box position, size, and aspect ratio |
| `cls` (BCE) | 0.4 | Wrong class prediction (reduced from default 0.5 — VOC's 20 classes is simpler than COCO's 80) |
| `dfl` (Distribution Focal Loss) | 1.5 | Imprecise box boundary distributions |

---

## 7. Directory Structure

```
Obj_Detection_PascalVOC/
│
├── Assignment_3-1.ipynb        # Main deliverable — all code, training, analysis
├── README.md                   # This file
├── .gitignore
│
├── VOCdevkit/                  # [git-ignored] Auto-downloaded raw dataset
│   └── VOC2007/
│       ├── JPEGImages/         # 9,963 JPEG images
│       ├── Annotations/        # XML bounding-box annotations (one per image)
│       ├── ImageSets/Main/     # train.txt / val.txt / test.txt split files
│       └── labels/             # YOLO-format .txt labels (generated by notebook cell 25)
│
├── voc_dataset/                # [git-ignored] Symlink tree consumed by Ultralytics
│   ├── train/images/ + labels/ # Symlinks → VOCdevkit train split
│   ├── val/images/   + labels/ # Symlinks → VOCdevkit val split
│   └── test/images/  + labels/ # Symlinks → VOCdevkit test split
│
├── voc.yaml                    # [git-ignored] Ultralytics dataset config (absolute paths —
│                               #   regenerate on each machine by re-running cell 26)
│
├── yolov8m.pt                  # [git-ignored] COCO-pretrained starting weights (50 MB)
│
├── VOCtrainval_06-Nov-2007.tar # [git-ignored] Downloaded archive (~450 MB)
├── VOCtest_06-Nov-2007.tar     # [git-ignored] Downloaded archive (~430 MB)
│
└── runs/
    └── voc_yolo/
        ├── phase1_frozen/      # Phase 1 training output
        │   ├── weights/
        │   │   ├── best.pt     # Best Phase 1 checkpoint — mAP@0.5 = 0.866 ← used for eval
        │   │   └── last.pt     # Final epoch checkpoint
        │   ├── results.csv     # Per-epoch metrics (loss, mAP, P, R)
        │   ├── results.png     # Training curves plot
        │   ├── confusion_matrix.png
        │   └── args.yaml       # Full training config snapshot
        └── phase2_finetune/    # Phase 2 training output (same structure)
```

> `runs/`, `voc_dataset/`, `voc.yaml`, `*.pt`, and `*.tar` files are all git-ignored.
> They are either too large for git (model weights are 50–100 MB each) or are fully
> reproducible by running the notebook from top to bottom.

---

## 8. How to Run

Open `Assignment_3-1.ipynb` in VS Code with the Jupyter extension, or run:

```bash
source .venv/bin/activate
jupyter notebook Assignment_3-1.ipynb
```

Run cells sequentially from top to bottom. Each group is labelled with a `# STEP N` comment.

| Cell(s) | Step | What it does |
|---|---|---|
| 1 | — | Imports all libraries (consolidated) |
| 5 | 1 | Downloads VOC 2007 tarballs and extracts `VOCdevkit/` — skips if already present |
| 7–8 | 2 | Checks and documents the dataset folder structure |
| 10 | 3 | Defines paths to images, annotations, and split files |
| 12 | 4 | Loads official train / val / test image IDs |
| 14 | 5 | Defines the 20 VOC class names |
| 16 | 6 | `read_voc_annotation(image_id)` — parses XML → list of `{class_name, bbox}` dicts |
| 18–20 | 7–8 | Visualises 2 random training samples with GT bounding boxes |
| 24 | — | WSL2 stability: writes `.wslconfig`, kills stale TensorBoard processes |
| 25 | 2 | Converts all VOC XML → YOLO `.txt` label files in `VOCdevkit/VOC2007/labels/` |
| 26 | 3 | Builds `voc_dataset/` symlink tree and writes `voc.yaml` |
| 27 | 4 | Sanity-checks one image/label pair |
| 28 | 5 | Loads `yolov8m.pt` (auto-downloaded by Ultralytics if missing) |
| 29 | 6 | **Phase 1 training** — backbone frozen, trains neck + head for 15 epochs (~15 min) |
| 30 | 7 | **Phase 2 training** — all layers unfrozen, 15 more epochs at lower LR (~20 min) |
| 31 | 8 | Evaluates best checkpoint on test set — prints mAP and per-class AP@0.5 |
| 32 | — | Side-by-side GT vs prediction visualisation on one random test image |
| 33–34 | — | Task 1 write-up: architecture design, strategy, results summary (markdown) |
| 36–37 | 2.1 | Per-class AP@0.5 ranked table and bar chart |
| 38–40 | 2.2 | 10 worst failure cases (5×2 grid) with per-image error type classification |
| 41 | 2.3 | Error discussion with ELEC5304 DIP terminology |

**Total runtime:** ~35 minutes on an RTX 4050 Laptop 6 GB.

---

## 9. Results

### Overall Test Set Performance

| Checkpoint | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Wall time |
|---|---|---|---|---|---|
| Phase 1 best (epoch 15) | **0.866** | 0.648 | 0.835 | 0.800 | ~15 min |
| Phase 2 best (epoch 14) | 0.847 | 0.637 | 0.837 | 0.774 | ~20 min |

Phase 1 outperforms Phase 2. With only 15 Phase 1 epochs the detection head had not fully
converged; the 10× lower LR in Phase 2 produced insufficient gradient updates to improve it.
The Phase 1 checkpoint is therefore used for all evaluation.

### Per-Class AP@0.5

| Rank | Class | AP@0.5 |
|---|---|---|
| 1 | horse | 0.943 |
| 2 | car | 0.934 |
| 3 | cow | 0.920 |
| 4 | bus | 0.912 |
| 5 | motorbike | 0.915 |
| … | … | … |
| 18 | chair | 0.713 |
| 19 | boat | 0.700 |
| 20 | pottedplant | 0.586 |

Large, distinct objects (horse, car, bus) score highest. Small, frequently-occluded objects
(bottle, pottedplant, boat) score lowest — consistent with the stride-8 aliasing limitation
discussed in the notebook's Section 2.3 DIP analysis.

---

## 10. Troubleshooting

**Phase 2 `FileNotFoundError` on Phase 1 weights**
Cell 30 uses `Path('runs').rglob('phase1_frozen/weights/best.pt')` to search recursively,
working around the Ultralytics path-nesting bug (where `project='runs/voc_yolo'` sometimes
saves to `runs/detect/runs/voc_yolo/` instead). If Phase 1 has not been run, execute cell 29
first.

**WSL2 kernel crash during training**
Caused by OOM or TensorBoard port conflict. Cells 24, 29, and 30 all run
`pkill -f tensorboard` before training (port-agnostic — TensorBoard may grab 6006/6007/6008).
If crashes persist, confirm `.wslconfig` has `memory=6GB` and restart WSL (`wsl --shutdown`
in PowerShell).

**CUDA out of memory**
Reduce `batch` from 16 to 8 in cells 29 and 30.

**`voc.yaml` has wrong absolute paths on a different machine**
Re-run cell 26 to regenerate `voc.yaml` with the correct absolute paths for the current machine.

**`YOLO` model not found / ultralytics import error**
Activate the virtual environment (`source .venv/bin/activate`) and run
`pip install ultralytics` again.

---

## 11. References

1. **Redmon, J. et al. (2016).** You Only Look Once: Unified, Real-Time Object Detection.
   *IEEE CVPR 2016.* https://arxiv.org/abs/1506.02640

2. **Jocher, G. et al. (2023).** Ultralytics YOLOv8.
   GitHub: https://github.com/ultralytics/ultralytics

3. **Everingham, M. et al. (2010).** The PASCAL Visual Object Classes (VOC) Challenge.
   *International Journal of Computer Vision, 88(2), 303–338.*
   https://link.springer.com/article/10.1007/s11263-009-0275-4

4. **Lin, T.-Y. et al. (2017).** Feature Pyramid Networks for Object Detection.
   *IEEE CVPR 2017.* https://arxiv.org/abs/1612.03144

5. **Liu, S. et al. (2018).** Path Aggregation Network for Instance Segmentation.
   *IEEE CVPR 2018.* https://arxiv.org/abs/1803.01534

6. **He, K. et al. (2016).** Deep Residual Learning for Image Recognition.
   *IEEE CVPR 2016.* https://arxiv.org/abs/1512.03385

7. **Wang, C.-Y. et al. (2020).** CSPNet: A New Backbone that can Enhance Learning Capability
   of CNN. *IEEE CVPR Workshops 2020.* https://arxiv.org/abs/1911.11929

8. **Li, X. et al. (2020).** Generalized Focal Loss: Learning Qualified and Distributed
   Bounding Boxes for Dense Object Detection (DFL). *NeurIPS 2020.*
   https://arxiv.org/abs/2006.04388

9. **Rezatofighi, H. et al. (2019).** Generalized Intersection over Union (GIoU).
   *IEEE CVPR 2019.* https://arxiv.org/abs/1902.09630

10. **Loshchilov, I. & Hutter, F. (2019).** Decoupled Weight Decay Regularization (AdamW).
    *ICLR 2019.* https://arxiv.org/abs/1711.05101

11. **Gonzalez, R.C. & Woods, R.E. (2018).** *Digital Image Processing* (4th ed.).
    Pearson. — Sampling theorem, spatial frequency, morphological operations, image pyramids.
