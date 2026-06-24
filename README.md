# DRISHTI-CORE Anti-UAV Detector

DRISHTI-CORE is a detector-only implementation of:

**DRISHTI: Motion-Guided Sparse Temporal Mixture-of-Experts for Tiny UAV Detection**

The current codebase is scoped to the Anti-UAV dataset. It stops at bounding-box
and confidence prediction. Swarm behavior classification, graph reasoning,
tracking memory, and threat scoring are intentionally out of scope until a
swarm-labeled dataset is available.

## Current Pipeline

```text
3 RGB frames [t-1, t, t+1]
  -> MotionCNN
  -> top-k crop proposal
  -> frozen LocateAnything-compatible crop encoder
  -> motion score concatenation
  -> 5-step temporal fusion transformer
  -> sparse top-2 MoE
  -> detection head
  -> [x, y, w, h, obj, conf]
```

## Stages

- Stage 1: `MotionCropProposal`
  - input `[B, 9, H, W]`
  - MotionCNN `9 -> 32 -> 64 -> 64 -> 1`
  - output heatmap plus `K=8` crop windows
- Stage 2: top-k crop proposal
  - crop output `[B*8, 3, 64, 64]`
  - motion scores `[B*8, 1]`
- Stage 3: `FrozenCropEncoder`
  - frozen local stand-in for LocateAnything
  - crop embedding `[B*8, 256]`
  - motion-augmented feature `[B*8, 257]`
- Stage 4: `TemporalFusion`
  - input `[B, 5, 8, 257]`
  - transformer with `d_model=257`, `nhead=4`, `layers=2`
  - output `[B, 8, 256]`
- Stage 5: `DRISHTIMoE`
  - 8 experts
  - top-2 routing
  - each expert: `Linear(256 -> 512) -> ReLU -> Linear(512 -> 256)`
- Stage 6: `DRISHTIDetectionHead`
  - output `[x, y, w, h, obj, conf]`

## Training Stages

Use `configure_drishti_training_stage(model, stage)` for procedure-aligned
freezing:

```python
configure_drishti_training_stage(model, "detector")  # MotionCNN + detection head
configure_drishti_training_stage(model, "temporal")  # temporal fusion only
configure_drishti_training_stage(model, "moe")       # MoE router + experts only
configure_drishti_training_stage(model, "all")       # smoke/debug end-to-end training
```

Checkpoint names:

```text
detector_best.pt
temporal_best.pt
moe_best.pt
```

## Quick Check

```powershell
python -m pytest -q
python experiment.py --smoke --epochs 1 --device cpu
```

The smoke experiment uses synthetic Anti-UAV-style moving boxes so the pipeline
can be verified without downloading data.

## Faster Full Training

Raw `.mp4` training is slow because each batch must seek and decode video
frames. For full training, extract frames once and train from image files:

```bash
python -m train.extract_frames \
  --data-root "/home/newuser1/dataset" \
  --output-root "/home/newuser1/dataset_frames" \
  --splits train val test \
  --modalities visible \
  --workers 4
```

This writes:

```text
/home/newuser1/dataset_frames/
  train/<video_id>/visible/000000.jpg
  train/<video_id>/visible.json
  val/<video_id>/visible/000000.jpg
  val/<video_id>/visible.json
```

Then train with `--frames-root`:

```bash
python experiment.py \
  --full \
  --frames-root "/home/newuser1/dataset_frames" \
  --train-split train \
  --val-split val \
  --modality visible \
  --stage detector \
  --epochs 80 \
  --batch-size 8 \
  --num-workers 4 \
  --height 448 \
  --width 448 \
  --image-channels 3 \
  --crop-size 64 \
  --num-crops 8 \
  --feature-dim 256 \
  --clip-stride 16 \
  --device cuda \
  --results-dir results_detector
```

## Anti-UAV Data

For extracted Anti-UAV-RGBT videos:

```powershell
python experiment.py `
  --full `
  --data-root "D:\Anti-UAV-RGBT" `
  --train-split train `
  --val-split test `
  --modality visible `
  --stage detector `
  --height 448 `
  --width 448 `
  --crop-size 64 `
  --num-crops 8 `
  --feature-dim 256 `
  --device cuda
```

For COCO-format Anti-UAV frames:

```powershell
python experiment.py `
  --full `
  --train-image-root D:\AntiUAV\train2017 `
  --train-ann-file D:\AntiUAV\annotations_train.json `
  --val-image-root D:\AntiUAV\val2017 `
  --val-ann-file D:\AntiUAV\annotations_val.json `
  --stage detector `
  --height 448 `
  --width 448 `
  --device cuda
```

## Project Layout

```text
models/
  drishti/
    config.py
    types.py
    motion_proposal.py
    crop_encoder.py
    temporal_fusion.py
    moe.py
    detection_head.py
    pipeline.py
train/
  antiuav.py
  drishti_loss.py
tests/
  test_modules.py
experiment.py
procedure.md
```
